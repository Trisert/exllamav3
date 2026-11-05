// fa60: flash-attention forward for Pascal (sm_60, GP100), head_dim 256, fp16 in/out, for the prefill path.
// No tensor cores, no cp.async, 48 KB of shared memory per block: everything runs on HFMA2 (twice the fp32 FMA
// rate on GP100). PyTorch's memory-efficient SDPA kernel for these shapes is a SIMT fp32 kernel.
//
// Layout and causal convention as fa75 (turing/fa75.cu): q [Tq, Hq, 256], k/v [Tkv, Hkv, 256] (row and head
// strides given, last dim contiguous), o [Tq, Hq, 256] contiguous; query i sees keys j <= i + Tkv - Tq.
// GQA: head h reads kv head h / (Hq / Hkv).
//
// Block = 64 query rows of one head, 256 threads, one block per SM. Q (64 x 256) stays in shared memory, K and V
// tiles of 16 keys pass through one shared buffer (the next tile's global loads are in flight in registers).
//  - S = Q K^T: each quad of lanes owns 4 rows x 4 keys and splits the head dim (lane dq takes the 16-byte chunks
//    c % 4 == dq), so one LDS.128 feeds 4 HFMA2 instead of 2 (on sm_60 an LDS.128 costs the shared-memory pipe
//    ~4 cycles per warp even when lanes broadcast, which bounded the old 2 x 2 tile). HFMA2 lanes hold two
//    consecutive head dims; fp16 partials of 8 terms per lane, lane-summed in fp16 (one HADD2.F32), go into fp32,
//    then a quad reduce-scatter (2 shuffle rounds) leaves each lane one row x 4 keys.
//  - online softmax in fp32 (log2 domain), P stored as fp16 to shared memory, transposed [key][row].
//  - O += P V: each thread owns 4 rows x 16 columns; P broadcast from one half of a half2 (HFMA2 operand select),
//    V columns paired in the HFMA2 lanes. The 16-key partial accumulates in fp16 (P <= 1, 16 terms), then
//    O = O * alpha + partial in fp32.
//
// Precision (Qwen3.6-27B real activations, 13 layers at 6-8k keys, max abs error against an fp32 reference, outputs
// up to |45|): SDPA 2.4e-3..2.1e-2; fa60 with 32 fp16 terms per S partial 2.6e-2..1.3e-1, with 8 terms (DSLICE 16,
// the setting below) 1.4e-2..7.4e-2 at +7% time; an fp32-FFMA Q K^T (converted operands) gets 5e-3..3e-2 but costs
// +55%. Model-level KL against SDPA (256 positions x 2 prompts, 8k and 16k tokens, -cq 4) is ~1e-3 for all three,
// against 0.8e-3..5.4e-3 between the fp16 and the -cq 4 cache with SDPA, so the cheaper form is kept.
// The 4 x 4 quad tile (Nov 5) keeps 8 fp16 terms per lane; kernel time 615 -> 530 ms for one 24k-token prefill of
// one Qwen3.6 layer (12 chunks of 2048 queries, 16 q / 2 kv heads), ~8.9 TFLOPS of the ~16.8 measured HFMA2 peak.
// KL against SDPA (Qwen3.6-35B, 16k prefix, last 512 positions, -cq 4): 3.5e-3 (old 2 x 2 tile 7.0e-3).
#if !defined(USE_ROCM)

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include "fa60.cuh"

namespace fa60
{

constexpr int HD = 256;
constexpr int BM = 64;
constexpr int BN = 16;
constexpr int NT = 256;
constexpr int QS = HD + 8;
constexpr int PS = BM + 8;

__device__ __forceinline__ half2 u2h(uint32_t u) { return *reinterpret_cast<half2*>(&u); }

template <int DSLICE, int PVK>
__global__ void __launch_bounds__(NT, 1)
fa60_kernel
(
    const half* __restrict__ q, int64_t q_row, int64_t q_head,
    const half* __restrict__ k, int64_t k_row, int64_t k_head,
    const half* __restrict__ v, int64_t v_row, int64_t v_head,
    half* __restrict__ o,
    int Tq, int Tkv, int Hq, int Hkv,
    float scale_log2, int causal, int offs
)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
    __shared__ __align__(16) half Qs[BM * QS];
    __shared__ __align__(16) half KVs[BN * QS];
    __shared__ __align__(16) half Ps[BN * PS];
    __shared__ float alpha_s[BM];
    __shared__ float l_s[BM];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int m_tile = gridDim.x - 1 - blockIdx.x;
    const int hq = blockIdx.y;
    const int hk = hq / (Hq / Hkv);
    const int m0 = m_tile * BM;

    q += hq * q_head;
    k += hk * k_head;
    v += hk * v_head;

    #pragma unroll
    for (int j = 0; j < (BM * HD / 8) / NT; ++j)
    {
        const int idx = tid + j * NT;
        const int r = idx >> 5, c = idx & 31;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (m0 + r < Tq) val = *reinterpret_cast<const uint4*>(q + (int64_t)(m0 + r) * q_row + c * 8);
        *reinterpret_cast<uint4*>(Qs + r * QS + c * 8) = val;
    }

    // S mapping: lane = rh * 16 + kb * 4 + dq. The quad (rh, kb) owns rows 8 * warp + 4 * rh + 0..3 and keys
    // 4 * kb + 0..3; lane dq of the quad sums the 16-byte head-dim chunks c with c % 4 == dq (64 dims).
    // After the quad reduction lane dq holds row 8 * warp + 4 * rh + dq, keys 4 * kb + 0..3.
    const int rh = lane >> 4, kb = (lane >> 2) & 3, dq = lane & 3;
    const int srow = 8 * warp + 4 * rh + dq;
    const int pr = tid >> 4, pc = tid & 15;

    float oacc[4][16];
    #pragma unroll
    for (int r = 0; r < 4; ++r)
        #pragma unroll
        for (int c = 0; c < 16; ++c) oacc[r][c] = 0.0f;
    float mrow = -1e30f;
    float lrow = 0.0f;

    int kv_end = causal ? min(Tkv, m0 + BM + offs) : Tkv;
    kv_end = max(kv_end, 0);
    const int n_tiles = (kv_end + BN - 1) / BN;

    uint4 st[(BN * HD / 8) / NT];
    // Per-thread source pointers of the K / V tile rows, advanced by BN rows per tile (no 64-bit products in the loop)
    const half* kg_[(BN * HD / 8) / NT];
    const half* vg_[(BN * HD / 8) / NT];
    #pragma unroll
    for (int j = 0; j < (BN * HD / 8) / NT; ++j)
    {
        const int idx = tid + j * NT;
        const int r = idx >> 5, c = idx & 31;
        kg_[j] = k + (int64_t) r * k_row + c * 8;
        vg_[j] = v + (int64_t) r * v_row + c * 8;
    }
    const int64_t k_step = (int64_t) BN * k_row, v_step = (int64_t) BN * v_row;
    auto gload = [&] (const half* const* src, int n0)
    {
        #pragma unroll
        for (int j = 0; j < (BN * HD / 8) / NT; ++j)
        {
            const int r = (tid + j * NT) >> 5;
            st[j] = n0 + r < Tkv ? *reinterpret_cast<const uint4*>(src[j]) : make_uint4(0, 0, 0, 0);
        }
    };
    auto sstore = [&] ()
    {
        #pragma unroll
        for (int j = 0; j < (BN * HD / 8) / NT; ++j)
        {
            const int idx = tid + j * NT;
            const int r = idx >> 5, c = idx & 31;
            *reinterpret_cast<uint4*>(KVs + r * QS + c * 8) = st[j];
        }
    };

    if (n_tiles > 0)
    {
        gload(kg_, 0);
        sstore();
    }
    __syncthreads();

    const half* qp = Qs + (8 * warp + 4 * rh) * QS + dq * 8;
    const half* kp = KVs + (4 * kb) * QS + dq * 8;

    for (int kt = 0; kt < n_tiles; ++kt)
    {
        const int n0 = kt * BN;

        gload(vg_, n0);
        #pragma unroll
        for (int j = 0; j < (BN * HD / 8) / NT; ++j) { vg_[j] += v_step; kg_[j] += k_step; }

        // S partial over this lane's 64 dims, 4 rows x 4 keys
        float sf[4][4];
        #pragma unroll
        for (int a = 0; a < 4; ++a)
            #pragma unroll
            for (int b = 0; b < 4; ++b) sf[a][b] = 0.0f;
        #pragma unroll
        for (int d0 = 0; d0 < HD; d0 += DSLICE)
        {
            half2 acc[4][4];
            #pragma unroll
            for (int a = 0; a < 4; ++a)
                #pragma unroll
                for (int b = 0; b < 4; ++b) acc[a][b] = __float2half2_rn(0.0f);
            // chunks d0 / 8 + dq, + 4, ...: this lane's DSLICE / 4 dims of the slice
            #pragma unroll
            for (int d = d0; d < d0 + DSLICE; d += 32)
            {
                uint4 qa[4], ka[4];
                #pragma unroll
                for (int a = 0; a < 4; ++a) qa[a] = *reinterpret_cast<const uint4*>(qp + a * QS + d);
                #pragma unroll
                for (int b = 0; b < 4; ++b) ka[b] = *reinterpret_cast<const uint4*>(kp + b * QS + d);
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                    #pragma unroll
                    for (int a = 0; a < 4; ++a)
                        #pragma unroll
                        for (int b = 0; b < 4; ++b)
                            acc[a][b] = __hfma2(u2h(reinterpret_cast<const uint32_t*>(&qa[a])[i]),
                                                u2h(reinterpret_cast<const uint32_t*>(&ka[b])[i]), acc[a][b]);
            }
            #pragma unroll
            for (int a = 0; a < 4; ++a)
                #pragma unroll
                for (int b = 0; b < 4; ++b)
                {
                    // lane sum in fp16, one HADD2.F32 (16 terms), then into the fp32 total
                    sf[a][b] += __half2float(__hadd(__low2half(acc[a][b]), __high2half(acc[a][b])));
                }
        }

        // Quad reduce-scatter: lane dq ends with row dq (4 keys)
        float s2[2][4];
        {
            const bool hi = dq & 2;
            #pragma unroll
            for (int a = 0; a < 2; ++a)
                #pragma unroll
                for (int b = 0; b < 4; ++b)
                {
                    const float send = hi ? sf[a][b] : sf[a + 2][b];
                    const float keep = hi ? sf[a + 2][b] : sf[a][b];
                    s2[a][b] = keep + __shfl_xor_sync(0xffffffff, send, 2);
                }
        }
        float x[4];
        {
            const bool hi = dq & 1;
            #pragma unroll
            for (int b = 0; b < 4; ++b)
            {
                const float send = hi ? s2[0][b] : s2[1][b];
                const float keep = hi ? s2[1][b] : s2[0][b];
                x[b] = (keep + __shfl_xor_sync(0xffffffff, send, 1)) * scale_log2;
            }
        }

        // Mask, online softmax (log2 domain); the row's 16 keys sit on the 4 kb quads (lane bits 2, 3)
        const bool need_mask = (n0 + BN > Tkv) || (causal && n0 + BN - 1 > m0 + offs);
        const int row = m0 + srow;
        if (need_mask)
        {
            #pragma unroll
            for (int b = 0; b < 4; ++b)
            {
                const int key = n0 + 4 * kb + b;
                if (key >= Tkv || (causal && key > row + offs)) x[b] = -1e30f;
            }
        }
        float mx = fmaxf(fmaxf(x[0], x[1]), fmaxf(x[2], x[3]));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 4));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 8));
        const float m_new = fmaxf(mrow, mx);
        const float alpha = exp2f(mrow - m_new);
        mrow = m_new;
        float p[4];
        #pragma unroll
        for (int b = 0; b < 4; ++b) p[b] = exp2f(x[b] - m_new);
        float sum = (p[0] + p[1]) + (p[2] + p[3]);
        sum += __shfl_xor_sync(0xffffffff, sum, 4);
        sum += __shfl_xor_sync(0xffffffff, sum, 8);
        lrow = lrow * alpha + sum;
        #pragma unroll
        for (int b = 0; b < 4; ++b) Ps[(4 * kb + b) * PS + srow] = __float2half_rn(p[b]);
        if (kb == 0) alpha_s[srow] = alpha;
        __syncthreads();

        sstore();
        const bool more = kt + 1 < n_tiles;
        if (more) gload(kg_, n0 + BN);
        __syncthreads();

        float al[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) al[r] = alpha_s[4 * pr + r];
        #pragma unroll
        for (int j0 = 0; j0 < BN; j0 += PVK)
        {
            half2 part[4][8];
            #pragma unroll
            for (int r = 0; r < 4; ++r)
                #pragma unroll
                for (int c = 0; c < 8; ++c) part[r][c] = __float2half2_rn(0.0f);
            #pragma unroll
            for (int j = j0; j < j0 + PVK; ++j)
            {
                const uint2 pp = *reinterpret_cast<const uint2*>(Ps + j * PS + 4 * pr);
                const uint4 v0 = *reinterpret_cast<const uint4*>(KVs + j * QS + pc * 8);
                const uint4 v1 = *reinterpret_cast<const uint4*>(KVs + j * QS + 128 + pc * 8);
                const half2 p01 = u2h(pp.x), p23 = u2h(pp.y);
                half2 ph[4] = { __low2half2(p01), __high2half2(p01), __low2half2(p23), __high2half2(p23) };
                const uint32_t* V0 = reinterpret_cast<const uint32_t*>(&v0);
                const uint32_t* V1 = reinterpret_cast<const uint32_t*>(&v1);
                #pragma unroll
                for (int c = 0; c < 4; ++c)
                    #pragma unroll
                    for (int r = 0; r < 4; ++r)
                    {
                        part[r][c] = __hfma2(ph[r], u2h(V0[c]), part[r][c]);
                        part[r][4 + c] = __hfma2(ph[r], u2h(V1[c]), part[r][4 + c]);
                    }
            }
            #pragma unroll
            for (int r = 0; r < 4; ++r)
            {
                const float a = j0 == 0 ? al[r] : 1.0f;
                #pragma unroll
                for (int c = 0; c < 8; ++c)
                {
                    const float2 f = __half22float2(part[r][c]);
                    if (j0 == 0)
                    {
                        oacc[r][2 * c] = fmaf(oacc[r][2 * c], a, f.x);
                        oacc[r][2 * c + 1] = fmaf(oacc[r][2 * c + 1], a, f.y);
                    }
                    else
                    {
                        oacc[r][2 * c] += f.x;
                        oacc[r][2 * c + 1] += f.y;
                    }
                }
            }
        }
        __syncthreads();

        if (more) sstore();
        __syncthreads();
    }

    if (kb == 0) l_s[srow] = lrow;
    __syncthreads();

    #pragma unroll
    for (int r = 0; r < 4; ++r)
    {
        const int row = m0 + 4 * pr + r;
        if (row >= Tq) continue;
        const float l = l_s[4 * pr + r];
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        half* o_ = o + ((int64_t) row * Hq + hq) * HD;
        half2 h[8];
        #pragma unroll
        for (int c = 0; c < 8; ++c) h[c] = __floats2half2_rn(oacc[r][2 * c] * inv, oacc[r][2 * c + 1] * inv);
        *reinterpret_cast<uint4*>(o_ + pc * 8) = *reinterpret_cast<const uint4*>(&h[0]);
        *reinterpret_cast<uint4*>(o_ + 128 + pc * 8) = *reinterpret_cast<const uint4*>(&h[4]);
    }
#endif
}

} // namespace fa60

// q [Tq, Hq, 256], k/v [Tkv, Hkv, 256] (last dim contiguous), o [Tq, Hq, 256] contiguous
void fa60_fwd(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor o, double scale, bool causal)
{
    const at::cuda::OptionalCUDAGuard guard(q.device());
    TORCH_CHECK(q.dtype() == at::kHalf && k.dtype() == at::kHalf && v.dtype() == at::kHalf && o.dtype() == at::kHalf, "fa60_fwd: fp16");
    TORCH_CHECK(q.is_cuda() && k.device() == q.device() && v.device() == q.device() && o.device() == q.device(),
                "fa60_fwd: q, k, v, o on one CUDA device");
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && v.dim() == 3 && o.dim() == 3, "fa60_fwd: 3-d tensors");
    TORCH_CHECK(q.size(2) == fa60::HD && k.size(2) == fa60::HD && v.size(2) == fa60::HD, "fa60_fwd: head_dim 256");
    TORCH_CHECK(q.stride(2) == 1 && k.stride(2) == 1 && v.stride(2) == 1 && o.is_contiguous(), "fa60_fwd: contiguous head dim");
    TORCH_CHECK(q.stride(1) % 8 == 0 && q.stride(0) % 8 == 0 && k.stride(0) % 8 == 0 && k.stride(1) % 8 == 0 &&
                v.stride(0) % 8 == 0 && v.stride(1) % 8 == 0, "fa60_fwd: 16-byte aligned strides");
    TORCH_CHECK(((uintptr_t) q.data_ptr() & 15) == 0 && ((uintptr_t) k.data_ptr() & 15) == 0 &&
                ((uintptr_t) v.data_ptr() & 15) == 0 && ((uintptr_t) o.data_ptr() & 15) == 0, "fa60_fwd: 16-byte aligned data");
    const int Tq = q.size(0), Hq = q.size(1), Tkv = k.size(0), Hkv = k.size(1);
    TORCH_CHECK(Hq % Hkv == 0 && v.size(0) == Tkv && v.size(1) == Hkv && o.size(0) == Tq && o.size(1) == Hq, "fa60_fwd: shapes");
    TORCH_CHECK(!causal || Tkv >= Tq, "fa60_fwd: causal needs Tkv >= Tq");
    if (Tq == 0) return;
    dim3 grid((Tq + fa60::BM - 1) / fa60::BM, Hq);
    // DSLICE 64 over the quad = 16 dims per lane = 8 fp16 terms per HFMA2 lane, PVK 16: see the header comment
    // (DSLICE 128 is 3% faster but doubles the KL against SDPA at 16k tokens)
    auto kern = fa60::fa60_kernel<64, 16>;
    kern<<<grid, fa60::NT, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const half*) q.data_ptr(), q.stride(0), q.stride(1),
        (const half*) k.data_ptr(), k.stride(0), k.stride(1),
        (const half*) v.data_ptr(), v.stride(0), v.stride(1),
        (half*) o.data_ptr(),
        Tq, Tkv, Hq, Hkv, (float)(scale * 1.4426950408889634), causal ? 1 : 0, Tkv - Tq);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#endif
