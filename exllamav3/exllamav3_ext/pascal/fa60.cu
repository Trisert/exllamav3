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
//  - S = Q K^T: each thread owns 2 rows x 2 keys; HFMA2 lanes hold two consecutive head dims, so Q and K are read
//    in their natural row layout. fp16 partials over 64 dims (32 terms per lane) are summed into fp32.
//  - online softmax in fp32 (log2 domain), P stored as fp16 to shared memory, transposed [key][row].
//  - O += P V: each thread owns 4 rows x 16 columns; P broadcast from one half of a half2 (HFMA2 operand select),
//    V columns paired in the HFMA2 lanes. The 16-key partial accumulates in fp16 (P <= 1, 16 terms), then
//    O = O * alpha + partial in fp32.
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
constexpr int QS = HD + 8;           // row stride (halves) of Q and K/V tiles: 528 B, rows 4 banks apart
constexpr int PS = BM + 8;           // row stride (halves) of the transposed P tile: 144 B
constexpr int DSLICE = 64;           // head dims per fp16 partial of S (32 HFMA2 terms per lane)

__device__ __forceinline__ half2 u2h(uint32_t u) { return *reinterpret_cast<half2*>(&u); }

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

    const int m_tile = gridDim.x - 1 - blockIdx.x;      // heaviest (last) query tiles first
    const int hq = blockIdx.y;
    const int hk = hq / (Hq / Hkv);
    const int m0 = m_tile * BM;

    q += hq * q_head;
    k += hk * k_head;
    v += hk * v_head;

    // Stage Q: 64 rows x 32 16-byte chunks, 8 per thread
    #pragma unroll
    for (int j = 0; j < (BM * HD / 8) / NT; ++j)
    {
        const int idx = tid + j * NT;
        const int r = idx >> 5, c = idx & 31;
        uint4 val = make_uint4(0, 0, 0, 0);
        if (m0 + r < Tq) val = *reinterpret_cast<const uint4*>(q + (int64_t)(m0 + r) * q_row + c * 8);
        *reinterpret_cast<uint4*>(Qs + r * QS + c * 8) = val;
    }

    // S mapping: rows sr0 = 8 * warp + rg and sr0 + 4, keys kg and kg + 8 (conflict-free LDS.128: the four rows
    // and eight keys of one load sit 4 banks apart)
    const int rg = lane >> 3, kg = lane & 7;
    const int sr0 = 8 * warp + rg, sr1 = sr0 + 4;
    // PV mapping: rows 4 * pr .. 4 * pr + 3, columns pc * 8 + 0..7 and 128 + pc * 8 + 0..7
    const int pr = tid >> 4, pc = tid & 15;

    float oacc[4][16];
    #pragma unroll
    for (int r = 0; r < 4; ++r)
        #pragma unroll
        for (int c = 0; c < 16; ++c) oacc[r][c] = 0.0f;
    float mrow[2] = { -1e30f, -1e30f };
    float lrow[2] = { 0.0f, 0.0f };

    int kv_end = causal ? min(Tkv, m0 + BM + offs) : Tkv;
    kv_end = max(kv_end, 0);
    const int n_tiles = (kv_end + BN - 1) / BN;

    uint4 st[(BN * HD / 8) / NT];
    auto gload = [&] (const half* src, int64_t rs, int n0)
    {
        #pragma unroll
        for (int j = 0; j < (BN * HD / 8) / NT; ++j)
        {
            const int idx = tid + j * NT;
            const int r = idx >> 5, c = idx & 31;
            st[j] = n0 + r < Tkv ? *reinterpret_cast<const uint4*>(src + (int64_t)(n0 + r) * rs + c * 8) : make_uint4(0, 0, 0, 0);
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
        gload(k, k_row, 0);
        sstore();
    }
    __syncthreads();

    const half* qa = Qs + sr0 * QS;
    const half* qb = Qs + sr1 * QS;
    const half* ka = KVs + kg * QS;
    const half* kb = KVs + (kg + 8) * QS;

    for (int kt = 0; kt < n_tiles; ++kt)
    {
        const int n0 = kt * BN;

        // V(kt) in flight during Q K^T
        gload(v, v_row, n0);

        // S = Q K^T, 2 x 2 per thread
        float sf[2][2] = { { 0.0f, 0.0f }, { 0.0f, 0.0f } };
        #pragma unroll
        for (int d0 = 0; d0 < HD; d0 += DSLICE)
        {
            half2 acc[2][2];
            #pragma unroll
            for (int a = 0; a < 2; ++a)
                #pragma unroll
                for (int b = 0; b < 2; ++b) acc[a][b] = __float2half2_rn(0.0f);
            #pragma unroll
            for (int d = d0; d < d0 + DSLICE; d += 8)
            {
                const uint4 a0 = *reinterpret_cast<const uint4*>(qa + d);
                const uint4 a1 = *reinterpret_cast<const uint4*>(qb + d);
                const uint4 b0 = *reinterpret_cast<const uint4*>(ka + d);
                const uint4 b1 = *reinterpret_cast<const uint4*>(kb + d);
                const uint32_t* A0 = reinterpret_cast<const uint32_t*>(&a0);
                const uint32_t* A1 = reinterpret_cast<const uint32_t*>(&a1);
                const uint32_t* B0 = reinterpret_cast<const uint32_t*>(&b0);
                const uint32_t* B1 = reinterpret_cast<const uint32_t*>(&b1);
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    acc[0][0] = __hfma2(u2h(A0[i]), u2h(B0[i]), acc[0][0]);
                    acc[0][1] = __hfma2(u2h(A0[i]), u2h(B1[i]), acc[0][1]);
                    acc[1][0] = __hfma2(u2h(A1[i]), u2h(B0[i]), acc[1][0]);
                    acc[1][1] = __hfma2(u2h(A1[i]), u2h(B1[i]), acc[1][1]);
                }
            }
            #pragma unroll
            for (int a = 0; a < 2; ++a)
                #pragma unroll
                for (int b = 0; b < 2; ++b)
                {
                    const float2 f = __half22float2(acc[a][b]);
                    sf[a][b] += f.x + f.y;
                }
        }

        // Mask, online softmax (log2 domain); P to shared memory as fp16, transposed
        const bool need_mask = (n0 + BN > Tkv) || (causal && n0 + BN - 1 > m0 + offs);
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh)
        {
            const int rl = hh ? sr1 : sr0;
            const int row = m0 + rl;
            float x[2];
            #pragma unroll
            for (int e = 0; e < 2; ++e)
            {
                x[e] = sf[hh][e] * scale_log2;
                if (need_mask)
                {
                    const int key = n0 + kg + 8 * e;
                    if (key >= Tkv || (causal && key > row + offs)) x[e] = -1e30f;
                }
            }
            float mx = fmaxf(x[0], x[1]);
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 4));
            const float m_new = fmaxf(mrow[hh], mx);
            const float alpha = exp2f(mrow[hh] - m_new);
            mrow[hh] = m_new;
            const float p0 = exp2f(x[0] - m_new);
            const float p1 = exp2f(x[1] - m_new);
            float sum = p0 + p1;
            sum += __shfl_xor_sync(0xffffffff, sum, 1);
            sum += __shfl_xor_sync(0xffffffff, sum, 2);
            sum += __shfl_xor_sync(0xffffffff, sum, 4);
            lrow[hh] = lrow[hh] * alpha + sum;
            Ps[kg * PS + rl] = __float2half_rn(p0);
            Ps[(kg + 8) * PS + rl] = __float2half_rn(p1);
            if (kg == 0) alpha_s[rl] = alpha;
        }
        __syncthreads();                                   // K(kt) no longer read, P visible

        sstore();                                          // V(kt)
        const bool more = kt + 1 < n_tiles;
        if (more) gload(k, k_row, n0 + BN);                // K(kt+1) in flight during P V
        __syncthreads();

        // O += P V, 4 rows x 16 columns per thread, 16-key fp16 partial
        half2 part[4][8];
        #pragma unroll
        for (int r = 0; r < 4; ++r)
            #pragma unroll
            for (int c = 0; c < 8; ++c) part[r][c] = __float2half2_rn(0.0f);
        #pragma unroll
        for (int j = 0; j < BN; ++j)
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
            const float a = alpha_s[4 * pr + r];
            #pragma unroll
            for (int c = 0; c < 8; ++c)
            {
                const float2 f = __half22float2(part[r][c]);
                oacc[r][2 * c] = fmaf(oacc[r][2 * c], a, f.x);
                oacc[r][2 * c + 1] = fmaf(oacc[r][2 * c + 1], a, f.y);
            }
        }
        __syncthreads();                                   // V(kt), P, alpha no longer read

        if (more) sstore();                                // K(kt+1)
        __syncthreads();
    }

    if (kg == 0)
    {
        l_s[sr0] = lrow[0];
        l_s[sr1] = lrow[1];
    }
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
    fa60::fa60_kernel<<<grid, fa60::NT, 0, at::cuda::getCurrentCUDAStream()>>>(
        (const half*) q.data_ptr(), q.stride(0), q.stride(1),
        (const half*) k.data_ptr(), k.stride(0), k.stride(1),
        (const half*) v.data_ptr(), v.stride(0), v.stride(1),
        (half*) o.data_ptr(),
        Tq, Tkv, Hq, Hkv, (float)(scale * 1.4426950408889634), causal ? 1 : 0, Tkv - Tq);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#endif
