// Flash-decoding split kernel (phase 1) for the paged quantized cache, CUDA-native replacement for
// triton_paged._paged_attn_decode_split_kernel in the graphed decode (BC_Attention) on parts where Triton lowers
// tl.dot to scalar FMAs (sm_60): the Triton tile kernel reaches ~70 GB/s there at long contexts. Same launch
// arguments and partial layout, so the Triton combine kernel and the graph's parameter sites are unchanged.
//
// Scope: q_len 1, 8-bit K and V, head_dim 128, no window / softcap / sinks, the combine pass always runs (FINAL is
// the Triton kernel's). Values stay in the cache's H32-rotated domain: q is rotated here (as the Triton kernel does),
// the output is rotated back by the combine kernel.
//
// Cache layout (8 bits): per token row, per kv head, head_dim / 32 groups of 32 bytes (value i of a group in byte i)
// and one fp16 scale per group; value = (byte - 127.5) * scale / 128.
//
// One CTA of 4 warps per (batch * kv head, split). Scores: one key per lane (the dequantized key dotted with the
// group's GROUP q rows from shared memory, fp32), online softmax per warp, P x V with one 4-dim slice of the head per
// lane (coalesced V rows), fp32 accumulators. The four warps' (m, l, acc) merge in shared memory.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cmath>
#include <memory>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <c10/util/Exception.h>
#include "triton_kernel.h"
#include "util.h"
#include "util.cuh"

namespace attn_native {

constexpr int HD = 128;
constexpr int NG = HD / 32;     // scale groups per head
constexpr int TILE = 32;        // keys per warp step

__device__ __forceinline__ float byte_f(uint32_t w, int i)
{
    // Exact uint8 -> fp32: the byte as the low mantissa bits of 2^23
    return __uint_as_float(0x4B000000u | ((w >> (8 * i)) & 0xffu)) - 8388608.0f;
}

// Key word w (dims d .. d + 3) as two half2 of exact centered bytes: lo = {b0, b2} - 128, hi = {b1, b3} - 128. The
// masked bytes under 0x6400 are the fp16 patterns of 1024 + byte (LOP3, full rate), one HSUB2 takes 1152 off each
__device__ __forceinline__ void key_h2(uint32_t w, half2& lo, half2& hi)
{
    const half2 c1152 = __half2half2(__ushort_as_half(0x6480));
    half2_uint32 ul((w & 0x00ff00ffu) | 0x64006400u);
    half2_uint32 uh(((w >> 8) & 0x00ff00ffu) | 0x64006400u);
    lo = __hsub2(ul.as_half2, c1152);
    hi = __hsub2(uh.as_half2, c1152);
}

// WARPS: warps per CTA, each walking every WARPS-th key tile of the split (EXL3_ATTN_NATIVE_WARPS; 8 by default:
// LFM3 on the P100 at 32k 125.2 -> 128.1 t/s over 4, unchanged at 512)
// HF: the score loop in HFMA2 (EXL3_ATTN_NATIVE_HF): q as fp16 in smem, interleaved {q0, q2}, {q1, q3} per 4 dims to
// pair with key_h2, one LDS.128 per 8 dims and head; runs of 16 products per half accumulator, folded to fp32 per
// scale group. P x V stays fp32
template <int GROUP, int WARPS, bool HF>
__global__ __launch_bounds__(WARPS * 32)
void decode_split_kernel
(
    const half* __restrict__ q,
    const uint32_t* __restrict__ k_cache,
    const uint32_t* __restrict__ v_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ cache_seqlens,
    half* __restrict__ out,                 // unused (FINAL is the Triton kernel's)
    float* __restrict__ partial_o,
    float* __restrict__ partial_ml,
    const half* __restrict__ k_scales,
    const half* __restrict__ v_scales,
    const half* __restrict__ h32,
    int split_len,
    int num_pages_per_seq,
    int num_splits,
    const float* __restrict__ sinks,        // unused
    // Trailing constants (TritonKernel extra args)
    float scale,
    int n_kv_heads,
    int page_size,
    int live_split,
    int block_rows,
    // Triton's hidden scratch params (null)
    void* scratch0,
    void* scratch1
)
{
    const int pid = blockIdx.x;
    const int split = blockIdx.y;
    if (split >= num_splits) return;

    const int batch = pid / n_kv_heads;
    const int kv_head = pid - batch * n_kv_heads;
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int n_q_heads = n_kv_heads * GROUP;
    const int gpt = n_kv_heads * NG;            // scale groups per token row
    const int wpt = n_kv_heads * HD / 4;        // 32-bit words per token row

    __shared__ float q_s[HF ? 1 : GROUP][HF ? 1 : HD];      // rotated, pre-scaled q rows
    __shared__ __align__(16) half qh_s[HF ? GROUP : 1][HF ? HD : 8];   // HF: rotated q rows (fp16, unscaled, interleaved)
    __shared__ float qsum_s[GROUP][NG];         // per scale group sum of the q row (the 127.5 offset; HF: unscaled)
    __shared__ float m_s[WARPS][GROUP], l_s[WARPS][GROUP];
    __shared__ float acc_s[WARPS][GROUP][HD];

    // q rows of this kv head's group, rotated by H32 per 32-value group (h32 is the normalized Sylvester matrix, so
    // the product is a 5-stage Walsh-Hadamard butterfly over the lanes, one 32-value group per warp step), rounded to
    // fp16 as the Triton kernel's rotation is, then pre-scaled. The group sums feed the 127.5 offset
    for (int t = warp; t < GROUP * NG; t += WARPS)
    {
        const int h = t / NG, g = t % NG;
        float x = __half2float(q[((size_t) batch * n_q_heads + kv_head * GROUP + h) * HD + g * 32 + lane]);
        #pragma unroll
        for (int s = 1; s < 32; s <<= 1)
        {
            const float y = __shfl_xor_sync(0xffffffffu, x, s);
            x = (lane & s) ? y - x : x + y;
        }
        const half xh = __float2half_rn(x * 0.17677669529663687f);   // 1 / sqrt(32)
        if constexpr (HF)
        {
            // dim 4j + r -> half2 entry 2j + (r & 1), half r >> 1
            const int dd = g * 32 + lane, j = dd >> 2, r = dd & 3;
            qh_s[h][(2 * j + (r & 1)) * 2 + (r >> 1)] = xh;
            x = __half2float(xh);
        }
        else
        {
            x = __half2float(xh) * scale;
            q_s[h][g * 32 + lane] = x;
        }
        float sum = x;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (lane == 0) qsum_s[h][g] = sum;
    }
    __syncthreads();

    // Key range of this split (same partition as the Triton kernel: live length or the host's split_len)
    const int total = cache_seqlens[batch] + 1;
    int n_start, n_end;
    if (live_split)
    {
        const int l_span = CEIL_DIVIDE(CEIL_DIVIDE(total, num_splits), 16) * 16;
        n_start = split * l_span;
        n_end = min(n_start + l_span, total);
    }
    else
    {
        n_start = split * split_len;
        n_end = min(n_start + split_len, total);
    }

    float m[GROUP], l[GROUP], acc[GROUP][4];
    #pragma unroll
    for (int h = 0; h < GROUP; ++h)
    {
        m[h] = -INFINITY;
        l[h] = 0.0f;
        #pragma unroll
        for (int e = 0; e < 4; ++e) acc[h][e] = 0.0f;
    }

    const int* bt = block_table + (size_t) batch * num_pages_per_seq;
    const int gl = lane / 8;                    // scale group of this lane's 4 dims (P x V)

    // This lane's key of the tile at n0_ (row 0 past the split end; its score is masked)
    auto tok_of = [&] (int n0_) -> int
    {
        const int n = n0_ + lane;
        if (n >= n_end) return 0;
        const int page = n / page_size;
        return bt[page] * page_size + (n - page * page_size);
    };
    auto load_k = [&] (int t, uint4 (&kw)[8], uint2& ks)
    {
        const uint4* kr = reinterpret_cast<const uint4*>(k_cache + (size_t) t * wpt + kv_head * (HD / 4));
        #pragma unroll
        for (int c = 0; c < 8; ++c) kw[c] = kr[c];
        ks = *reinterpret_cast<const uint2*>(k_scales + (size_t) t * gpt + kv_head * NG);
    };

    for (int n0 = n_start + warp * TILE; n0 < n_end; n0 += WARPS * TILE)
    {
        const int n = n0 + lane;
        const bool valid = n < n_end;
        const int nvalid = min(TILE, n_end - n0);
        const int tok = tok_of(n0);
        uint4 kw[8];
        uint2 ksw;
        load_k(tok, kw, ksw);

        // Scores, one key per lane
        float sc[GROUP];
        {
            const half2 ks01 = *reinterpret_cast<const half2*>(&ksw.x);
            const half2 ks23 = *reinterpret_cast<const half2*>(&ksw.y);
            const float ksc[NG] = { __low2float(ks01), __high2float(ks01), __low2float(ks23), __high2float(ks23) };
            #pragma unroll
            for (int h = 0; h < GROUP; ++h) sc[h] = 0.0f;
            if constexpr (HF)
            {
                // q . (b - 127.5) = sum q (b - 128) + 0.5 sum q, per scale group
                #pragma unroll
                for (int g = 0; g < NG; ++g)
                {
                    half2 dh[GROUP];
                    #pragma unroll
                    for (int h = 0; h < GROUP; ++h) dh[h] = __half2half2(__ushort_as_half(0));
                    #pragma unroll
                    for (int c = 0; c < 2; ++c)
                    {
                        const uint4 w = kw[g * 2 + c];
                        const uint32_t ws[4] = { w.x, w.y, w.z, w.w };
                        #pragma unroll
                        for (int wp = 0; wp < 2; ++wp)
                        {
                            half2 lo0, hi0, lo1, hi1;
                            key_h2(ws[wp * 2], lo0, hi0);
                            key_h2(ws[wp * 2 + 1], lo1, hi1);
                            const int e0 = (g * 32 + c * 16 + wp * 8) / 2;      // half2 entry of the 8 dims
                            #pragma unroll
                            for (int h = 0; h < GROUP; ++h)
                            {
                                const uint4 qu = *reinterpret_cast<const uint4*>(&qh_s[h][e0 * 2]);
                                dh[h] = __hfma2(*reinterpret_cast<const half2*>(&qu.x), lo0, dh[h]);
                                dh[h] = __hfma2(*reinterpret_cast<const half2*>(&qu.y), hi0, dh[h]);
                                dh[h] = __hfma2(*reinterpret_cast<const half2*>(&qu.z), lo1, dh[h]);
                                dh[h] = __hfma2(*reinterpret_cast<const half2*>(&qu.w), hi1, dh[h]);
                            }
                        }
                    }
                    #pragma unroll
                    for (int h = 0; h < GROUP; ++h)
                        sc[h] = fmaf(ksc[g] * (scale * (1.0f / 128.0f)),
                                     __low2float(dh[h]) + __high2float(dh[h]) + 0.5f * qsum_s[h][g], sc[h]);
                }
            }
            else
            #pragma unroll
            for (int g = 0; g < NG; ++g)
            {
                float dg[GROUP];
                #pragma unroll
                for (int h = 0; h < GROUP; ++h) dg[h] = 0.0f;
                #pragma unroll
                for (int c = 0; c < 2; ++c)
                {
                    const uint4 w = kw[g * 2 + c];
                    const uint32_t ws[4] = { w.x, w.y, w.z, w.w };
                    #pragma unroll
                    for (int wi = 0; wi < 4; ++wi)
                    {
                        const int d0 = g * 32 + c * 16 + wi * 4;
                        const float b0 = byte_f(ws[wi], 0), b1 = byte_f(ws[wi], 1);
                        const float b2 = byte_f(ws[wi], 2), b3 = byte_f(ws[wi], 3);
                        #pragma unroll
                        for (int h = 0; h < GROUP; ++h)
                        {
                            const float4 qv = *reinterpret_cast<const float4*>(&q_s[h][d0]);
                            dg[h] = fmaf(qv.x, b0, dg[h]);
                            dg[h] = fmaf(qv.y, b1, dg[h]);
                            dg[h] = fmaf(qv.z, b2, dg[h]);
                            dg[h] = fmaf(qv.w, b3, dg[h]);
                        }
                    }
                }
                #pragma unroll
                for (int h = 0; h < GROUP; ++h)
                    sc[h] = fmaf(ksc[g] * (1.0f / 128.0f), dg[h] - 127.5f * qsum_s[h][g], sc[h]);
            }
        }

        // Online softmax (per warp; m is warp-uniform, l stays a per-lane partial)
        float p[GROUP];
        #pragma unroll
        for (int h = 0; h < GROUP; ++h)
        {
            const float s = valid ? sc[h] : -INFINITY;
            float tm = s;
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, o));
            const float m_new = fmaxf(m[h], tm);
            const float alpha = m[h] == -INFINITY ? 0.0f : __expf(m[h] - m_new);
            p[h] = valid ? __expf(s - m_new) : 0.0f;
            l[h] = l[h] * alpha + p[h];
            #pragma unroll
            for (int e = 0; e < 4; ++e) acc[h][e] *= alpha;
            m[h] = m_new;
        }

        // P x V: lane owns dims 4 * lane .. + 3 of every valid key in the tile
        const uint32_t* vbase = v_cache + kv_head * (HD / 4) + lane;
        const half* vsbase = v_scales + kv_head * NG + gl;
        #pragma unroll 8
        for (int j = 0; j < nvalid; ++j)
        {
            const int tj = __shfl_sync(0xffffffffu, tok, j);
            const uint32_t vw = vbase[(size_t) tj * wpt];
            const float vs = __half2float(vsbase[(size_t) tj * gpt]) * (1.0f / 128.0f);
            const float v0 = byte_f(vw, 0) - 127.5f, v1 = byte_f(vw, 1) - 127.5f;
            const float v2 = byte_f(vw, 2) - 127.5f, v3 = byte_f(vw, 3) - 127.5f;
            #pragma unroll
            for (int h = 0; h < GROUP; ++h)
            {
                const float pw = __shfl_sync(0xffffffffu, p[h], j) * vs;
                acc[h][0] = fmaf(pw, v0, acc[h][0]);
                acc[h][1] = fmaf(pw, v1, acc[h][1]);
                acc[h][2] = fmaf(pw, v2, acc[h][2]);
                acc[h][3] = fmaf(pw, v3, acc[h][3]);
            }
        }
    }

    // Merge the warps: l to a warp total, then (m, l, acc) across warps through shared memory
    #pragma unroll
    for (int h = 0; h < GROUP; ++h)
    {
        float lt = l[h];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) lt += __shfl_xor_sync(0xffffffffu, lt, o);
        if (lane == 0) { m_s[warp][h] = m[h]; l_s[warp][h] = lt; }
        #pragma unroll
        for (int e = 0; e < 4; ++e) acc_s[warp][h][lane * 4 + e] = acc[h][e];
    }
    __syncthreads();

    const size_t pbase = (size_t) pid * num_splits + split;
    float* po = partial_o + pbase * block_rows * HD;
    float* pml = partial_ml + pbase * block_rows * 2;
    for (int idx = threadIdx.x; idx < GROUP * HD; idx += WARPS * 32)
    {
        const int h = idx / HD, d = idx % HD;
        float mm = -INFINITY;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) mm = fmaxf(mm, m_s[w][h]);
        float a = 0.0f;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w)
            if (m_s[w][h] != -INFINITY) a += acc_s[w][h][d] * __expf(m_s[w][h] - mm);
        po[h * HD + d] = a;
        if (d == 0)
        {
            float ll = 0.0f;
            #pragma unroll
            for (int w = 0; w < WARPS; ++w)
                if (m_s[w][h] != -INFINITY) ll += l_s[w][h] * __expf(m_s[w][h] - mm);
            pml[h * 2] = mm;
            pml[h * 2 + 1] = ll;
        }
    }
    // Rows past the group (block_rows pads to 16): empty partials, so the combine kernel's weights are exactly zero
    for (int r = GROUP + threadIdx.x; r < block_rows; r += WARPS * 32)
    {
        pml[r * 2] = -INFINITY;
        pml[r * 2 + 1] = 0.0f;
    }
}

template <int GROUP>
static void* kernel_ptr(int warps, bool hf)
{
    if (hf) return warps == 4 ? (void*) decode_split_kernel<GROUP, 4, true> : (void*) decode_split_kernel<GROUP, 8, true>;
    return warps == 4 ? (void*) decode_split_kernel<GROUP, 4, false> : (void*) decode_split_kernel<GROUP, 8, false>;
}

}  // namespace attn_native

// Native split kernel wrapped as a TritonKernel (same launch sites); nullptr-free: errors on unsupported shapes
std::shared_ptr<TritonKernel> make_decode_split_native
(
    int group,
    int head_dim,
    int bits,
    float scale,
    int n_kv_heads,
    int page_size,
    bool live_split,
    int block_rows
)
{
    TORCH_CHECK(head_dim == attn_native::HD && bits == 8, "make_decode_split_native: head_dim 128, 8-bit cache only");
    TORCH_CHECK(group >= 1 && group <= block_rows && block_rows <= 32, "make_decode_split_native: bad row layout");
    const char* w_env = std::getenv("EXL3_ATTN_NATIVE_WARPS");
    const int warps = w_env && std::atoi(w_env) == 4 ? 4 : 8;
    const char* hf_env = std::getenv("EXL3_ATTN_NATIVE_HF");
    const bool hf = !hf_env || std::atoi(hf_env) != 0;     // default on (P100 +1.1 % at 512, +1.7..2.4 % at 32k)
    void* fptr = nullptr;
    switch (group)
    {
        case 1: fptr = attn_native::kernel_ptr<1>(warps, hf); break;
        case 2: fptr = attn_native::kernel_ptr<2>(warps, hf); break;
        case 4: fptr = attn_native::kernel_ptr<4>(warps, hf); break;
        case 8: fptr = attn_native::kernel_ptr<8>(warps, hf); break;
        default: TORCH_CHECK(false, "make_decode_split_native: GQA group ", group, " not instantiated");
    }
    cudaFunction_t fn;
    cuda_check(cudaGetFuncBySymbol(&fn, fptr));
    uint32_t sbits;
    std::memcpy(&sbits, &scale, 4);
    std::vector<void*> extra =
    {
        (void*) (uintptr_t) sbits,
        (void*) (intptr_t) n_kv_heads,
        (void*) (intptr_t) page_size,
        (void*) (intptr_t) (live_split ? 1 : 0),
        (void*) (intptr_t) block_rows,
    };
    return std::make_shared<TritonKernel>((CUfunction) fn, fptr, "decode_split_native", warps, 0, extra);
}
