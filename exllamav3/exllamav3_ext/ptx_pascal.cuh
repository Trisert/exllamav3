#pragma once

#include <cuda_fp16.h>
#include <cstdint>

// Emulated Tensor Core operations for Pascal (SM 6.0, GP100) using native FP16 (__hfma2) and warp shuffles.

__device__ __forceinline__ half2 exl3_shfl_h2(half2 v, int src)
{
    uint32_t u = *reinterpret_cast<const uint32_t*>(&v);
    u = __shfl_sync(0xffffffffu, u, src);
    return *reinterpret_cast<const half2*>(&u);
}

// Gather the two rows of A (g, g + 8) and the two columns of B (2t, 2t + 1) this lane's outputs need, as
// eight k-pairs each. Row m of A lives in lanes 4 * (m % 8) .. + 3 (lane 4g + i holds k = 2i, 2i + 1 in a[0] /
// a[1] and k = 2i + 8, 2i + 9 in a[2] / a[3]); column n of B lives in lanes 4n .. 4n + 3 likewise.
__device__ __forceinline__ void exl3_mma_gather
(
    const FragA& frag_a,
    const FragB& frag_b,
    half2 (&a_rows)[2][8],
    half2 (&b_cols)[2][8]
)
{
    int lane = threadIdx.x & 31;
    int g = lane >> 2;
    int t = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        a_rows[0][i]     = exl3_shfl_h2(frag_a.elems[0], 4 * g + i);
        a_rows[0][4 + i] = exl3_shfl_h2(frag_a.elems[2], 4 * g + i);
        a_rows[1][i]     = exl3_shfl_h2(frag_a.elems[1], 4 * g + i);
        a_rows[1][4 + i] = exl3_shfl_h2(frag_a.elems[3], 4 * g + i);
        b_cols[0][i]     = exl3_shfl_h2(frag_b.elems[0], 8 * t + i);
        b_cols[0][4 + i] = exl3_shfl_h2(frag_b.elems[1], 8 * t + i);
        b_cols[1][i]     = exl3_shfl_h2(frag_b.elems[0], 8 * t + 4 + i);
        b_cols[1][4 + i] = exl3_shfl_h2(frag_b.elems[1], 8 * t + 4 + i);
    }
}

// 16-element dot product using native __hfma2 (8 pairs of half2)
__device__ __forceinline__ half2 exl3_dot16_hfma2(const half2 (&a)[8], const half2 (&b)[8])
{
    half2 s = __float2half2_rn(0.0f);
    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        s = __hfma2(a[i], b[i], s);
    }
    return s;
}

__device__ __forceinline__ float exl3_dot16_f32(const half2 (&a)[8], const half2 (&b)[8])
{
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        float2 af = __half22float2(a[i]);
        float2 bf = __half22float2(b[i]);
        s = fmaf(af.x, bf.x, s);
        s = fmaf(af.y, bf.y, s);
    }
    return s;
}

// FP16 @ FP16 + FP32 -> FP32 (interleaved 4-way accumulation to hide FMA latency)
__device__ __forceinline__ void exl3_mma_m16n8k16_f32(const FragA& frag_a, const FragB& frag_b, FragC& frag_c)
{
    half2 a_rows[2][8], b_cols[2][8];
    exl3_mma_gather(frag_a, frag_b, a_rows, b_cols);

    float d00 = 0.0f, d01 = 0.0f, d10 = 0.0f, d11 = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        float2 a0 = __half22float2(a_rows[0][i]);
        float2 a1 = __half22float2(a_rows[1][i]);
        float2 b0 = __half22float2(b_cols[0][i]);
        float2 b1 = __half22float2(b_cols[1][i]);

        d00 = fmaf(a0.x, b0.x, fmaf(a0.y, b0.y, d00));
        d01 = fmaf(a0.x, b1.x, fmaf(a0.y, b1.y, d01));
        d10 = fmaf(a1.x, b0.x, fmaf(a1.y, b0.y, d10));
        d11 = fmaf(a1.x, b1.x, fmaf(a1.y, b1.y, d11));
    }
    frag_c.elems[0] += d00;
    frag_c.elems[1] += d01;
    frag_c.elems[2] += d10;
    frag_c.elems[3] += d11;
}

// FP16 @ FP16 + FP16 -> FP16 (Pascal P100 __hfma2 fast path with 4-way interleaved accumulators)
__device__ __forceinline__ void exl3_mma_m16n8k16_f16(const FragA& frag_a, const FragB& frag_b, FragC_h& frag_c)
{
    half2 a_rows[2][8], b_cols[2][8];
    exl3_mma_gather(frag_a, frag_b, a_rows, b_cols);

    half2 d00 = __float2half2_rn(0.0f);
    half2 d01 = __float2half2_rn(0.0f);
    half2 d10 = __float2half2_rn(0.0f);
    half2 d11 = __float2half2_rn(0.0f);

    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        d00 = __hfma2(a_rows[0][i], b_cols[0][i], d00);
        d01 = __hfma2(a_rows[0][i], b_cols[1][i], d01);
        d10 = __hfma2(a_rows[1][i], b_cols[0][i], d10);
        d11 = __hfma2(a_rows[1][i], b_cols[1][i], d11);
    }

    half s00 = __hadd(__low2half(d00), __high2half(d00));
    half s01 = __hadd(__low2half(d01), __high2half(d01));
    half s10 = __hadd(__low2half(d10), __high2half(d10));
    half s11 = __hadd(__low2half(d11), __high2half(d11));

    frag_c.elems[0] = __hadd2(frag_c.elems[0], __halves2half2(s00, s01));
    frag_c.elems[1] = __hadd2(frag_c.elems[1], __halves2half2(s10, s11));
}

// ldmatrix.sync.aligned.m8n8.x4.b16 emulation via 32-bit shared memory __shfl_sync
__device__ __forceinline__ void exl3_ldsm4(FragA& frag_a, const void* smem_ptr)
{
    int lane = threadIdx.x & 31;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    int row = lane >> 2;
    int col_bytes = (lane & 3) * 4;
    uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        uint32_t row_smem = __shfl_sync(0xffffffffu, smem, 8 * i + row);
        const char* ptr = static_cast<const char*>(__cvta_shared_to_generic(row_smem)) + col_bytes;
        a[i] = *reinterpret_cast<const uint32_t*>(ptr);
    }
}
