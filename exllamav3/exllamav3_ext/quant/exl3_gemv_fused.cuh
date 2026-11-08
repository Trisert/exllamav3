#pragma once

// Fused multi-matrix GEMV for the sm_60 direct-dot path.
//
// Several EXL3 linears of one layer share the same input (a GatedDeltaNet layer's in_proj_qkv and
// in_proj_z, an attention layer's q/k/v projections). Run as separate exl3_gemv calls each pays the
// full fixed cost: a cooperative launch, the input Hadamard prologue, two grid syncs and, on this
// device, a half-empty last wave (the column groups never divide the 56 SMs evenly: on Ornith's
// shapes one call is 100-140 GB/s against ~250 GB/s for a launch of twice the width). This kernel
// serves every matrix of a group from ONE cooperative launch: one prologue (one Hadamard per
// matrix, into its own scratch row of A_had), one grid sync, then a single column-group loop that
// walks the matrices' groups back to back, then one output Hadamard per matrix.
//
// Only the DIRECT dot-product path is implemented, i.e. what Pascal uses at m = 1 (MMODE 0) and
// m = 2 (MMODE 2); the arithmetic per output element is exactly exl3_gemv_kernel's (same k-split,
// same half2 partials, same quad reduction order), so the outputs are bit-identical to the separate
// calls. Integer bitrates 2..4 only, shuffle extraction (SMEM_STAGE = false).
//
// The per-matrix state (weight/out/scale pointers, widths, cumulative group counts) is passed by
// value in GemvFusedArgs; no graph capture, so the launch arguments are free-form.

#include <cooperative_groups.h>
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "exl3_gemv_kernel.cuh"
#include "hadamard_inner.cuh"

namespace exl3_gemv_fused_ns
{

constexpr int MAX_MATRICES = 4;

struct GemvFusedArgs
{
    const half* A;                  // (size_m, size_k) raw input, shared by every matrix
    const half* suh[MAX_MATRICES];  // per matrix input sign/scale
    half* A_had;                    // num_matrices * size_m * size_k scratch, one row group each
    const uint16_t* B[MAX_MATRICES];
    void* C[MAX_MATRICES];
    const half* svh[MAX_MATRICES];
    int n[MAX_MATRICES];            // output width of each matrix
    int g[MAX_MATRICES + 1];        // cumulative column-group counts (n / COLS), g[num] = total
    int size_m;
    int size_k;
    int num_matrices;
    int* locks;
};

template <int bits, bool c_fp32, int cb, int MMODE, int CFG>
__global__ __launch_bounds__(CFG == 0 ? 512 : 256, 1)
void exl3_gemv_fused_kernel(GemvFusedArgs args)
{
    constexpr bool DIRECT = true;
    constexpr int NR = MMODE == 2 ? 2 : 1;      // rows handled by the direct path
    constexpr int ROWS = NR;
    constexpr int WK = CFG == 0 ? 16 : 8;
    constexpr int WNT = CFG == 0 ? 2 : 4;
    constexpr int PF = CFG == 0 ? 4 : 2;
    constexpr int FOLD = CFG == 0 ? 4 : 2;
    constexpr int THREADS = WK * 32;
    constexpr int COLS = WNT * 16;
    constexpr int TWORDS = 8 * bits;
    constexpr bool TWO_PER_LOAD = bits == 2;
    constexpr int LOADS = TWO_PER_LOAD ? WNT / 2 : WNT;
    constexpr int LSTRIDE = TWO_PER_LOAD ? 2 * TWORDS : (TWORDS < 32 ? TWORDS : 32);

    const int size_m = args.size_m;
    const int size_k = args.size_k;
    const int num_matrices = args.num_matrices;

    auto grid = cooperative_groups::this_grid();

    // Input Hadamard per matrix, into its own scratch row group. Same inner function and scale
    // the regular GEMV prologue uses.
    {
        const int warps_grid = gridDim.x * blockDim.x / 32;
        const int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;
        const int total_warps = size_m * size_k / 128;
        for (int m = 0; m < num_matrices; ++m)
            for (int w = this_warp; w < total_warps; w += warps_grid)
                had_hf_r_128_inner<true, false>
                (
                    args.A + w * 128,
                    args.A_had + (size_t) m * size_m * size_k + w * 128,
                    args.suh[m] + (w * 128) % size_k,
                    0.088388347648f
                );
        grid.sync();
    }

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int kslices = size_k / 16;
    const int chunk = CEIL_DIVIDE(kslices, WK);
    const int ks0 = warp * chunk;
    const int myn = max(0, min(chunk, kslices - ks0));
    const half2 hzero = __half2half2(__ushort_as_half(0));
    const int r0 = lane >> 2;
    const size_t a_row0 = (size_t) r0 * (size_k / 2);

    // Per-lane extraction constants (see exl3_gemv_kernel.cuh, same values)
    [[maybe_unused]] int x_src_a = 0, x_src_b = 0, x_s2 = 0;
    if constexpr (bits == 2)
    {
        int i1 = lane >> 1;
        x_src_b = i1;
        x_src_a = (i1 + 15) & 15;
    }
    else if constexpr (bits == 3)
    {
        int t_offset = lane << 3;
        int b1 = (t_offset + 257) * 3;
        int b2 = b1 + 21;
        int i0 = (b1 - 16) / 32;
        int i2 = (b2 - 1) / 32;
        x_s2 = (i2 + 1) * 32 - b2;
        x_src_a = i0 % 24;
        x_src_b = i2 % 24;
    }

    __shared__ float sh_red[WK][ROWS][COLS];

    const int total_groups = args.g[num_matrices];
    for (int work = blockIdx.x; work < total_groups; work += gridDim.x)
    {
        // Which matrix this work item belongs to, and its column group within that matrix
        int mi = 0;
        while (mi + 1 < num_matrices && work >= args.g[mi + 1]) ++mi;
        const int group = work - args.g[mi];
        const int size_n = args.n[mi];
        const int ntiles = size_n / 16;
        const size_t slice_stride = (size_t) ntiles * TWORDS;
        const uint32_t* B32 = (const uint32_t*) args.B[mi];
        void* C = args.C[mi];
        const half2* A2 = (const half2*) (args.A_had + (size_t) mi * size_m * size_k);

        const uint32_t* bp = B32 + (size_t) ks0 * slice_stride + group * WNT * TWORDS + lane;
        auto ld_b_off = [&] (size_t off, int l) -> uint32_t
        {
            if constexpr (LSTRIDE < 32)
                return lane < LSTRIDE ? __ldcs(bp + off + l * LSTRIDE) : 0;
            else
                return __ldcs(bp + off + l * LSTRIDE);
        };
        auto ld_b = [&] (int i, int l) -> uint32_t { return ld_b_off((size_t) i * slice_stride, l); };
        size_t pf_off = (size_t) PF * slice_stride;

        uint32_t pf[PF][LOADS];
        #pragma unroll
        for (int d = 0; d < PF; ++d)
            if (d < myn)
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b(d, l);

        half2 dh[WNT][2][NR];
        float2 da[WNT][2][NR];
        #pragma unroll
        for (int t = 0; t < WNT; ++t)
            #pragma unroll
            for (int f = 0; f < 2; ++f)
                #pragma unroll
                for (int r = 0; r < NR; ++r)
                {
                    dh[t][f][r] = hzero;
                    da[t][f][r] = make_float2(0.0f, 0.0f);
                }

        for (int ib = 0; ib < myn; ib += PF)
        {
        #pragma unroll
        for (int d = 0; d < PF; ++d)
        {
            const int i = ib + d;
            if (i >= myn) break;

            uint32_t bw[LOADS];
            #pragma unroll
            for (int l = 0; l < LOADS; ++l)
                bw[l] = pf[d][l];

            if (i + PF < myn)
            {
                #pragma unroll
                for (int l = 0; l < LOADS; ++l)
                    pf[d][l] = ld_b_off(pf_off, l);
            }
            pf_off += slice_stride;

            // A fragments: lane covers k pairs (2(lane%4), +1) and (+8, +9) of every row
            const size_t a_col = (size_t) (ks0 + i) * 8 + (lane & 3);
            half2 x01[NR], x23[NR];
            #pragma unroll
            for (int r = 0; r < NR; ++r)
            {
                const bool ok = NR == 1 || r < size_m;
                x01[r] = ok ? A2[(size_t) r * (size_k / 2) + a_col] : hzero;
                x23[r] = ok ? A2[(size_t) r * (size_k / 2) + a_col + 4] : hzero;
            }

            #pragma unroll
            for (int t = 0; t < WNT; ++t)
            {
                FragB f0, f1;
                if constexpr (bits == 4)
                {
                    uint32_t aw = __shfl_sync(0xffffffffu, bw[t], (lane + 31) & 31);
                    exl3_gemv_ns::dq8_regs_4bits<cb>(aw, bw[t], f0, f1);
                }
                else if constexpr (bits == 2)
                {
                    const uint32_t w = bw[t >> 1];
                    const int base = (t & 1) << 4;
                    uint32_t bwv = __shfl_sync(0xffffffffu, w, base + x_src_b);
                    uint32_t awv = __shfl_sync(0xffffffffu, w, base + x_src_a);
                    exl3_gemv_ns::dq8_regs_2bits<cb>(awv, bwv, lane << 3, f0, f1);
                }
                else  // bits == 3
                {
                    uint32_t awv = __shfl_sync(0xffffffffu, bw[t], x_src_a);
                    uint32_t bwv = __shfl_sync(0xffffffffu, bw[t], x_src_b);
                    exl3_gemv_ns::dq8_regs_3bits<cb>(awv, bwv, x_s2, f0, f1);
                }
                #pragma unroll
                for (int r = 0; r < NR; ++r)
                {
                    dh[t][0][r] = __hfma2(x23[r], f0[1], __hfma2(x01[r], f0[0], dh[t][0][r]));
                    dh[t][1][r] = __hfma2(x23[r], f1[1], __hfma2(x01[r], f1[0], dh[t][1][r]));
                }
            }

            if ((d + 1) % FOLD == 0 || i + 1 == myn)
            {
                #pragma unroll
                for (int t = 0; t < WNT; ++t)
                    #pragma unroll
                    for (int f = 0; f < 2; ++f)
                        #pragma unroll
                        for (int r = 0; r < NR; ++r)
                        {
                            da[t][f][r].x += __low2float(dh[t][f][r]);
                            da[t][f][r].y += __high2float(dh[t][f][r]);
                            dh[t][f][r] = hzero;
                        }
            }
        }
        }

        // Quad reduction over the k splits, one lane per (row, column) stores into shared memory
        #pragma unroll
        for (int t = 0; t < WNT; ++t)
            #pragma unroll
            for (int f = 0; f < 2; ++f)
                #pragma unroll
                for (int r = 0; r < NR; ++r)
                {
                    float s = da[t][f][r].x + da[t][f][r].y;
                    s += __shfl_xor_sync(0xffffffffu, s, 1);
                    s += __shfl_xor_sync(0xffffffffu, s, 2);
                    if ((lane & 3) == 0)
                        sh_red[warp][r][t * 16 + f * 8 + (lane >> 2)] = s;
                }
        __syncthreads();

        const int rows_out = NR;
        for (int idx = threadIdx.x; idx < COLS * rows_out; idx += THREADS)
        {
            const int r = idx / COLS;
            const int c = idx % COLS;
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < WK; ++j)
                sum += sh_red[j][r][c];
            const int col = group * COLS + c;
            if constexpr (c_fp32) ((float*) C)[(size_t) r * size_n + col] = sum;
            else                  ((half*)  C)[(size_t) r * size_n + col] = __float2half_rn(sum);
        }
        __syncthreads();
    }

    // Output scales and Hadamard per matrix, same semantics as the regular GEMV epilogue
    grid.sync();
    {
        const int warps_grid = gridDim.x * blockDim.x / 32;
        const int this_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;
        for (int m = 0; m < num_matrices; ++m)
        {
            const int size_n = args.n[m];
            const int total_warps = size_m * size_n / 128;
            void* C = args.C[m];
            const half* svh = args.svh[m];
            for (int w = this_warp; w < total_warps; w += warps_grid)
            {
                if constexpr (c_fp32)
                    had_ff_r_128_inner<false, true>
                    (
                        ((const float*) C) + w * 128,
                        ((float*) C) + w * 128,
                        svh + (w * 128) % size_n,
                        0.088388347648f
                    );
                else
                    had_hf_r_128_inner<false, true>
                    (
                        ((const half*) C) + w * 128,
                        ((half*) C) + w * 128,
                        svh + (w * 128) % size_n,
                        0.088388347648f
                    );
            }
        }
    }
}

// Kernel instances used by the entry point: Pascal direct path, m = 1 (MMODE 0) and m = 2 (MMODE 2),
// 2..4 bpw integer mul1, narrow and wide configs, fp16 / fp32 outputs.
#define EXL3_GEMV_FUSED_INST(bits_, cb_, mm_, cfg_) \
    exl3_gemv_fused_kernel<bits_, false, cb_, mm_, cfg_>, \
    exl3_gemv_fused_kernel<bits_, true, cb_, mm_, cfg_>

using fp_gemv_fused_kernel = void (*)(GemvFusedArgs);

inline void* exl3_gemv_fused_select(int bits, int cb, bool c_fp32, int mmode, int cfg)
{
    #define SEL(bits_, cb_, fp32_, mm_, cfg_) \
        if (bits == bits_ && cb == cb_ && c_fp32 == fp32_ && mmode == mm_ && cfg == cfg_) \
            return (void*) exl3_gemv_fused_kernel<bits_, fp32_, cb_, mm_, cfg_>;
    #define SEL_GRID(bits_, cb_) \
        SEL(bits_, cb_, false, 0, 0) SEL(bits_, cb_, false, 0, 1) \
        SEL(bits_, cb_, true,  0, 0) SEL(bits_, cb_, true,  0, 1) \
        SEL(bits_, cb_, false, 2, 0) SEL(bits_, cb_, false, 2, 1) \
        SEL(bits_, cb_, true,  2, 0) SEL(bits_, cb_, true,  2, 1)
    SEL_GRID(2, 2)
    SEL_GRID(3, 2)
    SEL_GRID(4, 2)
    #undef SEL_GRID
    #undef SEL
    return nullptr;
}

}  // namespace exl3_gemv_fused_ns
