#pragma once

// Decode gated-residual mix, small-R form (EXL3_GR_MIX3, see gr_mix in hc_mix.cu).
//
// gr_dots_kernel2 holds a pair of fn rows per block (255 registers: two blocks per SM, so two
// waves on the P100 and three on the 2060) and re-reads the four streams of GR_RB rows (clamped
// rows included) from L2 for every pair: ~8 bytes of L2 traffic per fn byte. Here a block takes
// one stream h: its R stream rows go to shared memory once (pre-weighted with the norm weight
// when no weighted copy exists, the same __fmul_rn the old kernel applies in registers), and each
// warp streams one fn row against them. Per lane the chunks are the same (lane + 32 k), the FMA
// order inside a chunk is the same and the warp reduction is the same shfl_down tree, so the dots
// are bit-identical to gr_dots_kernel2.
//
// The finalize is gr_finalize_block with the row count as a template parameter (accumulators
// for R rows instead of GR_RB, so fewer registers), the up-table loads and all prologue dots
// loads issued before the first barrier, and the epilogue (sigmoid gates, stream mean) spread
// over the lanes instead of lane 0 alone. Same arithmetic in the same order, bit-identical.
// Microbenchmark, 13 MB per site, R = 1 / 2, cold tables: P100 54-70 / 58-71 us -> 35 / 40 us,
// RTX 2060 86-96 / 80-101 us -> 52-54 / 54-55 us.

#define GR3_THREADS 256                 // dots: 8 warps, one fn row each
#define GR3_WARPS (GR3_THREADS / 32)
#define GR3_MAX_R 4

template <int NCH, bool WEIGHTED, int R>
__global__ __launch_bounds__(GR3_THREADS)
void gr_dots_kernel3
(
    const float* __restrict__ streams, const float* __restrict__ wstreams, const half* __restrict__ fn,
    const half* __restrict__ wn, float* __restrict__ dots, const int M
)
{
    constexpr int H = 4;
    constexpr int D = NCH * 256;
    constexpr int D8 = D / 8;
    const int h = blockIdx.y;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int j = blockIdx.x * GR3_WARPS + warp;

    // Stream rows of this h, split into the low and high float4 of each 8-wide chunk so the
    // per-lane chunk reads below are conflict-free
    __shared__ float4 s_lo[R][D8];
    __shared__ float4 s_hi[R][D8];
    for (int idx = threadIdx.x; idx < R * D8; idx += GR3_THREADS)
    {
        const int r = idx / D8, c = idx % D8;
        const float4* src = (const float4*) ((WEIGHTED ? wstreams : streams) + ((size_t) r * H + h) * D) + 2 * c;
        float4 s0 = src[0], s1 = src[1];
        if constexpr (!WEIGHTED)
        {
            const int4 wk = ((const int4*) (wn + (size_t) h * D))[c];
            const half2* v2 = (const half2*) &wk;
            float2 v0 = __half22float2(v2[0]), v1 = __half22float2(v2[1]), v2_ = __half22float2(v2[2]), v3 = __half22float2(v2[3]);
            s0.x = __fmul_rn(s0.x, v0.x); s0.y = __fmul_rn(s0.y, v0.y); s0.z = __fmul_rn(s0.z, v1.x); s0.w = __fmul_rn(s0.w, v1.y);
            s1.x = __fmul_rn(s1.x, v2_.x); s1.y = __fmul_rn(s1.y, v2_.y); s1.z = __fmul_rn(s1.z, v3.x); s1.w = __fmul_rn(s1.w, v3.y);
        }
        s_lo[r][c] = s0;
        s_hi[r][c] = s1;
    }

    // fn row loads issued before the barrier, in flight while the stream rows land
    int4 f[NCH];
    if (j < M)
    {
        const int4* f8 = (const int4*) (fn + ((size_t) j * H + h) * D);
        #pragma unroll
        for (int k = 0; k < NCH; ++k) f[k] = f8[lane + k * 32];
    }
    __syncthreads();

    if (j < M)
    {
        float acc[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) acc[r] = 0.0f;
        #pragma unroll
        for (int k = 0; k < NCH; ++k)
        {
            const int c = lane + k * 32;
            const half2* w2 = (const half2*) &f[k];
            float2 w0 = __half22float2(w2[0]), w1 = __half22float2(w2[1]), w2_ = __half22float2(w2[2]), w3 = __half22float2(w2[3]);
            #pragma unroll
            for (int r = 0; r < R; ++r)
            {
                const float4 s0 = s_lo[r][c], s1 = s_hi[r][c];
                float a = acc[r];
                a = fmaf(s0.x, w0.x, a); a = fmaf(s0.y, w0.y, a); a = fmaf(s0.z, w1.x, a); a = fmaf(s0.w, w1.y, a);
                a = fmaf(s1.x, w2_.x, a); a = fmaf(s1.y, w2_.y, a); a = fmaf(s1.z, w3.x, a); a = fmaf(s1.w, w3.y, a);
                acc[r] = a;
            }
        }
        #pragma unroll
        for (int r = 0; r < R; ++r)
        {
            float a = acc[r];
            for (int offset = 16; offset > 0; offset >>= 1) a += __shfl_down_sync(0xffffffffu, a, offset);
            if (lane == 0) dots[((size_t) r * (M + 1) + j) * H + h] = a;
        }
    }
    else if (j == M)
    {
        {
            // Sum of squares of the raw streams, same order as gr_dots_block
            #pragma unroll
            for (int r = 0; r < R; ++r)
            {
                const float4* s4 = (const float4*) (streams + ((size_t) r * H + h) * D);
                float a = 0.0f;
                for (int c = lane; c < D / 4; c += 32)
                {
                    float4 s = s4[c];
                    a = fmaf(s.x, s.x, fmaf(s.y, s.y, fmaf(s.z, s.z, fmaf(s.w, s.w, a))));
                }
                for (int offset = 16; offset > 0; offset >>= 1) a += __shfl_down_sync(0xffffffffu, a, offset);
                if (lane == 0) dots[((size_t) r * (M + 1) + M) * H + h] = a;
            }
        }
    }
}

#define GR3_THREADS_C 128               // finalize: 4 warps, one column quad each

template <bool HALF_OUT, int NIT, int R>
__global__ __launch_bounds__(GR3_THREADS_C)
void gr_finalize_kernel3
(
    const float* __restrict__ streams, const float* __restrict__ dots, const half* __restrict__ upt,
    const half* __restrict__ w, float* __restrict__ post, void* __restrict__ mixed,
    const int D, const int LR, const float rms_eps
)
{
    constexpr int H = 4;
    const int M = LR + (post ? H : 0);
    __shared__ float t_s[R * NIT * 64];
    __shared__ float rmr_s[R * H];
    const int tid = threadIdx.x;
    const int lane = tid % 32, warp = tid / 32;
    const int c = blockIdx.x * (GR3_THREADS_C / 32) + warp;     // column quad
    const int D4 = D / 4;
    const bool col_ok = c < D4;

    // Up-table loads first: independent of the dots
    int4 u4[NIT][H];
    #pragma unroll
    for (int it = 0; it < NIT; ++it)
    {
        const int i0 = it * 64 + lane * 2;
        #pragma unroll
        for (int h = 0; h < H; ++h)
            u4[it][h] = (col_ok && i0 < LR) ? *(const int4*) (upt + ((((size_t) h * D4 + c) * LR + i0) * 4)) : make_int4(0, 0, 0, 0);
    }

    // Prologue: every dots value this thread needs is loaded before the first barrier (one L2
    // round trip instead of one per loop step)
    constexpr int TIT = (R * NIT * 64 + GR3_THREADS_C - 1) / GR3_THREADS_C;
    float4 dv[TIT];
    #pragma unroll
    for (int k = 0; k < TIT; ++k)
    {
        const int idx = tid + k * GR3_THREADS_C;
        const int r = idx / LR, i = idx % LR;
        dv[k] = idx < R * LR ? *(const float4*) (dots + ((size_t) r * (M + 1) + i) * H) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    if (tid < R * H)
    {
        const int r = tid / H, h = tid % H;
        rmr_s[tid] = rsqrtf(dots[((size_t) r * (M + 1) + M) * H + h] / (float) D + rms_eps);
    }
    __syncthreads();
    const float inv_h = 1.0f / (float) H;
    #pragma unroll
    for (int k = 0; k < TIT; ++k)
    {
        const int idx = tid + k * GR3_THREADS_C;
        if (idx >= R * LR) break;
        const int r = idx / LR;
        float v = 0.0f;
        v = fmaf(rmr_s[r * H + 0], dv[k].x, v);
        v = fmaf(rmr_s[r * H + 1], dv[k].y, v);
        v = fmaf(rmr_s[r * H + 2], dv[k].z, v);
        v = fmaf(rmr_s[r * H + 3], dv[k].w, v);
        v *= inv_h;
        t_s[idx] = v * sigmoidf_(v);
    }
    if (post && blockIdx.x == 0 && tid < R * H)
    {
        const int r = tid / H, h = tid % H;
        const float* dr = dots + ((size_t) r * (M + 1) + LR + h) * H;
        float v = 0.0f;
        #pragma unroll
        for (int hh = 0; hh < H; ++hh) v = fmaf(rmr_s[r * H + hh], dr[hh], v);
        post[tid] = 2.0f * sigmoidf_(v * inv_h);
    }
    __syncthreads();
    if (!col_ok) return;

    float4 g[R][H];
    #pragma unroll
    for (int r = 0; r < R; ++r)
        #pragma unroll
        for (int h = 0; h < H; ++h) g[r][h] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    #pragma unroll
    for (int it = 0; it < NIT; ++it)
    {
        const int i0 = it * 64 + lane * 2;              // ranks i0, i0 + 1
        #pragma unroll
        for (int r = 0; r < R; ++r)
        {
            const float t0 = i0 < LR ? t_s[r * LR + i0] : 0.0f;
            const float t1 = i0 + 1 < LR ? t_s[r * LR + i0 + 1] : 0.0f;
            #pragma unroll
            for (int h = 0; h < H; ++h)
            {
                const half2* u2 = (const half2*) &u4[it][h];
                float2 a0 = __half22float2(u2[0]), a1 = __half22float2(u2[1]);   // rank i0: 4 columns
                float2 b0 = __half22float2(u2[2]), b1 = __half22float2(u2[3]);   // rank i0 + 1
                g[r][h].x = fmaf(t0, a0.x, fmaf(t1, b0.x, g[r][h].x));
                g[r][h].y = fmaf(t0, a0.y, fmaf(t1, b0.y, g[r][h].y));
                g[r][h].z = fmaf(t0, a1.x, fmaf(t1, b1.x, g[r][h].z));
                g[r][h].w = fmaf(t0, a1.y, fmaf(t1, b1.y, g[r][h].w));
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; ++r)
        #pragma unroll
        for (int h = 0; h < H; ++h)
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                g[r][h].x += __shfl_xor_sync(0xffffffffu, g[r][h].x, offset);
                g[r][h].y += __shfl_xor_sync(0xffffffffu, g[r][h].y, offset);
                g[r][h].z += __shfl_xor_sync(0xffffffffu, g[r][h].z, offset);
                g[r][h].w += __shfl_xor_sync(0xffffffffu, g[r][h].w, offset);
            }
    // Epilogue spread over the lanes (the butterfly left every sum in every lane): lane e of a
    // pass owns element (r, x) = (e / 16, e / 4 % 4) and stream h = e % 4; the four lanes of an
    // element combine their terms in stream order with the same FMA chain as gr_finalize_block
    #pragma unroll
    for (int pass = 0; pass < (16 * R + 31) / 32; ++pass)
    {
        const int e = pass * 32 + lane;
        const int r = min(e / 16, R - 1), x = e / 4 % 4, h = e % 4;
        float gv = 0.0f;
        #pragma unroll
        for (int rr = 0; rr < R; ++rr)
            #pragma unroll
            for (int hh = 0; hh < H; ++hh)
            {
                const float4 q = g[rr][hh];
                const float qv = x == 0 ? q.x : x == 1 ? q.y : x == 2 ? q.z : q.w;
                gv = (rr == r && hh == h) ? qv : gv;
            }
        const float sv = streams[((size_t) r * H + h) * D + 4 * c + x];
        const float wv = __half2float(w[(size_t) h * D + 4 * c + x]);
        const float coef = rmr_s[r * H + h] * inv_h;
        const float term = sigmoidf_(gv) * coef * wv;
        const int base = lane & ~3;
        float o = 0.0f;
        #pragma unroll
        for (int hh = 0; hh < H; ++hh)
            o = fmaf(__shfl_sync(0xffffffffu, term, base + hh), __shfl_sync(0xffffffffu, sv, base + hh), o);
        if (h == 0 && e < 16 * R)
        {
            if (HALF_OUT) ((half*) mixed)[(size_t) r * D + 4 * c + x] = __float2half_rn(o);
            else ((float*) mixed)[(size_t) r * D + 4 * c + x] = o;
        }
    }
}

// Launch the pair; false when the shape is not covered (caller takes the older kernels).
// part: bit 0 dots, bit 1 finalize (benchmarking), 3 for both.
static inline bool gr_mix3_launch
(
    const float* streams, const float* wstreams, const half* fn, const half* w, const half* upt,
    float* dots, float* post, void* mixed, bool half_out,
    int M, int D, int LR, int R, float rms_eps, cudaStream_t stream, int part = 3
)
{
    const int nch = D / 256, nit = (LR + 63) / 64;
    if (R < 1 || R > GR3_MAX_R || D % 256 || LR % 2) return false;
    if (nch != 10 || nit != 5) return false;      // instantiated for the Qwen3.8 shape (D 2560, LR 320) only

    if (part & 1)
    {
        dim3 grid_a((M + 1 + GR3_WARPS - 1) / GR3_WARPS, 4);
        #define DOTS3(NCH, WTD, RR) gr_dots_kernel3<NCH, WTD, RR><<<grid_a, GR3_THREADS, 0, stream>>>(streams, wstreams, fn, w, dots, M)
        #define DOTS3_R(NCH, WTD) switch (R) { case 1: DOTS3(NCH, WTD, 1); break; case 2: DOTS3(NCH, WTD, 2); break; \
                                               case 3: DOTS3(NCH, WTD, 3); break; case 4: DOTS3(NCH, WTD, 4); break; }
        if (wstreams) { DOTS3_R(10, true) }
        else          { DOTS3_R(10, false) }
        #undef DOTS3_R
        #undef DOTS3
    }
    if (part & 2)
    {
        const int grid_c = D / 4 / (GR3_THREADS_C / 32);
        #define FIN3(HO, NIT, RR) gr_finalize_kernel3<HO, NIT, RR><<<grid_c, GR3_THREADS_C, 0, stream>>>(streams, dots, upt, w, post, mixed, D, LR, rms_eps)
        #define FIN3_R(HO, NIT) switch (R) { case 1: FIN3(HO, NIT, 1); break; case 2: FIN3(HO, NIT, 2); break; \
                                             case 3: FIN3(HO, NIT, 3); break; case 4: FIN3(HO, NIT, 4); break; }
        if (half_out) { FIN3_R(true, 5) }
        else          { FIN3_R(false, 5) }
        #undef FIN3_R
        #undef FIN3
    }
    return true;
}
