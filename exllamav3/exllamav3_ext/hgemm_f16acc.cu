// Dense fp16 GEMM with fp16-accumulator tensor-core MMA and fp32 accumulation across K.
//
// GeForce parts run mma.sync with fp32 accumulators at half the rate of the fp16-accumulator
// form. This kernel runs the MMA with fp16 accumulators and flushes the partial sums into fp32
// registers every KSLICE elements of K, so the accumulation across K stays fp32 and the error
// is that of 32-term fp16 partials. Used by the reconstruct (prefill) paths through hgemm_recon /
// hgemm_batched, which fall back to cuBLAS when the device gains nothing (rate probe) or the
// shape is not covered or too small.
//
// C[b] = A[b] @ B[b], row-major, A [M, K], B [K, N], C [M, N] (fp16 or fp32 output), optional
// strided batch. Requires K % 64 == 0, N % 128 == 0, packed rows; M arbitrary.

#include <cuda_fp16.h>
#include "hgemm.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include "quant/exl3_devctx.cuh"
#include <mutex>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <cmath>

namespace f16acc
{

constexpr int BM = 128, BN = 128, BK = 64, PAD = 8;
constexpr int WARPS_M = 2, WARPS_N = 4, THREADS = WARPS_M * WARPS_N * 32;
constexpr int STAGES = 2, KSLICE = 32, GROUP_M = 8;
constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
constexpr int MT = WM / 16, NT = WN / 8;
constexpr int AS_STRIDE = BK + PAD, BS_STRIDE = BN + PAD;
constexpr int A_STAGE = BM * AS_STRIDE, B_STAGE = BK * BS_STRIDE;
constexpr size_t SMEM_BYTES = (size_t) STAGES * (A_STAGE + B_STAGE) * sizeof(half);
constexpr int MIN_ROWS = 384;

// Preserve the existing configuration on pre-Blackwell GPUs. The two tuned layouts
// share the 32-term partial-sum interval and differ only in scheduling and storage.
template <int TILE_N, bool TUNED> struct Config
{
    static constexpr int BM = 128, BN = TILE_N, BK = 64, PAD = TUNED ? 0 : 8;
    static constexpr int WARPS_M = 2, WARPS_N = BN / 32;
    static constexpr int THREADS = WARPS_M * WARPS_N * 32;
    static constexpr int STAGES = 2, GROUP_M = TUNED ? 16 : 8;
    static constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
    static constexpr int MT = WM / 16, NT = WN / 8;
    static constexpr int AS_STRIDE = BK + PAD, BS_STRIDE = BN + PAD;
    static constexpr int A_STAGE = BM * AS_STRIDE, B_STAGE = BK * BS_STRIDE;
    static constexpr size_t SMEM_BYTES = (size_t) STAGES * (A_STAGE + B_STAGE) * sizeof(half);
};

__device__ __forceinline__ void add_half_pair(float& a, float& b, uint32_t h)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    // PTX 8.6+: convert an FP16 partial and add it to the FP32 total in one instruction.
    asm volatile("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %2; "
                 "add.rn.f32.f16 %0, lo, %0; add.rn.f32.f16 %1, hi, %1; }"
                 : "+f"(a), "+f"(b) : "r"(h));
#else
    float2 f = __half22float2(*reinterpret_cast<half2*>(&h));
    a += f.x;
    b += f.y;
#endif
}


__device__ __forceinline__ uint32_t smem_u32(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool pred)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    int src_size = pred ? 16 : 0;    // 0 -> zero-fill, no global read
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem_u32(smem)), "l"(gmem), "r"(src_size));
#else
    (void) smem; (void) gmem; (void) pred;
#endif
}
__device__ __forceinline__ void cp_async_commit()
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}
template <int N> __device__ __forceinline__ void cp_async_wait()
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#else
    (void) N;
#endif
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t* r, const void* p)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
#else
    (void) r; (void) p;
#endif
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t* r, const void* p)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_u32(p)));
#else
    (void) r; (void) p;
#endif
}
__device__ __forceinline__ void mma_f16(uint32_t* c, const uint32_t* a, const uint32_t* b)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    (void) c; (void) a; (void) b;
#endif
}
__device__ __forceinline__ void mma_f32(float* c, const uint32_t* a, const uint32_t* b)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    (void) c; (void) a; (void) b;
#endif
}

template <bool OUT_F32, int TILE_N = 128, bool TUNED = false>
__global__ void __launch_bounds__((Config<TILE_N, TUNED>::THREADS), 1)
gemm_kernel
(
    const half* __restrict__ A, const half* __restrict__ B, void* __restrict__ C,
    int M, int N, int K, int ldc,
    long long strideA, long long strideB, long long strideC
)
{
    using CF = Config<TILE_N, TUNED>;
    constexpr int BM = CF::BM, BN = CF::BN, BK = CF::BK, THREADS = CF::THREADS;
    constexpr int STAGES = CF::STAGES, GROUP_M = CF::GROUP_M;
    constexpr int WARPS_M = CF::WARPS_M, WARPS_N = CF::WARPS_N;
    constexpr int WM = CF::WM, WN = CF::WN, MT = CF::MT, NT = CF::NT;
    constexpr int AS_STRIDE = CF::AS_STRIDE, BS_STRIDE = CF::BS_STRIDE;
    constexpr int A_STAGE = CF::A_STAGE, B_STAGE = CF::B_STAGE;

    extern __shared__ __align__(16) half smem[];
    half* As = smem;
    half* Bs = smem + STAGES * A_STAGE;

    // Grouped raster: consecutive blocks walk GROUP_M M-stripes over one N column before moving
    // on, so the B stripe stays L2-hot across GROUP_M A-stripes
    const int grid_n = gridDim.x, grid_m = gridDim.y;
    const int bid = blockIdx.y * grid_n + blockIdx.x;
    const int group_size = GROUP_M * grid_n;
    const int group = bid / group_size;
    const int first_m = group * GROUP_M;
    const int gm_eff = min(GROUP_M, grid_m - first_m);
    const int in_group = bid - group * group_size;
    const int bm = (first_m + in_group % gm_eff) * BM;
    const int bn = (in_group / gm_eff) * BN;

    const int bz = blockIdx.z;
    A += (long long) bz * strideA;
    B += (long long) bz * strideB;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp / WARPS_N, wn = warp % WARPS_N;

    constexpr int A_CPR = BK / 8, B_CPR = BN / 8;
    constexpr int A_ITERS = BM * A_CPR / THREADS, B_ITERS = BK * B_CPR / THREADS;
    auto load_tile = [&](int stage, int kt)
    {
        const int k0 = kt * BK;
        half* as = As + stage * A_STAGE;
        half* bs = Bs + stage * B_STAGE;
        #pragma unroll
        for (int i = 0; i < A_ITERS; ++i)
        {
            int c = tid + i * THREADS;
            int row = c / A_CPR, chunk = c % A_CPR;
            int grow = bm + row;
            bool pred = grow < M;
            const half* src = A + (long long) (pred ? grow : 0) * K + k0 + chunk * 8;
            if constexpr (TUNED) cp_async16(as + row * AS_STRIDE + (chunk ^ (row % A_CPR)) * 8, src, pred);
            else cp_async16(as + row * AS_STRIDE + chunk * 8, src, pred);
        }
        #pragma unroll
        for (int i = 0; i < B_ITERS; ++i)
        {
            int c = tid + i * THREADS;
            int row = c / B_CPR, chunk = c % B_CPR;
            const half* src = B + (long long) (k0 + row) * N + bn + chunk * 8;
            if constexpr (TUNED) cp_async16(bs + row * BS_STRIDE + (chunk ^ (row % B_CPR)) * 8, src, true);
            else cp_async16(bs + row * BS_STRIDE + chunk * 8, src, true);
        }
    };

    float acc[MT][NT][4];
    uint32_t hacc[MT][NT][2];
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j)
        {
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
            hacc[i][j][0] = hacc[i][j][1] = 0u;
        }
    auto flush = [&]()
    {
        #pragma unroll
        for (int i = 0; i < MT; ++i)
            #pragma unroll
            for (int j = 0; j < NT; ++j)
            {
                float2 f0 = __half22float2(*reinterpret_cast<half2*>(&hacc[i][j][0]));
                float2 f1 = __half22float2(*reinterpret_cast<half2*>(&hacc[i][j][1]));
                acc[i][j][0] += f0.x; acc[i][j][1] += f0.y;
                acc[i][j][2] += f1.x; acc[i][j][3] += f1.y;
                hacc[i][j][0] = 0u; hacc[i][j][1] = 0u;
            }
    };

    const int KT = K / BK;
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
    {
        if (s < KT) load_tile(s, s);
        cp_async_commit();
    }

    const int a_lrow = lane & 15, a_lcol = (lane >> 4) * 8;
    const int b_lrow = (lane & 7) + ((lane >> 3) & 1) * 8, b_lcol = (lane >> 4) * 8;
    constexpr int KK = BK / 16;
    static_assert(KSLICE == 16 * 2, "flush cadence below assumes kslice 32 = two k16 steps");

    uint32_t af[2][MT][4];
    uint32_t bf[2][NT][2];
    auto load_frags = [&](int buf, const half* as, const half* bs, int kk)
    {
        #pragma unroll
        for (int i = 0; i < MT; ++i)
            if constexpr (TUNED)
                ldmatrix_x4(af[buf][i], as + (i * 16 + a_lrow) * AS_STRIDE +
                            (((kk * 16 + a_lcol) / 8) ^ ((i * 16 + a_lrow) % A_CPR)) * 8);
            else
                ldmatrix_x4(af[buf][i], as + (i * 16 + a_lrow) * AS_STRIDE + kk * 16 + a_lcol);
        #pragma unroll
        for (int j = 0; j < NT; j += 2)
        {
            uint32_t r[4];
            if constexpr (TUNED)
                ldmatrix_x4_trans(r, bs - wn * WN + (kk * 16 + b_lrow) * BS_STRIDE +
                                  (((wn * WN + j * 8 + b_lcol) / 8) ^ ((kk * 16 + b_lrow) % B_CPR)) * 8);
            else
                ldmatrix_x4_trans(r, bs + (kk * 16 + b_lrow) * BS_STRIDE + j * 8 + b_lcol);
            bf[buf][j][0] = r[0]; bf[buf][j][1] = r[1]; bf[buf][j + 1][0] = r[2]; bf[buf][j + 1][1] = r[3];
        }
    };

    for (int kt = 0; kt < KT; ++kt)
    {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            int nk = kt + STAGES - 1;
            if (nk < KT) load_tile(nk % STAGES, nk);
            cp_async_commit();
        }
        const half* as = As + (kt % STAGES) * A_STAGE + wm * WM * AS_STRIDE;
        const half* bs = Bs + (kt % STAGES) * B_STAGE + wn * WN;

        if constexpr (TUNED)
        {
            // Preserve each pair of k16 MMAs and the FP32 addition order, but flush a
            // fragment immediately so conversions/additions can overlap other MMAs.
            #pragma unroll
            for (int kk = 0; kk < KK; kk += 2)
            {
                load_frags(0, as, bs, kk);
                load_frags(1, as, bs, kk + 1);
                #pragma unroll
                for (int i = 0; i < MT; ++i)
                    #pragma unroll
                    for (int j = 0; j < NT; ++j)
                    {
                        uint32_t h[2] = {};
                        mma_f16(h, af[0][i], bf[0][j]);
                        mma_f16(h, af[1][i], bf[1][j]);
                        add_half_pair(acc[i][j][0], acc[i][j][1], h[0]);
                        add_half_pair(acc[i][j][2], acc[i][j][3], h[1]);
                    }
            }
        }
        else
        {
            load_frags(0, as, bs, 0);
            #pragma unroll
            for (int kk = 0; kk < KK; ++kk)
            {
                if (kk + 1 < KK) load_frags((kk + 1) & 1, as, bs, kk + 1);
                const int cur = kk & 1;
                #pragma unroll
                for (int i = 0; i < MT; ++i)
                    #pragma unroll
                    for (int j = 0; j < NT; ++j)
                        mma_f16(hacc[i][j], af[cur][i], bf[cur][j]);
                if (kk & 1) flush();     // every 32 of K
            }
        }
    }

    const int g = lane >> 2, t = lane & 3;
    #pragma unroll
    for (int i = 0; i < MT; ++i)
        #pragma unroll
        for (int j = 0; j < NT; ++j)
        {
            int col = bn + wn * WN + j * 8 + t * 2;
            #pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                int row = bm + wm * WM + i * 16 + g + h * 8;
                if (row >= M) continue;
                float v0 = acc[i][j][h * 2], v1 = acc[i][j][h * 2 + 1];
                if constexpr (OUT_F32)
                {
                    float* c = reinterpret_cast<float*>(C) + (long long) bz * strideC + (long long) row * ldc + col;
                    *reinterpret_cast<float2*>(c) = make_float2(v0, v1);
                }
                else
                {
                    half* c = reinterpret_cast<half*>(C) + (long long) bz * strideC + (long long) row * ldc + col;
                    *reinterpret_cast<half2*>(c) = __floats2half2_rn(v0, v1);
                }
            }
        }
}

// Rate probe: independent register-resident MMA chains, no memory traffic
template <bool F16>
__global__ void __launch_bounds__(256) rate_kernel(int iters, float* sink)
{
    uint32_t a0 = threadIdx.x, a1 = a0 + 1, a2 = a0 + 2, a3 = a0 + 3, b0 = a0 + 5, b1 = a0 + 7;
    uint32_t a[4] = { a0, a1, a2, a3 }, b[2] = { b0, b1 };
    uint32_t h[8][2] = {};
    float f[8][4] = {};
    for (int i = 0; i < iters; ++i)
    {
        #pragma unroll
        for (int c = 0; c < 8; ++c)
        {
            if constexpr (F16) mma_f16(h[c], a, b);
            else mma_f32(f[c], a, b);
        }
    }
    float s = 0.f;
    #pragma unroll
    for (int c = 0; c < 8; ++c) s += F16 ? __uint_as_float(h[c][0] ^ h[c][1]) : f[c][0] + f[c][3];
    if (s == 123.456f) sink[threadIdx.x] = s;
}

// Per-device decision: -1 unknown, 0 off, 1 on
static int g_enabled[MAX_DEVICES];
static bool g_init = false;
static std::mutex g_mutex;

static float probe_ms(bool f16, int blocks, int iters, float* sink, cudaStream_t stream)
{
    auto run = [&]() { if (f16) rate_kernel<true><<<blocks, 256, 0, stream>>>(iters, sink);
                       else rate_kernel<false><<<blocks, 256, 0, stream>>>(iters, sink); };
    run();
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0, stream);
    for (int r = 0; r < 3; ++r) run();
    cudaEventRecord(e1, stream);
    cudaEventSynchronize(e1);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms;
}

bool enabled(int device)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_init)
    {
        for (int i = 0; i < MAX_DEVICES; ++i) g_enabled[i] = -1;
        g_init = true;
    }
    if (device < 0 || device >= MAX_DEVICES) return false;
    if (g_enabled[device] >= 0) return g_enabled[device] == 1;

    int on = 0;
    const char* env = std::getenv("EXL3_HGEMM_F16ACC");
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    if (major < 8)
        on = 0;                                   // cp.async / ldmatrix.x4 / m16n8k16 need sm80+
    else if (env && std::strcmp(env, "0") == 0)
        on = 0;
    else if (env && std::strcmp(env, "1") == 0)
        on = 1;
    else
    {
        // Auto: enable where the fp16-accumulator MMA is at least 1.5x faster than the
        // fp32-accumulator one (GeForce: 2.0x; workstation / datacenter parts: 1.0x)
        const at::cuda::CUDAGuard guard(device);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream(device).stream();
        int sms = 1;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
        float* sink = nullptr;
        cudaMalloc(&sink, 256 * sizeof(float));
        float t32 = probe_ms(false, sms * 4, 512, sink, stream);
        float t16 = probe_ms(true, sms * 4, 512, sink, stream);
        cudaFree(sink);
        cudaGetLastError();
        on = (t32 > 0.f && t16 > 0.f && t32 / t16 >= 1.5f) ? 1 : 0;
    }
    g_enabled[device] = on;
    return on == 1;
}

// Devices, dtypes, ranks, matching shapes and the tile multiples of K and N
static bool covered_shapes(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c)
{
    if (!a.is_cuda() || a.device() != b.device() || a.device() != c.device()) return false;
    if (a.dtype() != at::kHalf || b.dtype() != at::kHalf) return false;
    if (c.dtype() != at::kHalf && c.dtype() != at::kFloat) return false;
    if (a.dim() != b.dim() || a.dim() != c.dim() || (a.dim() != 2 && a.dim() != 3)) return false;
    if (a.dim() == 3 && (a.size(0) != b.size(0) || a.size(0) != c.size(0))) return false;
    int64_t M = a.size(-2), K = a.size(-1), N = b.size(-1);
    if (b.size(-2) != K || c.size(-2) != M || c.size(-1) != N) return false;
    return K % BK == 0 && N % BN == 0;
}

// Hard shape coverage of the kernel (independent of the device decision).
static bool covered(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c)
{
    if (!covered_shapes(a, b, c)) return false;
    int64_t M = a.size(-2), K = a.size(-1), N = b.size(-1);
    if (K < 1 || N < 1 || M < 1) return false;
    if (M > std::numeric_limits<int>::max() || N > std::numeric_limits<int>::max() ||
        K > std::numeric_limits<int>::max()) return false;
    // CUDA grid.y and grid.z are limited to 65535. Keep every narrowing conversion checked.
    if ((M + BM - 1) / BM > 65535) return false;
    if (a.stride(-1) != 1 || b.stride(-1) != 1 || c.stride(-1) != 1) return false;
    if (a.stride(-2) != K || b.stride(-2) != N) return false;
    if (c.stride(-2) < N || c.stride(-2) > std::numeric_limits<int>::max() ||
        c.stride(-2) % 2 != 0) return false;
    const uintptr_t output_alignment = c.dtype() == at::kFloat ? 8 : 4;
    if (((uintptr_t) a.data_ptr() & 15) || ((uintptr_t) b.data_ptr() & 15) ||
        ((uintptr_t) c.data_ptr() & (output_alignment - 1))) return false;
    if (a.dim() == 3)
    {
        if (a.size(0) < 1 || a.size(0) > 65535 || a.size(0) != b.size(0) || a.size(0) != c.size(0)) return false;
        // Inputs may broadcast across batches; each batch must still start at a copy-aligned address.
        if (a.stride(0) % 8 || b.stride(0) % 8 || c.stride(0) % 2) return false;
        if (a.size(0) > 1 && c.stride(0) < (M - 1) * c.stride(-2) + N) return false;
    }
    return at::cuda::getDeviceProperties(a.device().index())->major >= 8;
}

static bool tuned_device(int device)
{
    // The layout/shape sweep was measured on GeForce Blackwell. Other Ampere+ parts retain
    // their existing layout; the rate probe still determines whether FP16 MMA is worthwhile.
    return at::cuda::getDeviceProperties(device)->major == 12;
}

static bool narrow_tile(const at::Tensor& a, const at::Tensor& b)
{
    const int64_t M = a.size(-2), K = a.size(-1), N = b.size(-1);
    const int64_t batch = a.dim() == 3 ? a.size(0) : 1;
    // Narrow grids, short rows and long reductions favor 128x64. Wide expansions and
    // batched 128+-row experts favor 128x128. Avoid a benchmark/synchronization during capture.
    if (N <= 1024 || M < 128) return true;
    if (batch > 1 || N > 4096) return false;
    return M < 512 || K > N || (M >= 1024 && M < 4096);
}

static bool worthwhile(const at::Tensor& a, const at::Tensor& b)
{
    int device = a.device().index();
    if (device < 0 || device >= MAX_DEVICES) return false;
    const auto* props = at::cuda::getDeviceProperties(device);
    int64_t M = a.size(-2), N = b.size(-1), K = a.size(-1);
    int64_t batch = a.dim() == 3 ? a.size(0) : 1;
    if (tuned_device(device) && M >= 64 && M * batch >= MIN_ROWS && N >= 1024 && K >= 512)
    {
        const int tile_n = narrow_tile(a, b) ? 64 : 128;
        int64_t blocks = ((M + BM - 1) / BM) * (N / tile_n) * batch;
        // The measured small-M and batched wins include grids below one block per SM.
        return blocks * 2 >= props->multiProcessorCount;
    }
    if (M < MIN_ROWS) return false;
    int64_t blocks = ((M + BM - 1) / BM) * (N / BN) * batch;
    return blocks >= props->multiProcessorCount;
}

template <bool OUT_F32, int TILE_N, bool TUNED>
static void launch_config(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c, cudaStream_t stream)
{
    using CF = Config<TILE_N, TUNED>;
    auto kern = gemm_kernel<OUT_F32, TILE_N, TUNED>;
    static std::once_flag attr_set[MAX_DEVICES];
    int device = a.device().index();
    TORCH_CHECK(device >= 0 && device < MAX_DEVICES, "hgemm_f16acc: device index");
    std::call_once(attr_set[device], [&]()
    {
        cuda_check(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) CF::SMEM_BYTES));
    });
    bool batched = a.dim() == 3;
    int batch = batched ? a.size(0) : 1;
    int M = a.size(-2), K = a.size(-1), N = b.size(-1);
    dim3 grid(N / CF::BN, (M + CF::BM - 1) / CF::BM, batch);
    kern<<<grid, CF::THREADS, CF::SMEM_BYTES, stream>>>(
        (const half*) a.data_ptr(), (const half*) b.data_ptr(), c.data_ptr(),
        M, N, K, (int) c.stride(-2),
        batched ? a.stride(0) : 0, batched ? b.stride(0) : 0, batched ? c.stride(0) : 0);
    cuda_check(cudaPeekAtLastError());
}

template <bool OUT_F32>
static void launch(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c, cudaStream_t stream)
{
    // Ada (sm_89) runs the narrow tuned layout for every shape: its smaller L2 favours the
    // 128x64 tile with the wider raster group, while Ampere loses with it (issue #384)
    const auto* props = at::cuda::getDeviceProperties(a.device().index());
    if (props->major == 8 && props->minor == 9) launch_config<OUT_F32, 64, true>(a, b, c, stream);
    else if (!tuned_device(a.device().index())) launch_config<OUT_F32, 128, false>(a, b, c, stream);
    else if (narrow_tile(a, b)) launch_config<OUT_F32, 64, true>(a, b, c, stream);
    else launch_config<OUT_F32, 128, true>(a, b, c, stream);
}

} // namespace f16acc

// Pascal sm_60 (GP100) form of the same contract: no tensor cores, but HFMA2 runs at twice the fp32 FMA rate,
// which cuBLAS only uses with an fp16 compute type (fp16 accumulation across all of K). This kernel keeps
// fp16 partials over KSLICE = 32 elements of K and adds them into fp32 registers, like the f16acc kernel
// above. 128 x 128 x 32 block tile, 256 threads, 8 x 8 outputs per thread (rows ty*4 + {0..3} and
// 64 + ty*4 + {0..3}, columns tx*4 + {0..3} and 64 + tx*4 + {0..3}). A is staged transposed in shared memory
// with each element duplicated into a half2 so one LDS.128 feeds four broadcast operands; B rows are staged
// as loaded. Global loads for the next K tile are issued before the current tile's math (register double
// buffering, two shared stages). Requires K % 32 == 0, N % 128 == 0, 16-byte aligned packed A / B rows.
namespace hfma2
{

constexpr int BM = 128, BN = 128, BK = 32, THREADS = 256;

template <bool OUT_F32>
__global__ void __launch_bounds__(THREADS, 1)
gemm_kernel
(
    const half* __restrict__ A,
    const half* __restrict__ B,
    void* __restrict__ C,
    const int M,
    const int N,
    const int K,
    const int ldc,
    const int64_t a_bstride,
    const int64_t b_bstride,
    const int64_t c_bstride
)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 600
    __shared__ __align__(16) half2 As[2][BK][BM];
    __shared__ __align__(16) half Bs[2][BK][BN];

    A += blockIdx.z * a_bstride;
    B += blockIdx.z * b_bstride;

    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

    // Global load mapping, two 16-deep halves of the K tile: A row tid / 2, k (tid & 1) * 8 + {0, 16};
    // B row tid / 16 + {0, 16}, columns (tid & 15) * 8
    const int a_row = tid >> 1, a_k = (tid & 1) * 8;
    const bool a_ok = m0 + a_row < M;
    const half* a_ptr = A + (size_t) (a_ok ? m0 + a_row : 0) * K + a_k;
    const int b_k = tid >> 4, b_n = (tid & 15) * 8;
    const half* b_ptr = B + (size_t) b_k * N + n0 + b_n;

    half2 acc[8][4];
    float accf[8][8];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        #pragma unroll
        for (int j = 0; j < 8; ++j) accf[i][j] = 0.0f;
    }

    uint4 ra[2], rb[2];
    auto gload = [&] (int kt)
    {
        #pragma unroll
        for (int l = 0; l < 2; ++l)
        {
            ra[l] = a_ok ? *reinterpret_cast<const uint4*>(a_ptr + kt * BK + l * 16) : make_uint4(0, 0, 0, 0);
            rb[l] = *reinterpret_cast<const uint4*>(b_ptr + (size_t) (kt * BK + l * 16) * N);
        }
    };
    auto sstore = [&] (int buf)
    {
        #pragma unroll
        for (int l = 0; l < 2; ++l)
        {
            const half* ha = reinterpret_cast<const half*>(&ra[l]);
            #pragma unroll
            for (int i = 0; i < 8; ++i) As[buf][l * 16 + a_k + i][a_row] = __half2half2(ha[i]);
            *reinterpret_cast<uint4*>(&Bs[buf][l * 16 + b_k][b_n]) = rb[l];
        }
    };

    const int KT = K / BK;
    gload(0);
    sstore(0);
    __syncthreads();

    for (int kt = 0; kt < KT; ++kt)
    {
        const int buf = kt & 1;
        if (kt + 1 < KT) gload(kt + 1);

        // One fp16 partial per K tile (KSLICE = BK = 32 terms)
        #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            #pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = __float2half2_rn(0.0f);
        }

        #pragma unroll
        for (int k = 0; k < BK; ++k)
        {
            const uint4 a0 = *reinterpret_cast<const uint4*>(&As[buf][k][ty * 4]);
            const uint4 a1 = *reinterpret_cast<const uint4*>(&As[buf][k][64 + ty * 4]);
            const uint2 b0 = *reinterpret_cast<const uint2*>(&Bs[buf][k][tx * 4]);
            const uint2 b1 = *reinterpret_cast<const uint2*>(&Bs[buf][k][64 + tx * 4]);
            const half2* av0 = reinterpret_cast<const half2*>(&a0);
            const half2* av1 = reinterpret_cast<const half2*>(&a1);
            const half2* bv0 = reinterpret_cast<const half2*>(&b0);
            const half2* bv1 = reinterpret_cast<const half2*>(&b1);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                acc[i][0] = __hfma2(av0[i], bv0[0], acc[i][0]);
                acc[i][1] = __hfma2(av0[i], bv0[1], acc[i][1]);
                acc[i][2] = __hfma2(av0[i], bv1[0], acc[i][2]);
                acc[i][3] = __hfma2(av0[i], bv1[1], acc[i][3]);
                acc[4 + i][0] = __hfma2(av1[i], bv0[0], acc[4 + i][0]);
                acc[4 + i][1] = __hfma2(av1[i], bv0[1], acc[4 + i][1]);
                acc[4 + i][2] = __hfma2(av1[i], bv1[0], acc[4 + i][2]);
                acc[4 + i][3] = __hfma2(av1[i], bv1[1], acc[4 + i][3]);
            }
        }

        #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            #pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                const float2 f = __half22float2(acc[i][j]);
                accf[i][2 * j] += f.x;
                accf[i][2 * j + 1] += f.y;
            }
        }

        if (kt + 1 < KT) sstore(buf ^ 1);
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        const int m = m0 + (i < 4 ? ty * 4 + i : 64 + ty * 4 + i - 4);
        if (m >= M) continue;
        if constexpr (OUT_F32)
        {
            float* c = reinterpret_cast<float*>(C) + blockIdx.z * c_bstride + (size_t) m * ldc + n0;
            *reinterpret_cast<float4*>(c + tx * 4) = make_float4(accf[i][0], accf[i][1], accf[i][2], accf[i][3]);
            *reinterpret_cast<float4*>(c + 64 + tx * 4) = make_float4(accf[i][4], accf[i][5], accf[i][6], accf[i][7]);
        }
        else
        {
            half* c = reinterpret_cast<half*>(C) + blockIdx.z * c_bstride + (size_t) m * ldc + n0;
            half2 h[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) h[j] = __floats2half2_rn(accf[i][2 * j], accf[i][2 * j + 1]);
            *reinterpret_cast<uint2*>(c + tx * 4) = *reinterpret_cast<const uint2*>(&h[0]);
            *reinterpret_cast<uint2*>(c + 64 + tx * 4) = *reinterpret_cast<const uint2*>(&h[2]);
        }
    }
#endif
}

static int g_enabled[MAX_DEVICES];
static bool g_init = false;
static std::mutex g_mutex;

// sm_60 only; EXL3_HGEMM_HFMA2=0 keeps cuBLAS
static bool enabled(int device)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_init)
    {
        for (int i = 0; i < MAX_DEVICES; ++i) g_enabled[i] = -1;
        g_init = true;
    }
    if (device < 0 || device >= MAX_DEVICES) return false;
    if (g_enabled[device] >= 0) return g_enabled[device] == 1;
    const auto* props = at::cuda::getDeviceProperties(device);
    const char* env = std::getenv("EXL3_HGEMM_HFMA2");
    int on = (props->major == 6 && props->minor == 0) ? 1 : 0;
    if (env && std::strcmp(env, "0") == 0) on = 0;
    g_enabled[device] = on;
    return on == 1;
}

// Shapes, strides and alignment the kernel handles
static bool covered(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c)
{
    if (!a.is_cuda() || a.device() != b.device() || a.device() != c.device()) return false;
    if (a.dtype() != at::kHalf || b.dtype() != at::kHalf) return false;
    if (c.dtype() != at::kHalf && c.dtype() != at::kFloat) return false;
    if (b.dim() != 2 && b.dim() != 3) return false;
    const bool batched = b.dim() == 3;
    if (batched && (a.dim() != 3 || c.dim() != 3 || a.size(0) != b.size(0) || a.size(0) != c.size(0))) return false;
    if (batched && (a.size(0) < 1 || a.size(0) > 65535)) return false;
    if (!batched && (a.dim() < 2 || c.dim() < 2)) return false;
    const int64_t K = a.size(-1), N = b.size(-1);
    int64_t M = 1;
    if (batched) M = a.size(1);
    else for (int d = 0; d < a.dim() - 1; ++d) M *= a.size(d);
    if (M < 1 || K < BK || K % BK || N < BN || N % BN) return false;
    if (b.size(-2) != K || c.size(-1) != N) return false;
    if (M > std::numeric_limits<int>::max() || K > std::numeric_limits<int>::max() ||
        N > std::numeric_limits<int>::max()) return false;
    if ((M + BM - 1) / BM > 65535) return false;
    // Packed A rows (all leading dims flattened), packed B rows, contiguous C columns
    if (!batched && !a.is_contiguous()) return false;
    if (batched && (a.stride(2) != 1 || a.stride(1) != K)) return false;
    if (b.stride(-1) != 1 || b.stride(-2) != N || c.stride(-1) != 1) return false;
    int64_t c_rows = 1;
    if (batched) c_rows = c.size(1);
    else for (int d = 0; d < c.dim() - 1; ++d) c_rows *= c.size(d);
    if (c_rows != M) return false;
    // Rows of C: one stride (flattened leading dims must be uniform)
    const int64_t ldc = c.stride(-2);
    if (!batched)
    {
        int64_t expect = ldc;
        for (int d = c.dim() - 2; d >= 0; --d)
        {
            if (c.size(d) > 1 && c.stride(d) != expect) return false;
            expect *= c.size(d);
        }
    }
    if (ldc < N || ldc > std::numeric_limits<int>::max()) return false;
    const int vec = c.dtype() == at::kFloat ? 4 : 4;   // float4 / 4-half stores
    if (ldc % vec) return false;
    const uintptr_t c_align = c.dtype() == at::kFloat ? 16 : 8;
    if (((uintptr_t) a.data_ptr() & 15) || ((uintptr_t) b.data_ptr() & 15) || ((uintptr_t) c.data_ptr() & (c_align - 1)))
        return false;
    if (batched && (a.stride(0) % 8 || b.stride(0) % 8 || c.stride(0) % 4)) return false;
    return true;
}

// One 256-thread block per SM (register bound), so the grid runs in ceil(blocks / SMs) waves. Against cuBLAS
// the kernel is ~1.5x faster per full wave on the P100 (2048-row prefill shapes); a partly filled last wave
// eats into that, and grids below one wave lose outright (160 x 2048 x 512: 0.3x, 2048 x 2048 x 512 with 64
// blocks on 56 SMs: 1.0x). Take the kernel when the wave-quantised estimate still wins by 10%
static bool worthwhile(const at::Tensor& a, const at::Tensor& b)
{
    const bool batched = b.dim() == 3;
    int64_t M = 1;
    if (batched) M = a.size(1);
    else for (int d = 0; d < a.dim() - 1; ++d) M *= a.size(d);
    const int64_t blocks = ((M + BM - 1) / BM) * (b.size(-1) / BN) * (batched ? a.size(0) : 1);
    const int sms = at::cuda::getDeviceProperties(a.device().index())->multiProcessorCount;
    const double waves = (double) blocks / sms;
    const double est = std::ceil(waves) / waves / 1.5;
    return est < 0.9;
}

static void launch(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c, cudaStream_t stream)
{
    const bool batched = b.dim() == 3;
    const int batch = batched ? a.size(0) : 1;
    const int K = a.size(-1), N = b.size(-1);
    int64_t M = 1;
    if (batched) M = a.size(1);
    else for (int d = 0; d < a.dim() - 1; ++d) M *= a.size(d);
    dim3 grid(N / BN, (M + BM - 1) / BM, batch);
    const int ldc = (int) c.stride(-2);
    const int64_t as = batched ? a.stride(0) : 0, bs = batched ? b.stride(0) : 0, cs = batched ? c.stride(0) : 0;
    if (c.dtype() == at::kFloat)
        gemm_kernel<true><<<grid, THREADS, 0, stream>>>((const half*) a.data_ptr(), (const half*) b.data_ptr(),
            c.data_ptr(), (int) M, N, K, ldc, as, bs, cs);
    else
        gemm_kernel<false><<<grid, THREADS, 0, stream>>>((const half*) a.data_ptr(), (const half*) b.data_ptr(),
            c.data_ptr(), (int) M, N, K, ldc, as, bs, cs);
    cuda_check(cudaPeekAtLastError());
}

} // namespace hfma2

// Pascal HFMA2 kernel for the reconstruct-path GEMMs; false = not sm_60, disabled or shape not covered
bool hgemm_hfma2_try(const at::Tensor& a, const at::Tensor& b, at::Tensor& c)
{
    if (!a.is_cuda() || !hfma2::enabled(a.device().index())) return false;
    if (!hfma2::covered(a, b, c) || !hfma2::worthwhile(a, b)) return false;
    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    hfma2::launch(a, b, c, stream);
    return true;
}

// Force the kernel (tests / benchmarks): errors on other devices or uncovered shapes
void hgemm_hfma2(at::Tensor a, at::Tensor b, at::Tensor c)
{
    TORCH_CHECK(a.is_cuda(), "hgemm_hfma2: CUDA tensors required");
    const auto* props = at::cuda::getDeviceProperties(a.device().index());
    TORCH_CHECK(props->major == 6 && props->minor == 0, "hgemm_hfma2: sm_60 only");
    TORCH_CHECK(hfma2::covered(a, b, c), "hgemm_hfma2: unsupported shape, strides or alignment (K % 32, N % 128, packed rows)");
    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    hfma2::launch(a, b, c, stream);
}

// Try the fp16-accumulator kernel; false = caller should use cuBLAS
bool hgemm_f16acc_try(const at::Tensor& a, const at::Tensor& b, at::Tensor& c)
{
    if (hgemm_hfma2_try(a, b, c)) return true;
    if (!f16acc::covered(a, b, c) || !f16acc::worthwhile(a, b)) return false;
    if (!f16acc::enabled(a.device().index())) return false;
    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (c.dtype() == at::kFloat) f16acc::launch<true>(a, b, c, stream);
    else f16acc::launch<false>(a, b, c, stream);
    return true;
}

// Force the kernel (tests / benchmarks): errors if the shape is not covered
void hgemm_f16acc(at::Tensor a, at::Tensor b, at::Tensor c)
{
    // Empty batch, rows or columns: nothing to compute. Empty reduction (K = 0): the product is zero
    if (a.numel() == 0 || b.numel() == 0 || c.numel() == 0)
    {
        TORCH_CHECK(f16acc::covered_shapes(a, b, c), "hgemm_f16acc: unsupported device, dtype or shape (K % 64, N % 128)");
        if (c.numel()) c.zero_();
        return;
    }
    TORCH_CHECK(f16acc::covered(a, b, c), "hgemm_f16acc: unsupported device, shape, strides or alignment (Ampere+, K % 64, N % 128; 16-byte input and vector-aligned output)");
    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    if (c.dtype() == at::kFloat) f16acc::launch<true>(a, b, c, stream);
    else f16acc::launch<false>(a, b, c, stream);
}

// 1 = fp16-accumulate kernel active on this device, 0 = cuBLAS (probe result or env override)
int hgemm_f16acc_status(int device)
{
    return f16acc::enabled(device) ? 1 : 0;
}

// Reconstruct-path GEMM: the fp16-accumulator kernel where it pays, else cuBLAS
void hgemm_recon(at::Tensor a, at::Tensor b, at::Tensor c)
{
    if (hgemm_f16acc_try(a, b, c)) return;
    hgemm(a, b, c);
}
