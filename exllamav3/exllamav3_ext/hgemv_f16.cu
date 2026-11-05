#include <cuda_fp16.h>
#include "hgemv_f16.cuh"
#include "routing_topk.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <limits>

/*
Dense fp16 GEMV for decode-class row counts, y[m, :] = x[m, :] @ w, w row-major [K, N] (the LinearFP16 layout),
m = 1..HGEMV_MAX_M, fp16 or fp32 output.

cuBLAS runs these shapes at 130-250 GB/s on the P100 (gemv2N / sgemm at m = 2), a fraction of the HBM2 bandwidth.
Here each lane owns 8 consecutive output columns and reads them with one 16-byte load per weight row, a warp
covers 256 columns (512 contiguous bytes per row) and the 8 warps of a block take interleaved rows of the block's
K range, 4 rows in flight per lane. Narrow outputs split K over blocks so the grid fills the GPU; the partial sums
go to a per-(device, stream) fp32 scratch and the last block of each column tile (completion counter, reset by
that block) adds them in a fixed order, so results are deterministic and the launch is graph-capturable.

Accumulation is fp32 (each weight converted once and reused for all m rows): the same contract as cuBLAS's
fp32-compute gemv, only the summation order differs.
*/

namespace hgemv
{

constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;
constexpr int COLS = 256;           // output columns per block (32 lanes x 8)
constexpr int UNROLL = 4;           // weight rows in flight per lane
constexpr int XSTAGE = 512;         // x rows staged in shared memory at a time
constexpr int MAX_SCRATCH = 1 << 20;   // fp32 partials per (device, stream)
constexpr int MAX_TILES = 1 << 16;     // completion counters per (device, stream)

__device__ __forceinline__ uint4 ldg_cs(const uint4* p)
{
    // Weights are streamed once: L2-only, evict-first
    uint4 r;
    asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}

__device__ __forceinline__ void fma8(float* acc, float xv, const uint4& w)
{
    const half2* h = (const half2*) &w;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        float2 f = __half22float2(h[i]);
        acc[2 * i + 0] = fmaf(xv, f.x, acc[2 * i + 0]);
        acc[2 * i + 1] = fmaf(xv, f.y, acc[2 * i + 1]);
    }
}

// No epilogue
struct NoEpi
{
    static constexpr bool active = false;
    __device__ __forceinline__ void operator()(const half*, int, int) const {}
};

template <int M, bool OUT_F32, class Epi = NoEpi>
__global__ __launch_bounds__(THREADS)
void hgemv_kernel
(
    const half* __restrict__ x,     // [M, K], row stride ldx
    const half* __restrict__ w,     // [K, N]
    void* __restrict__ y,           // [M, N], row stride ldy
    float* __restrict__ scratch,    // [ksplit, M, N] when gridDim.y > 1
    int* __restrict__ counters,     // [gridDim.x] when gridDim.y > 1, zero on entry, zero on exit
    const int K,
    const int N,
    const int ldx,
    const int ldy,
    const int kchunk,
    const Epi epi = Epi()           // runs on the final fp16 output in the block that wrote it (one column tile)
)
{
    __shared__ float xs[M][XSTAGE];
    __shared__ float red[WARPS][COLS];
    __shared__ int is_last;

    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;
    const int c0 = blockIdx.x * COLS;
    const int col = c0 + lane * 8;
    const bool col_ok = col < N;      // N % 8 == 0, so a lane's 8 columns are all in or all out
    const int kb = blockIdx.y * kchunk;
    const int ke = min(K, kb + kchunk);

    float acc[M][8];
    #pragma unroll
    for (int m = 0; m < M; ++m)
        #pragma unroll
        for (int i = 0; i < 8; ++i) acc[m][i] = 0.0f;

    const size_t row_stride = (size_t) N / 8;    // in uint4
    const uint4* wp = ((const uint4*) w) + (col_ok ? col / 8 : 0);

    for (int ks = kb; ks < ke; ks += XSTAGE)
    {
        const int kn = min(XSTAGE, ke - ks);
        __syncthreads();
        for (int i = t; i < XSTAGE; i += THREADS)
        {
            #pragma unroll
            for (int m = 0; m < M; ++m)
                xs[m][i] = i < kn ? __half2float(x[(size_t) m * ldx + ks + i]) : 0.0f;
        }
        __syncthreads();

        if (col_ok)
        {
            for (int r = warp; r < kn; r += WARPS * UNROLL)
            {
                uint4 wv[UNROLL];
                #pragma unroll
                for (int u = 0; u < UNROLL; ++u)
                {
                    const int rr = r + u * WARPS;
                    if (rr < kn) wv[u] = ldg_cs(wp + (size_t) (ks + rr) * row_stride);
                    else wv[u] = make_uint4(0, 0, 0, 0);
                }
                #pragma unroll
                for (int u = 0; u < UNROLL; ++u)
                {
                    const int rr = min(r + u * WARPS, XSTAGE - 1);   // zero weights past kn, xs is in bounds
                    #pragma unroll
                    for (int m = 0; m < M; ++m)
                        fma8(acc[m], xs[m][rr], wv[u]);
                }
            }
        }
    }

    // Block reduction over the warps, one row of the output at a time
    const bool split = gridDim.y > 1;
    #pragma unroll
    for (int m = 0; m < M; ++m)
    {
        __syncthreads();
        float4* rp = (float4*) &red[warp][lane * 8];
        rp[0] = make_float4(acc[m][0], acc[m][1], acc[m][2], acc[m][3]);
        rp[1] = make_float4(acc[m][4], acc[m][5], acc[m][6], acc[m][7]);
        __syncthreads();
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < WARPS; ++i) s += red[i][t];
        const int n = c0 + t;
        if (n < N)
        {
            if (split)
                scratch[((size_t) blockIdx.y * M + m) * N + n] = s;
            else if (OUT_F32)
                ((float*) y)[(size_t) m * ldy + n] = s;
            else
                ((half*) y)[(size_t) m * ldy + n] = __float2half_rn(s);
        }
    }
    if (!split)
    {
        if constexpr (Epi::active && !OUT_F32)
        {
            __syncthreads();
            epi((const half*) y, ldy, M);
        }
        return;
    }

    // Last block of this column tile adds the partials of all K slices, in slice order
    __threadfence();
    __syncthreads();
    if (t == 0)
    {
        const int prev = atomicAdd(&counters[blockIdx.x], 1);
        is_last = prev == (int) gridDim.y - 1;
    }
    __syncthreads();
    if (!is_last) return;
    __threadfence();
    const int n = c0 + t;
    if (n < N)
    {
        #pragma unroll
        for (int m = 0; m < M; ++m)
        {
            float s = 0.0f;
            for (int k = 0; k < (int) gridDim.y; ++k)
                s += __ldcg(scratch + ((size_t) k * M + m) * N + n);
            if (OUT_F32)
                ((float*) y)[(size_t) m * ldy + n] = s;
            else
                ((half*) y)[(size_t) m * ldy + n] = __float2half_rn(s);
        }
    }
    if (t == 0) counters[blockIdx.x] = 0;
    if constexpr (Epi::active && !OUT_F32)
    {
        __syncthreads();
        epi((const half*) y, ldy, M);
    }
}

struct Buffers
{
    float* scratch = nullptr;
    int* counters = nullptr;
};

static std::mutex g_mutex;
static std::map<std::pair<int, cudaStream_t>, Buffers> g_buffers;
static int g_enabled[64];
static bool g_init = false;

// Default on for sm_60 (P100) and sm_75 (Turing); EXL3_HGEMV=0 keeps cuBLAS, =1 forces it on any CUDA device
static bool enabled(int device)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_init)
    {
        for (int i = 0; i < 64; ++i) g_enabled[i] = -1;
        g_init = true;
    }
    if (device < 0 || device >= 64) return false;
    if (g_enabled[device] >= 0) return g_enabled[device] == 1;
    const auto* props = at::cuda::getDeviceProperties(device);
    int on = ((props->major == 6 && props->minor == 0) || (props->major == 7 && props->minor == 5)) ? 1 : 0;
    const char* env = std::getenv("EXL3_HGEMV");
    if (env && std::strcmp(env, "0") == 0) on = 0;
    if (env && std::strcmp(env, "1") == 0) on = 1;
    g_enabled[device] = on;
    return on == 1;
}

// Scratch and counters for one (device, stream); null if they would have to be allocated during a capture.
// Entries (about 4.25 MiB each) live for the process: torch hands out streams from a fixed pool (32 per device) and
// the generator uses the current stream, so the map stays small. Code that creates fresh streams per request
// would grow it, and then needs a release hook (not needed here)
static const Buffers* get_buffers(int device, cudaStream_t stream)
{
    std::lock_guard<std::mutex> lock(g_mutex);
    auto key = std::make_pair(device, stream);
    auto it = g_buffers.find(key);
    if (it != g_buffers.end()) return &it->second;
    cudaStreamCaptureStatus status;
    cuda_check(cudaStreamIsCapturing(stream, &status));
    if (status != cudaStreamCaptureStatusNone) return nullptr;
    Buffers b;
    cuda_check(cudaMalloc(&b.scratch, sizeof(float) * MAX_SCRATCH));
    cuda_check(cudaMalloc(&b.counters, sizeof(int) * MAX_TILES));
    cuda_check(cudaMemsetAsync(b.counters, 0, sizeof(int) * MAX_TILES, stream));
    return &(g_buffers[key] = b);
}

static int num_sms(int device)
{
    static int sms[64] = {};
    if (!sms[device]) sms[device] = at::cuda::getDeviceProperties(device)->multiProcessorCount;
    return sms[device];
}

// K slices per column tile: about 4 blocks per SM, at least 64 rows per slice
static void pick_split(int device, int M, int K, int N, int& ksplit, int& kchunk)
{
    const int tiles = (N + COLS - 1) / COLS;
    const int target = num_sms(device) * 4;
    ksplit = (target + tiles - 1) / tiles;
    ksplit = std::min(ksplit, std::max(1, K / 64));
    ksplit = std::min(ksplit, std::max(1, MAX_SCRATCH / (M * N)));
    kchunk = (K + ksplit - 1) / ksplit;
    kchunk = (kchunk + WARPS * UNROLL - 1) / (WARPS * UNROLL) * (WARPS * UNROLL);
    ksplit = (K + kchunk - 1) / kchunk;
}

template <int M>
static void launch_m(const half* x, const half* w, void* y, bool f32, const Buffers* buf,
                     int K, int N, int ldx, int ldy, int ksplit, int kchunk, cudaStream_t stream)
{
    dim3 grid((N + COLS - 1) / COLS, ksplit);
    float* scratch = ksplit > 1 ? buf->scratch : nullptr;
    int* counters = ksplit > 1 ? buf->counters : nullptr;
    if (f32) hgemv_kernel<M, true><<<grid, THREADS, 0, stream>>>(x, w, y, scratch, counters, K, N, ldx, ldy, kchunk);
    else hgemv_kernel<M, false><<<grid, THREADS, 0, stream>>>(x, w, y, scratch, counters, K, N, ldx, ldy, kchunk);
}

} // namespace hgemv

// Shapes, strides and alignment the kernel handles: a [.., K] fp16 with packed rows (or one row), b [K, N] fp16
// packed, c [.., N] fp16/fp32 with one row stride
static bool hgemv_covered(const at::Tensor& a, const at::Tensor& b, const at::Tensor& c, int64_t& M)
{
    if (!a.is_cuda() || a.device() != b.device() || a.device() != c.device()) return false;
    if (a.dtype() != at::kHalf || b.dtype() != at::kHalf) return false;
    if (c.dtype() != at::kHalf && c.dtype() != at::kFloat) return false;
    if (b.dim() != 2 || a.dim() < 1 || c.dim() < 1) return false;
    const int64_t K = a.size(-1), N = b.size(-1);
    M = 1;
    for (int d = 0; d < a.dim() - 1; ++d) M *= a.size(d);
    if (M < 1 || M > HGEMV_MAX_M) return false;
    if (b.size(0) != K || c.size(-1) != N) return false;
    if (K < 1 || N < HGEMV_MIN_N || N % 8) return false;
    if (K > std::numeric_limits<int>::max() || N > std::numeric_limits<int>::max() / 2) return false;
    if (b.stride(1) != 1 || b.stride(0) != N) return false;
    if (((uintptr_t) b.data_ptr()) & 15) return false;
    if (a.stride(-1) != 1 || c.stride(-1) != 1) return false;
    int64_t c_rows = 1;
    for (int d = 0; d < c.dim() - 1; ++d) c_rows *= c.size(d);
    if (c_rows != M) return false;
    // One row stride for the flattened leading dims of a and c
    for (const at::Tensor* t : { &a, &c })
    {
        if (t->dim() < 2) continue;
        int64_t expect = t->stride(-2);
        for (int d = t->dim() - 2; d >= 0; --d)
        {
            if (t->size(d) > 1 && t->stride(d) != expect) return false;
            expect *= t->size(d);
        }
    }
    const int64_t lda = a.dim() >= 2 ? a.stride(-2) : K;
    const int64_t ldc = c.dim() >= 2 ? c.stride(-2) : N;
    if (M > 1 && (lda < K || ldc < N)) return false;
    if (lda > std::numeric_limits<int>::max() || ldc > std::numeric_limits<int>::max()) return false;
    return true;
}

static void hgemv_launch(const at::Tensor& a, const at::Tensor& b, at::Tensor& c, int M,
                         const hgemv::Buffers* buf, cudaStream_t stream, int ksplit, int kchunk)
{
    const int K = a.size(-1), N = b.size(-1);
    const int lda = a.dim() >= 2 ? (int) a.stride(-2) : K;
    const int ldc = c.dim() >= 2 ? (int) c.stride(-2) : N;
    const half* x = (const half*) a.data_ptr();
    const half* w = (const half*) b.data_ptr();
    const bool f32 = c.dtype() == at::kFloat;
    switch (M)
    {
        case 1: hgemv::launch_m<1>(x, w, c.data_ptr(), f32, buf, K, N, lda, ldc, ksplit, kchunk, stream); break;
        case 2: hgemv::launch_m<2>(x, w, c.data_ptr(), f32, buf, K, N, lda, ldc, ksplit, kchunk, stream); break;
        case 3: hgemv::launch_m<3>(x, w, c.data_ptr(), f32, buf, K, N, lda, ldc, ksplit, kchunk, stream); break;
        case 4: hgemv::launch_m<4>(x, w, c.data_ptr(), f32, buf, K, N, lda, ldc, ksplit, kchunk, stream); break;
    }
    cuda_check(cudaPeekAtLastError());
}

bool hgemv_f16_try(const at::Tensor& a, const at::Tensor& b, at::Tensor& c, cudaStream_t stream)
{
    int64_t M;
    if (!a.is_cuda() || !hgemv::enabled(a.device().index())) return false;
    if (!hgemv_covered(a, b, c, M)) return false;
    const int device = a.device().index();
    int ksplit, kchunk;
    hgemv::pick_split(device, (int) M, (int) a.size(-1), (int) b.size(-1), ksplit, kchunk);
    const hgemv::Buffers* buf = nullptr;
    if (ksplit > 1)
    {
        buf = hgemv::get_buffers(device, stream);
        if (!buf) return false;
    }
    hgemv_launch(a, b, c, (int) M, buf, stream, ksplit, kchunk);
    return true;
}

// Force the kernel (tests / benchmarks), any CUDA device. ksplit <= 0: automatic
void hgemv_f16(at::Tensor a, at::Tensor b, at::Tensor c, int64_t ksplit)
{
    int64_t M;
    TORCH_CHECK(hgemv_covered(a, b, c, M), "hgemv_f16: unsupported shape, strides or alignment");
    const at::cuda::OptionalCUDAGuard device_guard(a.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int device = a.device().index();
    const int K = a.size(-1);
    int ks, kchunk;
    hgemv::pick_split(device, (int) M, K, (int) b.size(-1), ks, kchunk);
    if (ksplit > 0)
    {
        kchunk = (K + (int) ksplit - 1) / (int) ksplit;
        kchunk = (kchunk + hgemv::WARPS * hgemv::UNROLL - 1) / (hgemv::WARPS * hgemv::UNROLL) * (hgemv::WARPS * hgemv::UNROLL);
        ks = (K + kchunk - 1) / kchunk;
        TORCH_CHECK((int64_t) ks * M * b.size(-1) <= hgemv::MAX_SCRATCH, "hgemv_f16: ksplit too large for the scratch");
    }
    const hgemv::Buffers* buf = ks > 1 ? hgemv::get_buffers(device, stream) : nullptr;
    TORCH_CHECK(ks <= 1 || buf, "hgemv_f16: scratch cannot be allocated during a graph capture");
    hgemv_launch(a, b, c, (int) M, buf, stream, ks, kchunk);
}


// Router projection + softmax top-k in one launch (EXL3_ROUTER_FUSED=1, sm_60 / sm_75 where hgemv is on): the
// same hgemv kernel and split as the hgemm path, the top-k of routing_std_topk_kernel runs in the block that
// holds the final logits, so scores, indices and weights are bit-identical to the two launches
bool hgemv_router_topk_try(const at::Tensor& a, const at::Tensor& b, at::Tensor& c, at::Tensor& topk_indices,
                           at::Tensor& topk_weights, const void* per_expert_scale, const void* bias, cudaStream_t stream)
{
    // Read per call (cheap next to the launch) so tests can compare both paths in one process
    const char* env = std::getenv("EXL3_ROUTER_FUSED");
    if (!env || std::strcmp(env, "1") != 0) return false;
    int64_t M;
    if (!a.is_cuda() || !hgemv::enabled(a.device().index())) return false;
    if (!hgemv_covered(a, b, c, M)) return false;
    if (c.dtype() != at::kHalf || !c.is_contiguous()) return false;
    const int K = (int) a.size(-1), N = (int) b.size(-1);
    if (N > hgemv::COLS || N > hgemv::THREADS) return false;
    if (!topk_indices.is_contiguous() || !topk_weights.is_contiguous()) return false;
    const int device = a.device().index();
    int ksplit, kchunk;
    hgemv::pick_split(device, (int) M, K, N, ksplit, kchunk);
    const hgemv::Buffers* buf = nullptr;
    if (ksplit > 1)
    {
        buf = hgemv::get_buffers(device, stream);
        if (!buf) return false;
    }
    const int lda = a.dim() >= 2 ? (int) a.stride(-2) : K;
    const int ldc = N;
    RouterTopkEpi epi { (int64_t*) topk_indices.data_ptr(), (half*) topk_weights.data_ptr(),
                        (const bfloat16*) per_expert_scale, (const half*) bias, N, (int) topk_indices.size(-1) };
    dim3 grid(1, ksplit);
    float* scratch = ksplit > 1 ? buf->scratch : nullptr;
    int* counters = ksplit > 1 ? buf->counters : nullptr;
    const half* x = (const half*) a.data_ptr();
    const half* w = (const half*) b.data_ptr();
    #define RT_LAUNCH(MM) hgemv::hgemv_kernel<MM, false, RouterTopkEpi><<<grid, hgemv::THREADS, 0, stream>>> \
        (x, w, c.data_ptr(), scratch, counters, K, N, lda, ldc, kchunk, epi)
    switch (M)
    {
        case 1: RT_LAUNCH(1); break;
        case 2: RT_LAUNCH(2); break;
        case 3: RT_LAUNCH(3); break;
        case 4: RT_LAUNCH(4); break;
    }
    #undef RT_LAUNCH
    cuda_check(cudaPeekAtLastError());
    return true;
}
