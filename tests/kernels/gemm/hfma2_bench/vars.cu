// hfma2 GEMM variants side by side for ABAB timing (doc/pascal_progress.md "hfma2 prefill GEMM"):
// v0 = the kernel before Nov 6 verbatim (base_kernel.inc), v1 = plain-half A / LDS.128 A + B, v2 = v1 with one shared
// stage and 3 blocks/SM, v3/v4 = v1/v2 with the K loop unrolled by two tiles (v4 = the extension's kernel now),
// v5-v7 = grouped raster (no gain). calib = register-only HFMA2 kernel, the clock reference of bench.py
#include <torch/extension.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

constexpr int BN = 128, BK = 32;

namespace base {
using ::BN; using ::BK;
#include "base_kernel.inc"
}

#include "v1_kernel.inc"
#include "v3_kernel.inc"

template <int BM, bool F32>
static void launch_one(int v, dim3 grid, cudaStream_t s, const half* a, const half* b, void* c, int M, int N, int K, int ldc)
{
    if (v == 0) base::gemm_kernel<F32, BM><<<grid, BM * 2, 0, s>>>(a, b, c, M, N, K, ldc, 0, 0, 0);
    else if (v == 1) gemm_kernel_v1<F32, BM><<<grid, BM * 2, 0, s>>>(a, b, c, M, N, K, ldc, 0, 0, 0);
    else if (v == 2) gemm_kernel_v1<F32, BM, 1, 3 * 64 / BM><<<grid, BM * 2, 0, s>>>(a, b, c, M, N, K, ldc, 0, 0, 0);
    else if (v == 3) gemm_kernel_v3<F32, BM><<<grid, BM * 2, 0, s>>>(a, b, c, M, N, K, ldc, 0, 0, 0);
    else if (v == 4) gemm_kernel_v3<F32, BM, 1, 3 * 64 / BM><<<grid, BM * 2, 0, s>>>(a, b, c, M, N, K, ldc, 0, 0, 0);
    else if (v == 5) gemm_kernel_v3<F32, BM, 1, 3 * 64 / BM, 8><<<grid, BM * 2, 0, s>>>(a, b, c, M, N, K, ldc, 0, 0, 0);
    else if (v == 6) gemm_kernel_v3<F32, BM, 1, 3 * 64 / BM, 16><<<grid, BM * 2, 0, s>>>(a, b, c, M, N, K, ldc, 0, 0, 0);
    else if (v == 7) gemm_kernel_v3<F32, BM, 2, 128 / BM, 8><<<grid, BM * 2, 0, s>>>(a, b, c, M, N, K, ldc, 0, 0, 0);
    else TORCH_CHECK(false, "variant");
}

void run(int64_t v, at::Tensor a, at::Tensor b, at::Tensor c, int64_t bm)
{
    const c10::cuda::OptionalCUDAGuard guard(a.device());
    const int M = a.size(0), K = a.size(1), N = b.size(1);
    TORCH_CHECK(K % BK == 0 && N % BN == 0 && a.is_contiguous() && b.is_contiguous() && c.is_contiguous());
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    const half* ap = (const half*) a.data_ptr();
    const half* bp = (const half*) b.data_ptr();
    const bool f32 = c.dtype() == at::kFloat;
    dim3 grid(N / BN, (M + bm - 1) / bm, 1);
    if (bm == 64) { if (f32) launch_one<64, true>(v, grid, s, ap, bp, c.data_ptr(), M, N, K, N); else launch_one<64, false>(v, grid, s, ap, bp, c.data_ptr(), M, N, K, N); }
    else { if (f32) launch_one<128, true>(v, grid, s, ap, bp, c.data_ptr(), M, N, K, N); else launch_one<128, false>(v, grid, s, ap, bp, c.data_ptr(), M, N, K, N); }
    C10_CUDA_CHECK(cudaPeekAtLastError());
}

// Clock calibration: register-only HFMA2 chains (issue-bound at the HFMA2 peak), 4 x 256 threads per SM
__global__ void __launch_bounds__(256) calib_kernel(int iters, half2 m, half2 a, half2* sink)
{
    half2 x[12];
    #pragma unroll
    for (int c = 0; c < 12; ++c) x[c] = __hadd2(a, __float2half2_rn((float) threadIdx.x * 1e-3f + c));
    for (int i = 0; i < iters; ++i)
    {
        #pragma unroll
        for (int u = 0; u < 8; ++u)
            #pragma unroll
            for (int c = 0; c < 12; ++c) x[c] = __hfma2(x[c], m, a);
    }
    half2 s = x[0];
    #pragma unroll
    for (int c = 1; c < 12; ++c) s = __hadd2(s, x[c]);
    if (__low2float(s) == 1234.5f) sink[threadIdx.x] = s;
}

// Returns flop of one calib launch
double calib(int64_t iters, at::Tensor sink)
{
    cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
    int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, sink.device().index());
    calib_kernel<<<sms * 4, 256, 0, s>>>((int) iters, __float2half2_rn(0.999f), __float2half2_rn(0.001f), (half2*) sink.data_ptr());
    C10_CUDA_CHECK(cudaPeekAtLastError());
    return (double) sms * 4 * 256 * iters * 8 * 12 * 4;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); m.def("calib", &calib); }
