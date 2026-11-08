#pragma once

#include <ATen/Tensor.h>
#include <cuda_runtime.h>

// QTIP-style small-m GEMV path (see exl3_gemv_kernel.cuh). Launched from exl3_gemm when the
// heuristic applies; also exposed directly for testing. Same kernel arguments as
// exl3_gemm_kernel, so graph recording/patching is identical.

// Try to dispatch a GEMM call to the GEMV kernel. Returns false (launching nothing) if the
// call is not eligible. On success *launched_kernel receives the kernel pointer for graph
// recording. `force` bypasses the shape heuristic but not the hard constraints.
bool exl3_gemv_try_launch
(
    void** kernel_args,
    int size_m,
    int size_k,
    int size_n,
    int K,
    bool half_k,
    int cb,
    bool c_fp32,
    bool has_su_sv,
    int device,
    cudaStream_t stream,
    void** launched_kernel,
    bool force
);

// Kernel instances for the half-integer bitrates (comp_units/exl3_gemv_half_inst.cu)
void* exl3_gemv_select_kernel_half(int bits, bool c_fp32, int mmode, int cfg, bool smem);

// Direct entry point (testing): errors if the call is not hard-eligible
void exl3_gemv
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    bool mcg,
    bool mul1
);

// Fused multi-matrix GEMV (sm_60, m = 1 / 2, 2..4 bpw mul1): one cooperative launch for several
// linears that share the input. A_had is a scratch of num_matrices * rows * k halves; each matrix's
// output goes to its own C tensor. Bit-identical to the per-matrix exl3_gemv calls.
void exl3_gemv_fused
(
    const at::Tensor& A,
    const std::vector<at::Tensor>& Bs,
    const std::vector<at::Tensor>& Cs,
    const std::vector<at::Tensor>& suhs,
    at::Tensor& A_had,
    const std::vector<at::Tensor>& svhs,
    bool mcg,
    bool mul1
);
