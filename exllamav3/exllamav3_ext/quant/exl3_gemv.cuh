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
    bool force,
    // gated: A is [gate row | up row] and the kernel feeds silu(gate) * up (m = 1, mul1, 7 / 8 bpw); dry: decide
    // without launching (true = this call would launch)
    bool gated = false,
    bool dry = false
);

// Gated-input GEMV instances (comp_units/exl3_gemv_gated_inst.cu)
void* exl3_gemv_select_kernel_gated(int bits, bool c_fp32, int cfg, bool bal);

// Kernel instances for the half-integer bitrates (comp_units/exl3_gemv_half_inst.cu)
void* exl3_gemv_select_kernel_half(int bits, bool c_fp32, int mmode, int cfg, bool smem);

// Balanced-split (m = 1) kernel instances, mul1 2..8 bpw (comp_units/exl3_gemv_bal_inst.cu)
void* exl3_gemv_select_kernel_bal(int bits, bool c_fp32, int cfg);

// Multi-matrix small-m GEMV kernel instances, mul1 7 / 8 bpw (comp_units/exl3_mgemv_inst*.cu)
void* exl3_mgemv_select_kernel(int bits, bool c_fp32, int mmode, int cfg, bool bal);

// Try to dispatch an exl3_mgemm call (uniform or sliced, single input, no indices / weights; the caller checks those)
// with m <= 16 to the multi-matrix GEMV. mgemm_args are exl3_mgemm's kernel arguments, num_slices its output slot count
bool exl3_mgemv_try_launch
(
    void** mgemm_args,
    int size_m,
    int size_k,
    int size_n,
    int num_slices,
    int K,
    bool half_k,
    int cb,
    bool c_fp32,
    int device,
    cudaStream_t stream,
    void** launched_kernel
);

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
