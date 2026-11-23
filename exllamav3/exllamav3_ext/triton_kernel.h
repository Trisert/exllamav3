#pragma once

#include <ATen/Tensor.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

// A Triton kernel compiled ahead of time (triton.compile -> asm["cubin"]), loadable and
// launchable from C++ through the driver API. Launch args are the kernel's non-constexpr
// parameters in signature order; Triton appends two hidden trailing pointer params (global and
// profile scratch), which launch() supplies as null (scratch sizes must be 0, which holds for
// all kernels used here). Compatible with Graph::record_param: kernel nodes captured from
// cuLaunchKernel are identified by the CUfunction handle().
//
// The module is loaded into the primary context of the device that is current at construction;
// construct under the target torch device.

struct TritonKernel
{
    CUmodule mod = nullptr;
    CUfunction fn = nullptr;
    std::string name;
    int num_warps;
    int shared_bytes;
    int grid_y = 1;     // second launch-grid axis chosen at compile time (sub-tiles), read by the graph launch sites
    // Values appended after the launch site's args (before the scratch params): the constants a CUDA-native
    // stand-in takes as parameters where the Triton kernel has constexprs
    std::vector<void*> extra_args;
    // Graph node identity: the CUfunction for a Triton module, the runtime kernel symbol for a CUDA-native kernel
    // (cudaGraphKernelNodeGetParams reports that as the node's func)
    void* node_handle = nullptr;

    TritonKernel(py::bytes cubin, std::string name, int num_warps, int shared_bytes);
    // CUDA-native kernel with the same leading parameters (see attn_decode_native.cu); no module to unload
    TritonKernel(CUfunction fn, void* runtime_symbol, std::string name, int num_warps, int shared_bytes,
                 std::vector<void*> extra_args);
    ~TritonKernel();
    TritonKernel(const TritonKernel&) = delete;
    TritonKernel& operator=(const TritonKernel&) = delete;

    // args = kernel parameter values (device pointers and 32-bit ints, one value per 8-byte
    // slot), without the trailing scratch params
    void launch(int gx, int gy, int gz, std::vector<void*>& args, cudaStream_t stream) const;

    void* handle() const { return node_handle ? node_handle : (void*) fn; }
};
