#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "short_conv.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "util.h"
#include "util.cuh"

/*
One thread per (batch item, channel). Mirrors ShortConv._causal_conv1d + the surrounding glue of the PyTorch path:

    conv_input = b * x                                  fp16 product
    y          = cat(state[:ksz], conv_input)           ksz + L columns
    state     <- last min(S, ksz + L) columns (history) | last ksz columns into state[:ksz]
    out        = c * fp16(Wt(conv1d(Wt(y), weight, bias)))   Wt = weight dtype (bf16 / fp16), fp32 accumulation

The thread owns its channel row of the state, so reads and the final state write cannot race with other threads
(distinct slots per batch item).
*/

#define SC_MAX_KS 8
#define SC_MAX_L 16

__device__ __forceinline__ float sc_round(float v, __nv_bfloat16*) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float sc_round(float v, half*) { return __half2float(__float2half_rn(v)); }
__device__ __forceinline__ float sc_to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float sc_to_float(half v) { return __half2float(v); }

template <typename W, int LT>
__global__ __launch_bounds__(128)
void short_conv_decode_kernel
(
    const half* __restrict__ bc,
    half* __restrict__ state,
    const int* __restrict__ slots,
    const W* __restrict__ weight,
    const W* __restrict__ bias,
    half* __restrict__ out,
    int dim,
    int S,
    int ksz,
    int seqlen,
    bool history
)
{
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    const int bi = blockIdx.y;
    const int L = LT > 0 ? LT : seqlen;

    half* st = state + ((size_t) slots[bi] * dim + d) * S;

    half hv[SC_MAX_KS + (LT > 0 ? LT : SC_MAX_L)];
    #pragma unroll
    for (int j = 0; j < SC_MAX_KS; ++j)
        if (j < ksz) hv[j] = st[j];

    float wv[SC_MAX_KS];
    #pragma unroll
    for (int k = 0; k < SC_MAX_KS; ++k)
        wv[k] = k < ksz ? sc_to_float(weight[(size_t) d * ksz + k]) : 0.0f;
    const float bv = bias ? sc_to_float(bias[d]) : 0.0f;

    #pragma unroll
    for (int t = 0; t < (LT > 0 ? LT : SC_MAX_L); ++t)
    {
        if (t < L)
        {
            const half* row = bc + (size_t) (bi * L + t) * 3 * dim;
            hv[ksz + t] = __hmul(row[d], row[2 * dim + d]);
        }
    }

    #pragma unroll
    for (int t = 0; t < (LT > 0 ? LT : SC_MAX_L); ++t)
    {
        if (t < L)
        {
            // F.conv1d over ksz + L columns yields L + 1 outputs and the torch path keeps the last L: output t
            // reads columns t + 1 .. t + ksz. ATen's depthwise kernel starts the fp32 accumulator at the bias
            float acc = bv;
            #pragma unroll
            for (int k = 0; k < SC_MAX_KS; ++k)
                if (k < ksz)
                    acc = fmaf(wv[k], sc_round(__half2float(hv[t + 1 + k]), (W*) nullptr), acc);
            half co = __float2half_rn(sc_round(acc, (W*) nullptr));
            const half* row = bc + (size_t) (bi * L + t) * 3 * dim;
            out[(size_t) (bi * L + t) * dim + d] = __hmul(row[dim + d], co);
        }
    }

    if (history)
    {
        const int total = ksz + L;
        const int ws = S < total ? S : total;
        for (int j = 0; j < ws; ++j) st[S - ws + j] = hv[total - ws + j];
    }
    else
    {
        #pragma unroll
        for (int j = 0; j < SC_MAX_KS; ++j)
            if (j < ksz) st[j] = hv[L + j];
    }
}

void short_conv_decode
(
    const at::Tensor& bc,
    at::Tensor& state,
    const at::Tensor& slots,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::Tensor& out,
    int bsz,
    int seqlen,
    bool history
)
{
    const at::cuda::OptionalCUDAGuard device_guard(bc.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(bc, kHalf);
    TORCH_CHECK_DTYPE(state, kHalf);
    TORCH_CHECK_DTYPE(out, kHalf);
    TORCH_CHECK_DTYPE(slots, kInt);
    TORCH_CHECK_DIM(state, 3);
    TORCH_CHECK_DIM(weight, 2);
    TORCH_CHECK(bc.is_contiguous() && state.is_contiguous() && weight.is_contiguous() && out.is_contiguous(),
                "short_conv_decode: tensors must be contiguous");

    const int dim = state.size(1);
    const int S = state.size(2);
    const int ksz = weight.size(1);
    TORCH_CHECK(weight.size(0) == dim, "short_conv_decode: weight must be [dim, ksz]");
    TORCH_CHECK(ksz >= 1 && ksz <= SC_MAX_KS, "short_conv_decode: unsupported kernel size");
    TORCH_CHECK(seqlen >= 1 && seqlen <= SC_MAX_L, "short_conv_decode: unsupported seqlen");
    TORCH_CHECK(S >= ksz, "short_conv_decode: state shorter than the kernel size");
    TORCH_CHECK(bc.numel() == (int64_t) bsz * seqlen * 3 * dim, "short_conv_decode: bc shape mismatch");
    TORCH_CHECK(out.numel() == (int64_t) bsz * seqlen * dim, "short_conv_decode: out shape mismatch");
    TORCH_CHECK(slots.numel() >= bsz, "short_conv_decode: one slot per batch item");
    if (bias.has_value())
    {
        TORCH_CHECK(bias.value().dtype() == weight.dtype() && bias.value().numel() == dim && bias.value().is_contiguous(),
                    "short_conv_decode: bias must match the weight dtype");
    }

    dim3 block(128);
    dim3 grid(CEIL_DIVIDE(dim, 128), bsz);

    #define LAUNCH(WT, LT) \
        short_conv_decode_kernel<WT, LT><<<grid, block, 0, stream>>> \
        ( \
            (const half*) bc.data_ptr(), (half*) state.data_ptr(), (const int*) slots.data_ptr(), \
            (const WT*) weight.data_ptr(), (const WT*) OPTPTR(bias), (half*) out.data_ptr(), \
            dim, S, ksz, seqlen, history \
        )

    if (weight.dtype() == at::kBFloat16)
    {
        if (seqlen == 1) LAUNCH(__nv_bfloat16, 1); else LAUNCH(__nv_bfloat16, 0);
    }
    else if (weight.dtype() == at::kHalf)
    {
        if (seqlen == 1) LAUNCH(half, 1); else LAUNCH(half, 0);
    }
    else TORCH_CHECK(false, "short_conv_decode: weight must be bf16 or fp16");
    #undef LAUNCH

    cuda_check(cudaPeekAtLastError());
}
