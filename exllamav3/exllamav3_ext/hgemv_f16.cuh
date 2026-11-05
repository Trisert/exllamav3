#pragma once

#include <ATen/Tensor.h>
#include <cuda_runtime.h>

// Dense fp16 GEMV for decode-class row counts (hgemv_f16.cu): y = x @ w with w row-major [K, N] fp16, x [.., K]
// fp16 with 1..HGEMV_MAX_M rows, y fp16 or fp32, fp32 accumulation, deterministic split-K. hgemv_f16_try runs it
// where it is enabled (sm_60 and sm_75 by default, EXL3_HGEMV=0/1 overrides) and the shape is covered
// (N % 8 == 0, N >= HGEMV_MIN_N, packed 16-byte aligned w); false = caller uses cuBLAS
#define HGEMV_MAX_M 4
#define HGEMV_MIN_N 8

bool hgemv_f16_try(const at::Tensor& a, const at::Tensor& b, at::Tensor& c, cudaStream_t stream);
void hgemv_f16(at::Tensor a, at::Tensor b, at::Tensor c, int64_t ksplit);

// Router logits + softmax top-k in one launch (routing_std; EXL3_ROUTER_FUSED=1); false = caller runs the two launches
bool hgemv_router_topk_try(const at::Tensor& a, const at::Tensor& b, at::Tensor& c, at::Tensor& topk_indices,
                           at::Tensor& topk_weights, const void* per_expert_scale, const void* bias, cudaStream_t stream);
