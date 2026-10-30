#pragma once

#include <ATen/Tensor.h>
#include "graph.cuh"

void hgemm_gr
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c,
    Graph* graph
);

void hgemm
(
    at::Tensor a,
    at::Tensor b,
    at::Tensor c
);
void hgemm_batched
(
    at::Tensor a,
    at::Tensor w,
    at::Tensor c
);

// fp16-accumulator tensor-core GEMM with fp32 accumulation across K (hgemm_f16acc.cu): full
// rate on GeForce, where the fp32-accumulator MMA runs at half rate. hgemm_f16acc_try runs it
// when the device gains (rate probe, EXL3_HGEMM_F16ACC=0/1 overrides) and the shape is covered
// (K % 64, N % 128, aligned packed input rows and nonoverlapping output rows/batches).
// Shape/batch heuristics decide whether it pays; hgemm_recon adds the cuBLAS fallback.
bool hgemm_f16acc_try(const at::Tensor& a, const at::Tensor& b, at::Tensor& c);
void hgemm_f16acc(at::Tensor a, at::Tensor b, at::Tensor c);
int hgemm_f16acc_status(int device);
void hgemm_recon(at::Tensor a, at::Tensor b, at::Tensor c);

// Pascal sm_60 counterpart (hgemm_f16acc.cu): HFMA2 with 32-term fp16 partials added into fp32, tried first by
// hgemm_f16acc_try on sm_60 (EXL3_HGEMM_HFMA2=0 keeps cuBLAS). K % 32, N % 128, packed aligned rows
bool hgemm_hfma2_try(const at::Tensor& a, const at::Tensor& b, at::Tensor& c);
void hgemm_hfma2(at::Tensor a, at::Tensor b, at::Tensor c);
