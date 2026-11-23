#pragma once

#include <ATen/Tensor.h>

class Graph;

void rms_norm
(
    at::Tensor x,
    const c10::optional<at::Tensor> w,
    at::Tensor y,
    float epsilon,
    float constant_bias,
    float constant_scale,
    bool span_heads,
    bool add_residual,
    int w_groups = 1
);

void rms_norm_gr
(
    at::Tensor x,
    const c10::optional<at::Tensor> w,
    at::Tensor y,
    float epsilon,
    float constant_bias,
    float constant_scale,
    Graph* graph
);

// Post-norm residual then pre-norm in one launch: r += norm(x) * w1, out = norm(r) * w2 (fp16)
bool rms_norm_post_pre_ok(const at::Tensor& x, const at::Tensor& r);
void rms_norm_post_pre
(
    at::Tensor x,
    c10::optional<at::Tensor> w1,
    at::Tensor r,
    c10::optional<at::Tensor> w2,
    at::Tensor out,
    float eps1,
    float cb1,
    float cs1,
    float eps2,
    float cb2,
    float cs2
);

void rms_norm_res_in
(
    at::Tensor x,
    c10::optional<at::Tensor> w,
    at::Tensor y,
    at::Tensor r,
    float epsilon,
    float constant_bias,
    float constant_scale
);

void gated_rms_norm
(
    at::Tensor x,
    at::Tensor w,
    at::Tensor y,
    at::Tensor g,
    float epsilon,
    float constant_bias,
    int w_groups = 1,
    bool gate_first = false,
    int gate_act = 0
);

void gated_rms_norm_gr
(
    at::Tensor x,
    at::Tensor w,
    at::Tensor y,
    at::Tensor g,
    float epsilon,
    float constant_bias,
    Graph* graph,
    int w_groups = 1,
    bool gate_first = false,
    int gate_act = 0
);

void rms_norm_gr
(
    const at::Tensor& x,
    const c10::optional<at::Tensor>& w,
    at::Tensor& y,
    float epsilon,
    float constant_bias,
    float constant_scale,
    bool span_heads,
    Graph* graph
);
