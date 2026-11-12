#pragma once

#include <ATen/Tensor.h>

// Fused LFM2-style short-conv core for decode / short verify windows:
//   bc [bsz * seqlen, 3 * dim] fp16 (b | c | x), conv state [slots, dim, S] fp16, slots [bsz] int32,
//   weight [dim, ksz] (bf16 or fp16), optional bias [dim] (same dtype), out [bsz * seqlen, dim] fp16.
// Replaces chunk / b * x / cat / state copy / dtype cast / depthwise conv1d / cast / c * conv_out
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
);
