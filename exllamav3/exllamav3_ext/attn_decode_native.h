#pragma once

#include <memory>
#include "triton_kernel.h"

// CUDA-native flash-decoding split kernel for BC_Attention (attn_decode_native.cu): q_len 1, 8-bit cache, head_dim
// 128, GQA group 1 / 2 / 4 / 8; launched through the Triton split kernel's sites, combined by its combine kernel
std::shared_ptr<TritonKernel> make_decode_split_native
(
    int group,
    int head_dim,
    int bits,
    float scale,
    int n_kv_heads,
    int page_size,
    bool live_split,
    int block_rows
);
