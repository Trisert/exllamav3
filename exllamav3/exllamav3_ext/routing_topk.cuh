#pragma once

// Router top-k pieces shared by routing.cu and the fused router GEMV epilogue of hgemv_f16.cu
// (EXL3_ROUTER_FUSED): the standard softmax top-k for one row, so both paths select bit for bit alike

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "util.cuh"
#include "det_gemm.cuh"
#include "reduction.cuh"

#ifndef MAX_NUM_EXPERTS
#define MAX_NUM_EXPERTS 512
#endif
#ifndef MAX_K
#define MAX_K 32
#endif

using bfloat16 = __nv_bfloat16;

__device__ __forceinline__
void warp_reduce_best_f32(float& key, float& payload, int& idx)
{
    #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 && !defined(USE_ROCM)
        // Monotonic unsigned encoding of the float key, hardware max-reduce, then fetch the
        // winner's values from the lowest tied lane
        unsigned int ku = __float_as_uint(key);
        ku = (ku & 0x80000000u) ? ~ku : (ku | 0x80000000u);
        unsigned int m = __reduce_max_sync(0xffffffffu, ku);
        int src = __ffs(__ballot_sync(0xffffffffu, ku == m)) - 1;
        key = __shfl_sync(0xffffffffu, key, src);
        payload = __shfl_sync(0xffffffffu, payload, src);
        idx = __shfl_sync(0xffffffffu, idx, src);
    #else
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            float other_key = __shfl_down_sync(0xffffffffu, key, offset);
            float other_payload = __shfl_down_sync(0xffffffffu, payload, offset);
            int other_idx = __shfl_down_sync(0xffffffffu, idx, offset);
            if (other_key > key)
            {
                key = other_key;
                payload = other_payload;
                idx = other_idx;
            }
        }

        key = __shfl_sync(0xffffffffu, key, 0);
        payload = __shfl_sync(0xffffffffu, payload, 0);
        idx = __shfl_sync(0xffffffffu, idx, 0);
    #endif
}


// Top-K selection by one warp over num_candidates (<= MAX_NUM_EXPERTS) keys in shared memory.
// Each lane keeps its strided share of the keys in registers (at most MAX_NUM_EXPERTS / 32),
// every round picks the lane-local best, the warp reduces the round's winner and the owning
// lane retires it. Lane k ends up holding the round-k winner (payload, position). Ties: lowest
// position within a lane, lowest lane across the warp. One pass of K rounds replaces the
// per-warp pre-selection plus multi-warp merge stages: each stage cost K block-wide rounds, and
// for K > 16 a stage that keeps K of every 32 candidates could not shrink the field at all
__device__ __forceinline__
void warp_topk_shared
(
    const float* sh_key,
    const float* sh_payload,
    const int num_candidates,
    const int K,
    float& out_payload,
    int& out_idx
)
{
    constexpr int MAXC = MAX_NUM_EXPERTS / 32;
    int lane_id = threadIdx.x % 32;
    float ck[MAXC];
    #pragma unroll
    for (int j = 0; j < MAXC; ++j)
    {
        int pos = lane_id + 32 * j;
        ck[j] = pos < num_candidates ? sh_key[pos] : -1.0e30f;
    }
    out_payload = 0.0f;
    out_idx = -1;
    for (int k = 0; k < K; ++k)
    {
        float best_key = -1.0e30f;
        int best_j = -1;
        #pragma unroll
        for (int j = 0; j < MAXC; ++j)
            if (ck[j] > best_key) { best_key = ck[j]; best_j = j; }
        int my_pos = best_j >= 0 ? lane_id + 32 * best_j : -1;
        float best_payload = my_pos >= 0 ? (sh_payload ? sh_payload[my_pos] : best_key) : 0.0f;
        int best_idx = my_pos;
        warp_reduce_best_f32(best_key, best_payload, best_idx);
        if (my_pos >= 0 && my_pos == best_idx)
        {
            // Retire the winner (unrolled select: no dynamic register indexing)
            #pragma unroll
            for (int j = 0; j < MAXC; ++j)
                if (j == best_j) ck[j] = -1.0e30f;
        }
        if (lane_id == k)
        {
            out_payload = best_payload;
            out_idx = best_idx;
        }
    }
}


// Softmax top-k of one row of router logits (routing_std), in two steps so the fused router epilogue can run the
// selection of several rows on separate warps. Step 1, whole block (>= num_warps * 32 threads; warps past
// CEIL_DIVIDE(num_experts, 32) idle but reach the barriers): logits (+ bias) into sh_key, which holds
// num_experts + num_warps floats, and the max logit, returned to every thread
__device__ __forceinline__
float std_topk_load_max
(
    const half* __restrict__ scores,
    const half* __restrict__ bias,
    int num_experts,
    float* sh_key
)
{
    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int num_warps = (num_experts + 31) / 32;
    float* max_red = sh_key + num_experts;

    bool mask = t < num_experts;
    float logit = mask ? __half2float(__ldcg(scores + t)) : -1.0e30f;
    // Router bias (gpt-oss): biased logits drive both the top-k selection and the softmax
    if (bias && mask)
        logit += __half2float(bias[t]);
    if (mask)
        sh_key[t] = logit;

    // Max logit for a stable softmax
    float max_logit = logit;
    max_logit = warp_reduce_max_f(max_logit);
    max_logit = __shfl_sync(0xffffffffu, max_logit, 0);
    if (num_warps > 1)
    {
        if (lane_id == 0 && warp_id < num_warps) max_red[warp_id] = max_logit;
        __syncthreads();
        max_logit = lane_id < num_warps ? max_red[lane_id] : -1.0e30f;
        max_logit = warp_reduce_max_f(max_logit);
        max_logit = __shfl_sync(0xffffffffu, max_logit, 0);
    }
    __syncthreads();
    return max_logit;
}

// Step 2, one warp: top-k selection, softmax over the selected logits, per-expert scale, store
__device__ __forceinline__
void std_topk_select
(
    const float* sh_key,
    float max_logit,
    int64_t* __restrict__ topk_indices,
    half* __restrict__ topk_weights,
    const bfloat16* __restrict__ per_expert_scale,
    int num_experts,
    int K
)
{
    int lane_id = threadIdx.x % 32;
    float e;
    int out_idx;
    warp_topk_shared(sh_key, nullptr, num_experts, K, e, out_idx);
    e = lane_id < K ? exp_det(e - max_logit) : 0.0f;
    float sum = warp_reduce_sum_first_k(e, K) + 1e-20f;
    e /= sum;

    if (lane_id < K)
    {
        if (per_expert_scale)
            e *= __bfloat162float(per_expert_scale[out_idx]);
        topk_indices[lane_id] = (int64_t) out_idx;
        topk_weights[lane_id] = __float2half_rn(e);
    }
}

__device__ __forceinline__
void std_topk_row
(
    const half* __restrict__ scores,
    int64_t* __restrict__ topk_indices,
    half* __restrict__ topk_weights,
    const bfloat16* __restrict__ per_expert_scale,
    const half* __restrict__ bias,
    int num_experts,
    int K,
    float* sh_key
)
{
    float max_logit = std_topk_load_max(scores, bias, num_experts, sh_key);
    if (threadIdx.x / 32 == 0)
        std_topk_select(sh_key, max_logit, topk_indices, topk_weights, per_expert_scale, num_experts, K);
}

// hgemv epilogue: the block that holds the final router logits (one column tile, N <= 256) runs the top-k of
// every row on them
struct RouterTopkEpi
{
    int64_t* topk_indices;
    half* topk_weights;
    const bfloat16* per_expert_scale;
    const half* bias;
    int num_experts;
    int K;

    static constexpr bool active = true;

    __device__ __forceinline__ void operator()(const half* y, int ldy, int M) const
    {
        // Logits and max of every row first (whole block), then each row's selection on its own warp. (A 256-wide
        // bitonic sort in the selection order instead of the K serial warp-argmax rounds measured slower on the
        // P100, 27 vs 18 us for the fused launch)
        __shared__ float sh_key[4][256 + 8];
        __shared__ float sh_max[4];
        for (int m = 0; m < M; ++m)
        {
            float mx = std_topk_load_max(y + (size_t) m * ldy, bias, num_experts, sh_key[m]);
            if (threadIdx.x == 0) sh_max[m] = mx;
        }
        __syncthreads();
        const int w = threadIdx.x / 32;
        if (w < M)
            std_topk_select(sh_key[w], sh_max[w], topk_indices + K * w, topk_weights + K * w,
                            per_expert_scale, num_experts, K);
    }
};
