#pragma once

#include <ATen/Tensor.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

// Fused decode-shaped MoE path (bsz <= MAX_BSZN): two launches per layer run every (token, expert)
// slot through gate/up, activation and down, replacing the per-projection mgemm CUDA graphs of
// BC_BlockSparseMLP. See exl3_moe_coop_kernel.cuh / exl3_moe_coop.cu.

#define MOE_COOP_ACT_SILU 0
#define MOE_COOP_ACT_GELU 1
#define MOE_COOP_ACT_RELU2 2
#define MOE_COOP_ACT_SILU_OAI 3

#ifndef MOE_COOP_THREADS
#define MOE_COOP_THREADS 256
#endif
#ifndef MOE_COOP_MINB
#define MOE_COOP_MINB 2                 // blocks per SM the register budget is sized for (65536 / (THREADS * MINB) regs per thread)
#endif
#define MOE_COOP_WNT 2                  // adjacent 16-column tiles per warp (32-column groups per block)
#define MOE_COOP_COLS (MOE_COOP_WNT * 16)

struct MoeCoopParams
{
    // Inputs
    const half* x;              // (bsz, H), row stride x_stride
    int x_stride;
    const int64_t* sel;         // (bsz, topk) global expert indices
    const half* rw;             // (bsz, topk) routing weights
    int bsz;
    int topk;
    int H;                      // x width (the experts' input width before padding)
    int Hi;                     // gate/up in_features, padded (multiple of 128)
    int I;                      // intermediate width, padded (multiple of 128)
    int Ho;                     // down out_features, padded (multiple of 128)
    int H_out;                  // output width (<= Ho)
    int min_expert;             // -1: no expert-range filtering; else sel in [min, max) is local
    int max_expert;

    // Per-expert pointer tables (int64 device arrays indexed by local expert)
    const int64_t* g_trellis;   const int64_t* g_suh;   const int64_t* g_svh;
    const int64_t* u_trellis;   const int64_t* u_suh;   const int64_t* u_svh;
    const int64_t* d_trellis;   const int64_t* d_suh;   const int64_t* d_svh;
    const int64_t* g_bias;      // nullable
    const int64_t* u_bias;
    const int64_t* d_bias;
    int act;                    // MOE_COOP_ACT_*
    float act_limit;
    bool gated;                 // false: no gate projection, activation is relu(u) * u

    // Scratch (leading dim >= bsz * topk)
    half* had_g;                // (slots, Hi) rotated gate input (bsz > 1, written by the rot kernel)
    half* had_u;                // (slots, Hi) rotated up input
    bool a_global;              // true: the GEMV reads had_g / had_u; false (bsz 1): rotated in-block
    void* gu_g;                 // (slots, I) gate GEMV output, half or float (gu_f32)
    void* gu_u;                 // (slots, I)
    bool gu_f32;
    half* act_out;              // (slots, I) activation, rotated for the down projection
    float* d_out;               // (slots, Ho) down GEMV output before the output rotation
    int* ctr_a;                 // (slots_max, I / 128) completion counters of the gate/up stage
    int* ctr_b;                 // (MAX_BSZN, Ho / 128) completion counters of the down stage
    int ctr_a_len;
    int ctr_b_len;
    int ksplit_a;               // split-k blocks per chunk per stage (launcher; partial rows at slot + q * slots)
    int ksplit_b;
    int dbg;                    // diagnostics (EXL3_MOE_COOP_DBG): 1 skip A epilogue, 2 skip B reduction
    bool m34;                   // sm_60: runs of 2..4 rows take the one-output-per-lane tile (EXL3_MOE_COOP_M34, default on)
    bool compact;               // dual kernel: size the queue by the run table instead of the no-duplicate item grid
                                // (EXL3_MOE_COOP_COMPACT, default on; see exl3_moe_coop_d_kernel)
    int* runs;                  // run table (bsz > 1): [n_runs, -, run_start[slots_max + 1], order[slots_max]]
    int slots_max;
    int rows_max;               // token rows the output scratch holds
    int n_local;                // entries in the pointer tables
    int sh_gate_n;              // shared gate weight length (checked against H per call)

    // Persistent single-launch path (exl3_moe_coop_f_kernel), when the counter scratch has room
    bool f_ok;                  // ctr holds the queue and ready counters (exl3_moe_coop_ctr_len_fused)
    int* f_queue;               // [ticket, blocks done]
    int* f_ready;               // (slots_max) activation chunks stored per slot
    int f_items_a;              // queue items of the gate/up stage (its two-launch grid)
    int f_items_b;
    int* f_sh_done;             // dual kernel: shared-expert down items finished (routed params only)
    int f_sh_wait;              // dual kernel: down items of the shared expert to wait for, else 0

    // Output
    float* out;                 // (bsz, H_out), row stride out_stride
    int out_stride;

    // Shared expert (nullable): out += gate * sh_out, gate = sigmoid(x . sh_gate_w) or 1
    const float* sh_out;        // (bsz, H), row stride H
    const half* sh_gate_w;      // (H,) or null

    // Diagnostics (EXL3_MOE_TRACE=1): per-item records of the dual kernel, 4 u64 each after a 4-word header; null normally
    unsigned long long* trace;
};

// Launch with plain device pointers (called from BC_BlockSparseMLP). K: gate/up and down bit
// widths (gate and up always share one, the converter allocates them as one group); cb: 0 default,
// 1 mcg, 2 mul1 codebook, uniform across the three projections
void exl3_moe_coop_launch(const MoeCoopParams& p, float K_gu, float K_d, int cb, int device, cudaStream_t stream);

// Static part of the parameter block, validated once from the module's tensors (BC construction);
// K_gu / K_d / cb come back through the out-params. Gate tables/scratch are ignored when
// gated == false but must still be valid tensors (pass the up ones)
MoeCoopParams exl3_moe_coop_prepare
(
    int Hi,
    const at::Tensor& g_trellis, const at::Tensor& g_suh, const at::Tensor& g_svh,
    const at::Tensor& u_trellis, const at::Tensor& u_suh, const at::Tensor& u_svh,
    const at::Tensor& d_trellis, const at::Tensor& d_suh, const at::Tensor& d_svh,
    const c10::optional<at::Tensor>& g_bias,
    const c10::optional<at::Tensor>& u_bias,
    const c10::optional<at::Tensor>& d_bias,
    float Kg, float Ku, float Kd,
    bool mcg, bool mul1,
    int act,
    float act_limit,
    bool gated,
    at::Tensor& had_g,
    at::Tensor& had_u,
    at::Tensor& gu_g,
    at::Tensor& gu_u,
    at::Tensor& act_out,
    at::Tensor& d_out,
    at::Tensor& ctr,
    at::Tensor& out,
    const c10::optional<at::Tensor>& sh_gate_w,
    float& K_gu, float& K_d, int& cb
);

// Per-call part: input rows, routing, optional shared-expert output (bsz rows, width H), launch
void exl3_moe_coop_run
(
    MoeCoopParams p, float K_gu, float K_d, int cb,
    const at::Tensor& x, const at::Tensor& sel, const at::Tensor& rw,
    const c10::optional<at::Tensor>& sh_out
);

// Per-call fields of p (input rows, routing, optional shared-expert output) without launching
void exl3_moe_coop_bind
(
    MoeCoopParams& p,
    const at::Tensor& x, const at::Tensor& sel, const at::Tensor& rw,
    const c10::optional<at::Tensor>& sh_out
);

// A layer's shared expert (ps, one slot per token, its output = p's sh_out) and routed experts (p):
// one persistent launch where the dual kernel covers both widths (EXL3_MOE_COOP_FUSED), else the
// shared launch(es) followed by the routed one(s)
void exl3_moe_coop_launch_pair
(
    const MoeCoopParams& p, float K_gu, float K_d, int cb,
    const MoeCoopParams& ps, float Ks_gu, float Ks_d, int cbs,
    int device, cudaStream_t stream
);

// Tensor front end (prepare + run; the test entry point)
void exl3_moe_coop
(
    const at::Tensor& x,
    const at::Tensor& sel,
    const at::Tensor& rw,
    int min_expert,
    int max_expert,
    int Hi,
    const at::Tensor& g_trellis, const at::Tensor& g_suh, const at::Tensor& g_svh,
    const at::Tensor& u_trellis, const at::Tensor& u_suh, const at::Tensor& u_svh,
    const at::Tensor& d_trellis, const at::Tensor& d_suh, const at::Tensor& d_svh,
    const c10::optional<at::Tensor>& g_bias,
    const c10::optional<at::Tensor>& u_bias,
    const c10::optional<at::Tensor>& d_bias,
    float Kg, float Ku, float Kd,
    bool mcg, bool mul1,
    int act,
    float act_limit,
    bool gated,
    at::Tensor& had_g,
    at::Tensor& had_u,
    at::Tensor& gu_g,
    at::Tensor& gu_u,
    at::Tensor& act_out,
    at::Tensor& d_out,
    at::Tensor& ctr,
    at::Tensor& out,
    const c10::optional<at::Tensor>& sh_out,
    const c10::optional<at::Tensor>& sh_gate_w
);

// Test entry point of exl3_moe_coop_launch_pair: gated SiLU experts without biases. tabs: the nine
// pointer tables (gate, up, down x trellis, suh, svh); scratch: had_g, had_u, gu_g, gu_u, act_out,
// d_out, ctr. The shared expert runs as expert 0 of its tables with weight 1 into sh_buf (bsz, H),
// merged into out through sigmoid(x . sh_gate_w)
void exl3_moe_coop_pair
(
    const at::Tensor& x, const at::Tensor& sel, const at::Tensor& rw, int Hi,
    const std::vector<at::Tensor>& tabs, float K, const std::vector<at::Tensor>& scratch,
    const std::vector<at::Tensor>& tabs_sh, float K_sh, const std::vector<at::Tensor>& scratch_sh,
    bool mcg, bool mul1,
    at::Tensor& out, at::Tensor& sh_buf, const at::Tensor& sh_gate_w
);

// Length of the int32 scratch for a module: completion counters + run table (slots_max =
// MAX_BSZN * top_k)
inline int64_t exl3_moe_coop_ctr_len(int64_t slots_max, int64_t bsz_max, int64_t I, int64_t Ho)
{
    return slots_max * (I / 128) + bsz_max * (Ho / 128) + 2 + (slots_max + 1) + slots_max;
}

// Same plus the persistent kernel's queue and per-slot ready counters (a scratch of the base
// length keeps the two-launch path)
inline int64_t exl3_moe_coop_ctr_len_fused(int64_t slots_max, int64_t bsz_max, int64_t I, int64_t Ho)
{
    return exl3_moe_coop_ctr_len(slots_max, bsz_max, I, Ho) + 2 + slots_max;
}

// EXL3_MOE_TRACE=1: records of the dual kernel since the last call, (n, 4) int64 [t0 ns, t1 ns, smid | kind << 8 | block << 16 | item << 32, 0]
at::Tensor exl3_moe_coop_trace_dump(int64_t device);
