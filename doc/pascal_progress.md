# Pascal (Tesla P100) fork: progress log

A running record of what was done, measured and ruled out, and what is still open. Details and methodology live in the
Pascal section of `README.md`, `doc/pascal_p100.md` and `doc/env_vars.md`; this file is the index. Add new entries at
the top of "Done", and move items between "Open ideas" and "Done" as they are measured.

Reference machine: Tesla P100 16 GB (SM 6.0, passively cooled, power cap 125 W, optionally 160 W) plus RTX 2060 6 GB
(SM 7.5). torch `cuda:0` = RTX 2060, `cuda:1` = P100; `nvidia-smi` index 0 = P100. Measured like `examples/chat.py`
(`-gs 4.0,15.0 -cs 32768 -cq 4`, MTP `-mtp -ndt 1 -dds`) with `examples/bench_chat.py`, `bench_chat_ab.sh` and
`bench_prefill.py`. P100 cooled to <= 62 C before every run, builds alternated (ABAB); it throttles at 79 C.

## Done (newest first)

| Date | Commit | What | Measured result |
| :--- | :--- | :--- | :--- |
| Nov 1 | `8e23476e` | Decode MoE as one persistent launch per layer (`EXL3_MOE_COOP_FUSED`): the two-launch blocks as work items of an atomic queue, EXL3 shared expert folded in; bit-identical | MoE kernels per layer (P100, m = 1 / m = 2 call): Qwen3.6 289 -> 156 us / 405 -> 268 us (rotation pre-kernels excluded), Ornith 155 -> 116 / 243 -> 192 us. End to end: Qwen3.6 plain 44.35 -> 55.45 t/s (x1.25), MTP 61.84 -> 71.55 (x1.16); Ornith plain 48.65 -> 51.92 (x1.07), MTP 61.79 -> 65.32 (x1.06) |
| Nov 1 | `f08a12ed` | fp16 GEMV (`hgemv_f16.cu`) for unquantized linears and `lm_head`, HFMA2 GEMM for their prefill (sm_60); shortcut limited to sm_60 / sm_75 | Ornith-1.5-35B-A3B 3bpw: plain 30.5 -> 48.65 t/s (x1.59), MTP 41.1 -> 61.7 t/s (x1.50); Qwen3.6-35B unchanged |
| Nov 1 | (setting) | Power limit 125 -> 160 W | Prefill +7..11% (35B), +19..31% (27B at 512 / 2048 tokens); decode unchanged; heat is the limit beyond ~8000 tokens |
| Oct 31 | `aec2e809` | `fa60`: HFMA2 flash attention for prefill, head_dim 256, sm_60; `free_mem` syncs all devices before `empty_cache` | Kernel 2.2-2.4x SDPA; end to end 27B +3.7% at 8192 / +7.1% at 16384 tokens; 35B +8% / +14%; load crash gone (0 of 15 loads) |
| Oct 31 | `77ff73de` | Dense 27B: 6 bpw GEMV (`lm_head`), sm_75 GEMV for mul1 K<=4, cheaper sm_60 mul1 front end | Qwen3.6-27B plain 10.2 -> 14.2 t/s (x1.39), MTP 14.4 -> 21.6 t/s (x1.50); Qwen3.8-27B 14.3 / 21.6 |
| Oct 31 | `7268757c` | Kernel-shape and int8 GEMV tests skip what does not fit 48 KB of shared memory | gemm + moe tests on the P100: 516 passed, 0 failed |
| Oct 30 | `95278d1f` | Prefill: fused MoE kernel on sm_60, HFMA2 GEMM for reconstruct-path GEMMs, GatedDeltaNet 16-column V tiles, reconstruct threshold 64 rows | 35B prefill x1.49 (2880 tokens 487 -> 723 t/s); decode unchanged |
| Oct 30 | `08d8e160` | sm_60 mul1 decode: native `vabsdiff4` byte sum, running prefetch offset in GEMV and MoE coop loops | 35B decode with MTP 50.5 -> 58.9 t/s (cool), x1.12 plain / x1.17 MTP sustained |
| Oct 30 | `dedf8a28` | Comparison against the unmodified `pascal-p100` branch (549b7eff) | Decode 35.3 / 34.0 / 32.8 -> 60.0 / 59.6 / 52.9 t/s (x1.6-1.75), prefill 191 / 187 / 181 -> 253 / 249 / 232 t/s |

Independent re-check of the persistent MoE kernel (main session, Nov 3, merged build, two turns of 500 tokens, greedy, P100 from
44-59 C at 1328 MHz, 160 W, 0 Xid events): Qwen3.6-35B-A3B plain 44.33 -> 54.53 t/s (x1.23), MTP 61.87 -> 71.64 t/s (x1.16,
96.9% accepted, same as before); Ornith plain 48.65 -> 51.82 t/s (x1.07), MTP 61.73 -> 63.16 t/s (x1.02, 82.5% accepted,
same as before; Opus measured 65.3). gemm + moe tests: 672 passed on the P100, 653 on the RTX 2060, 0 failed.

chat.py-like check (Nov 3, `bench_chat.py -sample` = the chat.py sampler; Qwen3.6-35B-A3B, `-mtp -ndt 1 -dds`, the prompt "Implement a quicksort in go", ~800-900 tokens, P100 from 60-67 C): `-cs 262144` greedy 69.8 t/s (94.2% accepted),
`-cs 32768` greedy 69.6 (94.2%), `-cs 262144` with the chat.py sampler 69.2 (92.6%) and 67.2 (90.3%). The cache size does not matter for decode speed.
A user run of `chat.py` on the previous build (no persistent MoE kernel, `.so` of Nov 1) gave 56.4 t/s at 88.2% accepted.

Models run on this machine: Qwen3.6-35B-A3B 3bpw (~71.5 t/s MTP, ~55 plain), Qwen3.6-27B and Qwen3.8-27B 4bpw (21.6 t/s MTP,
14.2 plain), Ornith-1.5-35B-A3B 3bpw (65.3 MTP, 51.9 plain), Muse-Glimmer-30B SC 3bpw (11.6 t/s plain; its DFlash drafter
loads and runs but accepts ~20% of tokens and is slower, 10.3 t/s).

Prefill by prompt length (125 W): 35B 440 / 796 / 754 / 680 / 565 t/s at 512 / 2048 / 8192 / 16384 / 30000 tokens;
27B 159 / 169 / 124 / 85 / 75 t/s (heat-limited from 8192 tokens on).

MTP settings (35B, two prompts, 160 W): `-ndt 1` is the safe choice (57.3 t/s); `-ndt 2` with `-dds` is clearly worse
(36); `-ndt 3 -dds -dc 0.6` gave 60.6 (+5%, within noise of two prompts); `-sdh` changes nothing.

## Ruled out (measured)

- Two-XMAD product for mul1 (ptxas rebuilds the 32-bit multiply from every PTX form tried), higher GEMV occupancy (the
  P100 GEMV is issue-bound, not memory-bound), cuBLAS `COMPUTE_16F`, fused-MoE row limits of 32 / 64, 32-row MoE tiles.
- mcg codebook work: mcg MoE coop kernels are not slower than mul1 (5.0 ms for 31 layers vs 5.5 ms for 30).
- Decode MoE knobs as a fix for the stage tails: `EXL3_MOE_COOP_KSPLIT=2/3/4` and `EXL3_MOE_COOP_WIDE=0` within +-10%
  or worse on the P100 (the narrow and split blocks are latency-bound at one block per SM). The persistent kernel for the
  lone 16 + 16 block shared-expert launch is slower (80 vs 71 us), so the single-expert launch keeps two launches.
- Same-input projection fusion (~0.2 ms per token), a tensor-core GEMV on sm_75, graphed decode for projections that are
  not EXL3, an fp32 Q.K variant of `fa60` (+55% time, no KL gain), `-ndt 2` for the dense 27B.
- Triton prefill attention on sm_60 (illegal memory access, predates the fork).

## Known issues

- `flash_attn_nc` (cacheless path) at `-gs 4.5,15.5` hits an illegal memory access on unmodified code; it disappears
  with `CUDA_LAUNCH_BLOCKING=1`. The P100 logged Xid 31 page faults before `fa60` existed; the `free_mem` fix removed the
  load-time crash, the original cause is only a hypothesis (`empty_cache` racing a kernel on another device).
- Checkpoints that quantize the Gated DeltaNet q/k/v separately (e.g. the Ling-3.0-tiny quant tried) cannot be loaded:
  the EXL3 path needs the fused `qkv_proj`. Ling models also use MLA, whose absorb kernel needs 64 KB of shared memory.
- Tests: 521 -> 616 -> 672 passed on the P100 (gemm + moe), 597 -> 653 on the RTX 2060, 0 failed. `tests/kernels/attention` has
  31 failures and 3 errors on sm_60 that predate this work; `test_fa75.py` must be deselected on the P100.
- Long prompts (>8000 tokens) on the dense 27B are heat-limited (79 C, 400-900 MHz): the measured numbers there are
  sustained, not kernel speed.

## Open ideas (not done)

- **MoE megakernel for decode, phase 1 profile (Nov 1, gate passed).** Measured per MoE layer on the P100 (torch
  profiler kernel times inside the real decode, `-gs 4.0,15.0 -cs 32768 -cq 4`, 1328 MHz, 50-58 C):

  | us per MoE layer | Qwen3.6 m = 1 | Qwen3.6 m = 2 | Ornith m = 1 | Ornith m = 2 |
  | :--- | ---: | ---: | ---: | ---: |
  | router GEMV + top-k (+ norm) | 19 (+5) | 21 (+5) | 16 (+4) | 16 (+4) |
  | shared expert, coop a + b (K5, 16 blocks each) | 62 + 41 | 82 + 56 | fp16 GEMVs | fp16 GEMVs |
  | routed rotation pre-kernel (m > 1 only, x2 with shared) | - | 7 + 7 | - | 8 |
  | routed coop a (gate/up) | 110 | 160 | 96 | 148 |
  | routed coop b (down) | 76 | 107 | 59 | 95 |
  | gaps between these launches (graph replay) | ~0.7 each | | ~0.7 each | |

  The brief's "182 us for mul1" was the routed pair only; with the K5 shared expert the Qwen3.6 MoE costs ~290 us of
  kernels per layer (8.7 ms of 18.1 ms P100 busy per token). Inside the kernels (globaltimer / clock64 per block,
  scratch build, microbenchmark at the same shapes): sm_60 compiles the coop kernels to 107-114 registers x 512
  threads, so **one block per SM**. Stage a has 64 blocks (8 slots x 2 projections x 4 groups of 128 columns): 56 run,
  then 8 SMs run a second ~55 us block while 48 idle (span 108 us, 57 us of mean block work). Stage b has 128 blocks of
  only 65 K weights (8 k-slices per warp), ~20 us each, latency-bound (2.2 weights/cycle/SM against 4.7 in a stage-a
  block), 3 waves where 2.3 would do. The shared expert's launches use 16 of 56 SMs. No counter waits (the counters are
  arrive-only). The sm_60 running prefetch offset is in the coop loop. The existing knobs do not fix it
  (`EXL3_MOE_COOP_KSPLIT=2/3/4`, `EXL3_MOE_COOP_WIDE=0`: within +-10% or worse). Avoidable time: well over 15% of the
  layer (tails ~40 + ~15 us, shared expert serial on 16 SMs ~100 us), so the gate passed; see Done for what was built.
  Left after it: the m > 1 input-rotation pre-kernel (2 x ~7 us per layer per MTP verify, could move into the queue),
  router GEMV + top-k (~19 us per layer, a separate pair of launches), finer or split-k items for the remaining tail
  (not bit-identical: the gate/up partials are fp16), lower register use for 2 blocks per SM (not tried).
- **MoE megakernel for decode** (proposed Nov 1; done in its persistent-queue form, see above). The decode MoE ran as two launches
  (`exl3_moe_coop`, shared expert as a one-expert coop launch, block coordination through counters). Floors for the 35B at
  m = 1 (8 routed experts + shared, ~28 M weights, ~10.6 MB at 3 bpw per layer): ~18 us at 593 GB/s, ~60 us from instruction
  issue (~8.6 slots per weight, 56 SMs x 64 lanes), against 161 us measured per layer (mcg, Ornith). The gap (~2.7x over the
  issue floor) is not weight traffic: candidates are grid syncs between the gate/up and down stages, tail imbalance across
  8 small experts, router + top-k + norm + residual as separate launches. A cut to ~100 us per layer would save ~1.9 ms of
  ~20 ms per token (~9%); at m = 2 (MTP verify) the layer costs 1.6x (7.7 vs 4.8 ms for 31 layers). Estimate, not measured.
- **MoE coop kernel at m = 2** (MTP verify): after the persistent kernel 268 us per layer (Qwen3.6) against 156 at m = 1,
  about linear in the distinct experts read (16 vs 8), so the per-row cost is not the issue; the rotation pre-kernels remain.
- **Keep small, read-every-token matrices in fp16 on the P100** (lm_head, GatedDeltaNet projections): Opus is sceptical
  (Nov 1, from the code and logged profiles, not measured). Per weight the quantized GEMV (~8.6 issue slots, ~500 G weights/s)
  is faster than the fp16 GEMV at 390-590 GB/s (~200-300 G weights/s); fp16 can only win on tiny shapes where launch or
  occupancy dominates, and the 35B's 6 bpw lm_head GEMV (~1.3 ms) is already no slower than fp16 (1.73 ms). Expected 0-3%,
  VRAM +2-3 GB. The earlier explanation that Ornith overtook Qwen3.6 in plain decode because fp16 reads are cheap is NOT
  established: a more likely contributor is that the Qwen3.6 EXL3 shared expert runs as two serial launches of 16 blocks
  (16 of 56 SMs) while Ornith's is fp16. Check before investing.
- P100 GEMV: at most ~15% left towards the ~7-slot floor. RTX 2060 GEMV: 15-25% left. 6 bpw `lm_head`: 3.3 ms against
  ~1.3 ms at bandwidth. `fa60` sits at ~42% of the HFMA2 peak (48 KB shared memory, 16-key tiles).
- Cooling the P100 (the real limit on long prompts); power cap above 160 W would only add heat.
- Text-quality check of generated output against `pascal-p100` (only KL / perplexity were checked).
- `-ndt 3 -dds -dc 0.6` on Ornith and the 27B; the rest of the MTP sweep was stopped.
- Muse-Glimmer: DFlash drafter accepts only ~20% (cause not established: quantization of the drafter, short prompts, or
  a numeric issue on sm_60); try `-dds`, a lower `-ndt` or a 4 bpw drafter.
- Qwen3.8-Flash-Next (125B total, 512 experts): 30 GB of weights at 1.66 bpw plus a 20 GB n-gram table; needs CPU expert
  offload (`-mcs`, experimental) and ~50 GB of disk. Dropped for now.
- Disk is nearly full (~12 GB free of 222 GB): the two 27B models and Ornith take ~50 GB.

## Ranking of next steps (Opus, Nov 1, estimates from code reading, none measured)

1. **DONE (Nov 1, see Done):** MoE coop work distribution (includes the m = 2 MTP case): stage a at m = 1 is 64 blocks of 512 threads on 56 SMs (8 SMs
   get 2 blocks, so the busiest SMs carry ~1.75x the ideal work), stage b is 128 short blocks dominated by start-up and
   prefetch latency, and the Qwen3.6 shared expert is a serial pair of launches with 16 blocks per stage. Fix: finer work
   units, shared expert folded into the routed launch as a 9th slot, possibly a and b in one persistent launch. Expected
   Qwen3.6 plain +5-9%, Ornith +4-7%, MTP similar; confidence medium, effort medium.
2. CUDA graphs for Ornith's non-EXL3 layers: cap ~5% (measured 1 ms gap), realistically 2-4%; Ornith only; effort medium.
3. P100 EXL3 GEMV, remaining ~15%: +3-5%; confidence low (several PTX forms exhausted); effort large.
4. fp16 for small matrices: 0-3%, sceptical (see above).
Not worth it: `fa60` tuning (prefill only, <2%), changing the `-gs` split (~0.5 ms per GPU, they run serially).

## MoE coop profile

Superseded by the persistent-kernel entry in "Done" and the phase 1 table in "Open ideas" (per-layer times before and
after, `EXL3_MOE_COOP_FUSED`).

## MLA shared-memory fallbacks (GLM-4.7-Flash), integrated Nov 3, `265b3945`

Three Python files of the Triton MLA layer (`modules/mla_attn.py`, `modules/attention_fn/bc_mla.py`,
`modules/attention_fn/mla_triton.py`) were changed by the user so that MLA models with `qk_nope_head_dim` 192 (padded to 256) fit
GPUs with <= 64 KB of shared memory (RTX 2060 64 KB, P100 48 KB): eager fallback on `BCKernelTooLarge`, `absorb_bn` ladder
128 / 64 / 32, absorb tiles sized with `pick_config` and a halving ladder with a `torch.bmm` fallback, gather block 32 below 64 KB,
prefill MHA `block_m` down to 16. Reported by the user: GLM-4.7-Flash 3bpw H4 over both GPUs, `-cs 262144 -cq 4 -mtp -ndt 1 -dds`,
~14.35 t/s, acceptance 66.7%. Check in the main session on the merged build: loads and generates, plain 10.8 t/s (2 turns of 500
tokens); the MTP run with `-cs 262144` failed with `CUDA error: out of memory` in `coop_autotune.cu:325` while the RTX 2060 held
475 MiB of desktop processes (Chromium and others); the cause was not isolated (desktop memory on the 6 GB card, the persistent MoE
scratch, or the context size) and the model was deleted afterwards, so the question is open. Ling-type models, which hit the same
absorb limit, were not retried.
