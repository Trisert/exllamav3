# Pascal (Tesla P100) fork: progress log

A running record of what was done, measured and ruled out, and what is still open. Details and methodology live in the
Pascal section of `README.md`, `doc/pascal_p100.md` and `doc/env_vars.md`; this file is the index. Add new entries at
the top of "Done", and move items between "Open ideas" and "Done" as they are measured.

Reference machine: Tesla P100 16 GB (SM 6.0, passively cooled, power cap 125 W, optionally 160 W) plus RTX 2060 6 GB
(SM 7.5). torch `cuda:0` = RTX 2060, `cuda:1` = P100; `nvidia-smi` index 0 = P100. Measured like `examples/chat.py`
(`-gs 4.0,15.0 -cs 32768 -cq 4`, MTP `-mtp -ndt 1 -dds`) with `examples/bench_chat.py`, `bench_chat_ab.sh` and
`bench_prefill.py`. P100 cooled to <= 62 C before every run, builds alternated (ABAB); it throttles at 79 C.

## Strategy: paths forward (Nov 6)

Desk analysis (no GPU runs): this log, `doc/env_vars.md`, the kernel and generator sources, the local model files and
web sources. Labels: **[M]** measured on this machine (where in this file or the code), **[D]** derived by arithmetic from
measured numbers, **[E]** estimate, **[L]** literature or third-party. The Qwen3.8-27B Stage 0 profile (branch
`dense27-stage0`) was running when this was written; every 27B kernel item below depends on it.

Use case: local coding agents (long contexts, tool outputs re-read every turn, mostly code out), 27B in non-thinking
mode with the model-card sampling (temp 0.7, top_p 0.8, top_k 20, presence 1.5).

### Findings that change the plan

1. **For agents, prefill dominates the turn, not decode.** A turn in an agent session is "new tool output in, a short
   tool call out". With 2500 new tokens and 300 output tokens per turn [E: assumed workload], prefill from the measured
   curves (27B 159 / 169 / 124 / 85 / 75 t/s at 0.5k / 2k / 8k / 16k / 30k tokens, heat-limited from 8k [M, "Done"];
   35B 440-796 [M]; Flash-Next 196-237 up to 16k [M]; beyond the measured range [E]):

   | seconds per turn (prefill + decode) | 10k context | 50k | 100k |
   | :--- | ---: | ---: | ---: |
   | 27B, today (heat-limited prefill, `-ndt 2` decode 28 t/s) | 22 + 11 = 33 | 39 + 12 = 50 [E] | 50 + 12 = 62 [E] |
   | 27B, P100 not throttling (prefill at kernel speed [E]) | 17 + 11 = 27 | 21 + 12 = 33 | 28 + 12 = 40 |
   | Qwen3.6-35B-A3B 3 bpw | 3 + 4 = 8 | 5 + 4 = 10 [E] | 7 + 5 = 11 [E] |
   | Flash-Next `-ndt 3`, full replies (32 t/s) | 11 + 9 = 21 | 12 + 10 = 23 [E] | 14 + 11 = 25 [E] |

   Prefill is 65-80% of a 27B turn. The kernel-speed column assumes 170 t/s for the GEMMs plus attention that grows
   with context: 16 layers x 24 heads x 256 x 4 FLOP per key = 0.39 MFLOP per token per 1k of context, against 54 GFLOP
   per token for the weights, so at 50k / 100k attention adds +36% / +73% of the FLOPs (at fa60's lower efficiency
   ~1.2x that) [D]. The 75 t/s at 30k is heat (400-900 MHz instead of 1328) [M], so **cooling the P100 is the largest
   single prefill lever** for the 27B, and it costs no agent time. Kernel side: 169 t/s x 54 GFLOP = 9.1 TFLOP/s, ~48%
   of the P100's HFMA2 peak (~19 TFLOP/s) [D], fa60 at ~42% [M], so sm_60 prefill kernels have < ~1.5x left in total.
2. **The 27B decode floor is not the ~30 ms bandwidth floor.** Weights read per token: decoder 11.39 GiB (MLP 7.98,
   GatedDeltaNet 2.63, attention 0.78) + lm_head 0.89 GiB (6 bpw) = 12.28 GiB [M: safetensors headers; the 2.37 GiB
   embedding is a CPU table, the vision tower is not read]. At the measured read peaks (P100 450-590 GB/s, 2060 280-300)
   with 16 of 64 layers on the 2060 that is 27-33 ms [D]. But the EXL3 GEMV is issue-bound on both cards: P100
   17408 x 5120 at 4 bpw takes 212 us at m = 1 = 420 G weights/s, 210 GB/s [M: comment in `exl3_gemv.cu`]; the 2060
   runs 16 layers of the 27B in 18 ms = 340 G weights/s [M: `EXL3_GEMV_SM75` entry]. 19.6 G weights on the P100 (incl.
   lm_head) / 420 G/s + 6.1 G on the 2060 / 340 G/s = **47 + 18 = 65 ms, consistent with the measured plain decode** [D]
   (caveats: the 420 G/s is one shape, and the 2060's 18 ms is its total GPU time including non-GEMV ops; Stage 0 gives
   the real split). So the 27B decode is GEMV per-weight cost to first order, confidence medium. The ideal ~8.6 issue
   slots per weight (~550 G weights/s on the P100 at 1328 MHz) would give ~36 + 18 = 54 ms [D], so the kernel headroom
   left in the P100 GEMV is ~20-25%, not 2x. Kernel fusion and graph work on the 27B are bounded by whatever non-GEMV
   time Stage 0 finds (the BC attention / GDN / MLP decode paths are already graphed).
   Corollary: **on sm_60 fewer bits do not buy decode speed** (cost scales with weights, not bytes; weak evidence: the
   K = 3 5120 x 17408 GEMV at ~0.3 ms vs K = 4 17408 x 5120 at 212 us, different shapes [M, code comments]). Do not go
   below 4 bpw on the 27B for speed.
3. **The 2-GPU pipeline costs the 27B ~18 of 65 ms**, and tensor parallelism is the only lever that removes it. The two
   GPUs never overlap in one stream [M]. Upstream has TP (`-tp`, `native` backend: all-reduce through pinned host memory,
   sm_60 fallbacks for the acquire/release loads in `ptx.cuh` exist) and the MTP head borrows the target lm_head in TP
   mode (`Model.lm_head_argmax`); `-sdh` is disabled under TP (`generator.py` line 449), CPU MoE offload asserts
   layer-split (so no TP for Flash-Next), BC graph paths fall back under TP (eager decode). Never tried on this fork.
   Bound for the 27B: a full layer set costs 72 ms on the 2060 and 58 ms on the P100 [D, from point 2]. The 2060 can
   hold ~35% of each layer (5.6 GiB minus ~0.57 GiB of desktop, minus cache and scratch [M/E]), so TP time =
   max(0.35 x 72, 0.65 x 58 + 3.3 lm_head) = 41 ms + 128 all-reduces per token (2 per layer; >= 50 us each if
   they cost what the measured cross-GPU event ping-pong costs [M], so >= 6.4 ms) = **~45-50 ms vs 65 (x1.3-1.45
   plain)**, excluding the host overhead of eager multi-process TP decode, which is unknown and could eat the gain;
   MTP steps scale alike [E]. Confidence low. Prefill:
   the 2060 takes ~35% of the FLOPs instead of 25% of the layers [E: x1.2-1.35, and less P100 heat]. For Qwen3.6 the
   2060's serial share is 5.5 of 27.7 ms [M] minus ~80 all-reduces: <= ~10% [E], not worth it.
4. **PCIe: the P100 sits on a gen3 x4 link, the 2060 on gen3 x16** [M: `/sys/bus/pci/devices/0000:03:00.0`
   (0x15f8) `current_link_width` 4, max 16; `0000:06:00.0` (0x1f08) x16; cross-check with `nvidia-smi -q`]. ~3.5 GB/s
   to the P100 caps TP prefill all-reduces (42 MB per all-reduce at a 4096-token chunk, 128 per chunk: ~1.5 s per
   ~24 s chunk [D]) and the GPU-streamed experts of Flash-Next prefill (`EXL3_MOE_STREAM_T` scales its threshold with
   the probed link rate). Moving the P100 to the CPU-attached x16 slot is a user decision worth checking.
5. **Presence penalty is applied over the prompt too.** `SS_PresFreqP` penalizes every token found in `past_ids`, which
   is the whole `sequence_ids` (prompt + output; `job.prepare_sampling_past_ids`), windowed to the last sustain + decay
   tokens (`rep_pen.cu`; `chat.py` defaults both to `-penr` 1024, so full penalty over 1024 tokens fading over 1024 more).
   vLLM's `presence_penalty`, which the Qwen card assumes, counts the generated text only [L: vLLM SamplingParams docs:
   "based on whether they appear in the generated text so far"; `repetition_penalty` is the one that includes the
   prompt]. In an agent turn the last 1-2k tokens are the tool output the model must quote (edit `old_string`, paths,
   identifiers), so presence 1.5 here pushes the model away from exact copies: a code-fidelity risk and an acceptance
   loss for MTP and suffix drafting. It also forces the serial verification walk (`reqs_past_ids`). The card itself warns
   that high values cause "language mixing and a slight decrease in model performance" [L: model README]. Note also that
   the card's SWE-bench Pro / DeepSWE numbers were run in thinking mode (temp 1.0, top_p 0.95, presence 0) with the
   Claude Code harness [L], not with the non-thinking preset.
6. **Verify width has cliffs on sm_60, so long drafts do not transfer from the literature.** The GEMV covers m <= 8
   (`EXL3_GEMV_MAX_M`); m = 9..64 runs the emulated-MMA tile GEMM, ~16x slower per projection (5120 x 17408 K = 3:
   ~4.8 ms vs ~0.3 ms, `exl3_gemv.cu` comment [M]); reconstruct + HFMA2 GEMM only from 64 rows
   (`EXL3_RECON_THRESHOLD_SM60`). Inside m <= 8, MMODE 4 covers m = 3..4 only, m = 5..8 take the slower MMODE 1; the
   gated-residual decode kernels stop at R = 4; the 27B jumps at 4 rows (unexplained, Stage 0). SuffixDecoding's 3.9x
   on SWE-bench agent traces [L: Oliaro et al., NeurIPS 2025; Snowflake Arctic Inference ~4x] assumes a near-free wide
   verify (H100-class, memory-bound); here a hybrid draft of more than 7 tokens would cost a tile-GEMM pass.
7. **Sampling is not a lever.** Verification samples the target at each position and compares with the drafted argmax,
   which is speculative sampling with a deterministic draft (acceptance = p_target(draft)); only multi-candidate (tree)
   drafts raise acceptance under sampling. Measured: 27B `-ndt 2` 27.9 t/s with the model-card sampling vs 28.2 greedy
   [M, Nov 5 session notes]. Literature: EAGLE-3 acceptance length at T = 1 is ~6% below T = 0 on MT-Bench / HumanEval
   [L: LK-losses paper, arXiv 2602.23881, Table of T = 0 / 1].
8. **The ~7 ms of "no GPU" per Flash-Next step is the CPU expert wait, not launch gaps** (unprofiled ~7 ms at `-ndt 3`,
   CPU worker ~5.5 ms busy per step [M, "draft length and MTP step profile"]). CUDA graphs would not remove it; starting
   the CPU job earlier or moving hot experts to the GPU would.
9. **Quality: at full precision the 27B and Flash-Next are about tied on coding; the quant gap is not.** Model-card /
   aggregator numbers [L]: SWE-bench Pro 27B 61.7, Flash-Next 62.5, Qwen3.6-35B-A3B 49.5 (Qwen3.6-27B 53.5); LiveCodeBench
   v6 90.3 / 91.9 / 80.4; NL2Repo 42.3 / 48.1 / 29.4. EXL3 KL vs bpw on Llama-3.1-8B (repo chart
   `doc/llama31_8b_instruct_kld_bpw.png`): 4.0 bpw 0.013, 2.5 0.151, 2.0 0.312, 1.8 0.764 [L]; a third-party Gemma-3 27B
   set: 4.0 bpw 0.039, 3.0 0.11, 2.0 0.45 [L]. Flash-Next runs 1.5 bpw routed experts (5 bpw attention). Large MoE experts
   are believed more tolerant of low bits than a dense 8B, but the quant's own card shows a real loss: the
   `turboderp/Qwen3.8-Flash-Next-exl3` revisions score GPQA-Diamond / ERQA 86.4 / 62.7 at 1.66 bpw, 89.4 / 64.3 at 2.18,
   91.9 / 68.5 at 3.68, against 91.7 / 68.8 unquantized [L: HF model card]. The 27B scores GPQA-D 89.2 unquantized
   [L], and 4 bpw is near-lossless territory on every KL chart above. On this evidence the 1.66 bpw Flash-Next is likely
   below the 4 bpw 27B on reasoning; a local coding check (P9) is still the deciding measurement. Confidence medium.

   Other model options, from what has run here: Qwen3.6-35B-A3B at 3.5 bpw would fit (~15.3 GB [D]; 4 bpw ~17.5 GB does
   not leave room for cache), a small step on the Llama KL chart (3.0 -> 3.5 bpw: 0.053 -> 0.030 [L]) for a coder that
   is 12 points behind on SWE-bench Pro; Muse-Glimmer-30B ran at 11.6 t/s with a DFlash drafter accepting ~20% [M] and
   loses all 8 overlapping benchmarks to the Qwen3.8-27B [L: orcarouter summary of the card]; GLM-4.7-Flash ran at
   ~11-14 t/s [M]. Mixed precision on the 27B (more bits for small sensitive tensors) is cheap in time only up to 4 bpw:
   5-6 bpw tensors take the staged shared-memory GEMV, which is slower per weight (the 6 bpw lm_head runs ~385 G
   weights/s [D: 1.27 G weights in 3.3 ms] vs ~420 for 4 bpw); check per tensor before converting. The P100 + 2060 is
   best used for one dense ~27B model at 4 bpw with MTP, all in VRAM, with a fast A3B MoE as the fallback.

### Headroom per model

| | measured | floor | gap is | confidence |
| :--- | :--- | :--- | :--- | :--- |
| Qwen3.8-27B 4 bpw, plain | 65 ms/token [M] | bandwidth 27-33 ms [D]; GEMV per-weight cost today 65 ms [D]; ideal issue rate ~54 ms [D] | GEMV issue cost (both cards) + 2-GPU serialization (18 ms); TP is the large lever, kernels ~10-15% | medium (Stage 0 pending) |
| 27B MTP, code / prose | 28.2 (`-ndt 2`; 27.9 sampled) / 22.2 (`-ndt 1 -dds`) t/s [M] | with TP ~36-40 / ~29-31 t/s [E] | m = 4 step jump (Stage 0); drafting is fine on code, prose ndt 2 unmeasured | low (TP) |
| Qwen3.8-Flash-Next 1.66 bpw | ~84 ms per `-ndt 3` step, 3.3-3.7 tokens [M]; full replies code 31.7, prose 17.5-19.6 t/s [M] | MoE bytes ~50 us/layer vs ~600 measured (issue-bound decode of 1.5 bpw trellis) [M] | P100 busy ~60 ms, 2060 13 ms serial, CPU ~7 ms; remaining listed items 1-3 ms each; prose acceptance 29-40% [M] is a drafting limit | high |
| Qwen3.6-35B-A3B 3 bpw | ~26.4 ms/iter, 74 t/s [M] | the log's candidate list reaches ~21.7 ms (midpoints) [E] | MoE m = 2 (now partly addressed by the m34 kernels, not re-measured on Qwen3.6) | medium |

CPU: idle for the 27B and Qwen3.6 (only the host thread); the AVX2 expert kernel does ~2 GB/s of trellis per core [M],
so CPU compute cannot take a useful share of a dense model. It is enough for suffix-automaton drafting.

### Candidate paths

Gains are per model; "code" = code-heavy agent output, "prose" = explanations. h = hours of agent work.

27B baselines for the absolute numbers: code 28.2 t/s (`-ndt 2`), prose 22.2 t/s (`-ndt 1 -dds`), prefill 159-169 t/s
short / 75 t/s at 30k [M]. Confidence is about the size of the gain.

| # | Path | Model | Expected gain | Effort | Risk | Confidence | Depends on |
| :--- | :--- | :--- | :--- | ---: | :--- | :--- | :--- |
| P1 | Cool the P100 (blower / shroud), then re-measure prefill at 8k-30k | all | 27B prefill at 30k 75 -> ~135 t/s [E, kernel-limited estimate]; removes thermal drift from every A/B; decode unchanged | 0 h (user) | none | medium (the 400-900 MHz clocks are measured) | - |
| P2 | Agent benchmark: `eval/spec_decode.py` agentic_code / agentic_curl traces (3k-30k tokens, `-temp`), reporting prefill t/s, decode t/s, tokens/step per category; a step-cost script (ms/step at fixed m = 1..8, 16, 32, 64); a resident-process A/B harness that toggles per-call env switches | all | none directly; makes every later gain measurable on the real workload and replaces the single quicksort prompt | 4-6 h | low | high | - |
| P3 | Presence penalty over generated tokens only (option or default: window starts at the first generated position); also verify what the user's OpenAI-compatible server passes as the penalty window (it may be the whole context) | 27B (non-thinking preset) | code fidelity of edits [unquantified]; decode code 28.2 -> ~28-29.5 t/s from acceptance on agent turns [E]; host walk parallel again | 2-4 h | low | low on t/s, high that the scope differs from vLLM | P2 to measure |
| P4 | TP probe for the 27B: `-tp -gs a,b` load, `bench_chat` plain + MTP, KL vs layer-split; then tune the split | 27B | plain 15.3 -> ~20-22 t/s, MTP code 28.2 -> ~36-40, prose 22.2 -> ~29-31, prefill x1.2-1.35 [E] | 2 h probe, +6-12 h if it runs | high: untested on sm_60, eager decode under TP (host overhead not in the estimate), x4 link, 2060 VRAM | low | - |
| P5 | 27B Stage 0 top item (likely the m = 4 jump, then GEMV / lm_head at m = 3..4) | 27B | if m = 4 costs like m = 3: `-ndt 3` code ~31-34 t/s vs 28.2 [E]; prose 0 | 4-10 h | medium | low (cause unknown) | `dense27-stage0` |
| P6 | P100 GEMV MMODE 4 for m = 5..8 and slab dispatch (8-row GEMV slabs) for m = 9..32 instead of the tile GEMM | 27B, Flash-Next | enables `-hdt` 4-15 and wider drafts at ~linear cost; nothing by itself | 6-10 h | medium (bit-identity per row is testable) | medium | P2 step-cost curve |
| P7 | `-hdt` suffix drafting on agent turns (exists, never measured here). Bound [D, CPU script on `agentic_code_29` + `agentic_curl_16`, 12k assistant tokens, tool calls rendered approximately]: 41-47% of output tokens lie in spans whose last 3-7 tokens occurred earlier; suffix drafting alone gives 1.84 tokens/step at H = 3, 1.69 at H = 7, 1.44 at H = 15 (MTP `-ndt 2` gives ~2.6-2.8 on code [M]) | 27B, Qwen3.6, Flash-Next | on top of MTP only where suffix drafts are longer than MTP's: edit-heavy turns x1.05-1.3 [E], prose 0; on the 27B `-hdt 2` is just m = 3 and `-hdt 3` lands on the m = 4 jump | 1-2 h to measure | low | low | P2; P5 for the 27B; P6 for `-hdt` > 3. Measure first on Qwen3.6 |
| P8 | Agent serving config: `-ccs` + `-dcd` (prefix pages and recurrent checkpoints survive reloads and evictions), `recurrent_checkpoint_interval_pp` 4096-8192 (default 32768: a history edit at 60k re-prefills from 32k, ~6 min), `-cs` >= 131072 (4-bit KV = 16 KB/token, 1.6 GB at 100k [D]); GDN checkpoint ~80 MB [D], 4 GB `-rcs` = ~50. Reuse needs a byte-stable history: the 27B template keeps past reasoning (`preserve_thinking` defaults on), so it is the harness that must resend history unchanged | 27B | avoids multi-minute re-prefills; restores a 100k context from disk in ~10 s at 210 MB/s instead of ~15 min of prefill [D] | 2-3 h incl. a check | low | high (when misses happen) | - |
| P9 | Quality gate: `eval/humaneval.py` (or 20 agent tasks) on 27B 4.0 vs Flash-Next 1.66 vs Qwen3.6 3.0 | model choice | decides whether any more Flash-Next work is worth it (card GPQA already points to the 27B, finding 9) | 3-4 h (mostly GPU time) | low | - | - |
| P10 | Prefill: pipeline consecutive chunks across the two GPUs (2060 on chunk n+1 while the P100 runs chunk n; GDN state order is per layer, so valid) | 27B, Qwen3.6 | <= the 2060's share of prefill time: 27B ~170 -> ~190-200 t/s short context [E] | 6-12 h | medium | low (2060 prefill share unmeasured) | P1 (heat otherwise dominates) |
| P11 | fa60 attention (branch `fa60-prefill`, kernel -14% done) | 27B at long context | attention is ~25-45% of prefill FLOPs at 50-100k [D]; 1.5x faster attention = +8-15% there [E] | 4-8 h | low-medium | medium | P1 |
| P12 | Cost-aware draft length: pick ndt per round from recent acceptance x the measured ms/step(m) table (`-dds` only truncates on confidence) | 27B, Flash-Next prose | 27B prose 22.2 -> ~23-25 t/s [E]; Flash-Next prose `-ndt 2` is already +12% over ndt 3 [M] | 3-5 h | low | low | P2 step-cost curve |
| P13 | 2060 tensor-core prefill with weights streamed from pinned RAM over its x16 link (12.3 GB per chunk ~1 s) | 27B | unknown, up to ~2x prefill if the 2060 sustains ~20 TFLOPs on EXL3 reconstruct + GEMM [E, unmeasured] | 2-4 days | high | very low | microbenchmark first (see uncertainties) |
| P14 | Flash-Next leftovers (CPU job earlier, 2060 K = 5 GEMV, more GPU expert slots) | Flash-Next | 1-3 ms each of ~84 per step [M/E], <= ~10% total (code ~42 -> <= 46 t/s), each inside the 5% load spread | 4-10 h | medium | medium | P9 says Flash-Next is worth it |
| P15 | Qwen3.6 re-measure on the current build (MoE m34 / dual kernels apply at m = 2) | Qwen3.6 | unknown, the Flash-Next m = 2 microbenchmark went 484 -> 383 us [M] | 0.5 h | none | - | - |

**Recommended order for the next five work items:**

1. **P2, agent benchmark + step-cost curve + resident A/B harness.** Every open question below (penalty, `-hdt`, ndt on
   prose, the verify cliffs, TP) is decided by these numbers, and the current single-prompt `bench_chat` method has a
   ~5% load-to-load spread that hides most of the remaining items.
2. **P3, presence-penalty scope**, with P8 (agent serving config) in the same session. Cheap, about correctness of the
   actual use case, measured with P2 (agentic traces sampled with presence 1.5 over prompt vs over output vs 0: acceptance
   and whether quoted strings in tool calls match the context byte for byte).
3. **P4, TP probe for the 27B.** The only item with a plausible x1.3-1.45 on decode and x1.2-1.35 on prefill
   (confidence low). Two hours decides it; stop at the first blocker that needs more than a day.
4. **P5 (+P6 if the curve asks for it), the 27B Stage 0 top item.** After Stage 0 lands; P6 only if P2 shows m = 5..8 and
   9..16 are where agent turns want to verify (P7).
5. **P9, quality gate.** Before any more Flash-Next kernel work (P14). If Flash-Next 1.66 bpw does not beat the 27B on
   coding, freeze it as is.

In parallel, for the user: **P1 (P100 cooling)** and the slot question (finding 4).

### Recommended NOT doing (with evidence)

- **Tree / Medusa-style drafts and lookahead decoding on the P100.** Verify cost grows per row (issue-bound GEMV), m > 8
  falls to the tile GEMM (~16x per projection [M, code comment]), and GDN layers need a branched state (STree [L: Wu et
  al., NeurIPS 2025] needs custom scan kernels). Revisit only if P6 makes m = 9..32 cheap.
- **More Flash-Next kernel micro-work before P9.** Remaining items are 1-3 ms of ~84 ms each [M], the model does not fit
  in VRAM (CPU experts, 152-170 s loads, n-gram table random reads on the USB SSD at long context [M]), prose decode is
  17-20 t/s from a 29-40% draft acceptance that kernels cannot fix [M], and its 1.66 bpw quality is unmeasured.
- **The Qwen3.6 100 t/s target**, as long as quality matters more than speed: its own budget reaches ~89 t/s only with the
  MoE rewrite at its optimistic end ("Candidates towards 100 t/s"), and it is the weakest coder of the three (SWE-bench
  Pro 49.5 vs 61.7 [L]). Keep it as the fast fallback model (a turn costs ~8-11 s vs ~33-62 s for the 27B [E]).
- **Lower bpw on the 27B for speed** (finding 2: per-weight cost), **KV below 4 bit** (1.6 GB at 100k is ~4 ms of a
  65 ms step at 400 GB/s [D]), **`-rs32`** (doubles state and checkpoints, no speed).
- **Draft model on the 2060 / overlapping draft with verify.** With `-sdh` the draft costs ~0.5 ms on the 2060 plus a
  near-free head [M, Qwen3.6 Stage 0], so overlap saves <= ~2%; a separate drafter needs VRAM the 27B uses and MTP
  already accepts 85-97% on code [M].
- **CUDA graphs for the Flash-Next idle time** (finding 8), **8-bit gated-residual tables** (no gain, [M]),
  **`-ndt 4`** on Flash-Next and **`-ndt 3`** on the 27B until the m = 4 jump is explained [M], **router fusion**,
  **all layers on the P100 / experts split across GPUs** [M], **power cap above 160 W** (heat, [M]).
- **Rejection-sampling variants for the model-card sampler** (finding 7): the loss to sampling is ~1% on code [M].

### Key uncertainties and the cheapest measurement for each

| Uncertainty | Why it matters | Cheapest measurement |
| :--- | :--- | :--- |
| ms/step of the 27B and Flash-Next at m = 1..8, 16, 32, 64 | sets `-hdt`, ndt, P6, P12; explains the m = 4 jump | one load, a loop of `model.forward` at q_len m with a fixed cache position, 20 min |
| Does `-tp` load and run on P100 + 2060 (sm_60 paths, eager decode, x4 link), and what is the all-reduce cost per token | P4, the largest decode + prefill lever | one load of the 27B with `-tp -gs 4.5,15` (tune), `bench_chat` plain then `-mtp -ndt 2`, plus a KL check against layer-split: 1-2 h |
| Presence penalty over prompt: effect on acceptance and on exact copies in tool calls | correctness of the real use; P3 | `spec_decode.py -single "Agentic, code" -temp` (after adding presence to its sampler) with 1.5 vs 0 vs the fixed scope, diff the tool-call strings against the trace: 1 h |
| `-hdt` gain on top of MTP on agent traces (CPU bound: 41-47% of output tokens copyable, finding in P7) | P7 | on Qwen3.6 (GEMV covers m <= 8; its MoE cost grows with the distinct experts): `spec_decode.py` agentic workloads, `-mtp -ndt 1` vs `+ -hdt 3 / 7`: 1 h; on the 27B after P5 |
| Prose acceptance of the 27B MTP head under sampling, ndt 1 vs 2 | P12; agent replies contain prose | `spec_decode.py` creative/translation categories with `-temp`: 1 h |
| Flash-Next 1.66 bpw vs 27B 4.0 bpw coding quality | model choice, P14 | `eval/humaneval.py` on both (HumanEval is near saturation, so treat a gap > 3 points as signal), or 20 real tasks from the user's agent: 3-4 h |
| 27B prefill at 32k / 64k / 131k with and without heat | the end-to-end limit (finding 1) | `bench_prefill.py` at those lengths, logging clocks (`nvidia-smi dmon`): 1 h; repeat after P1 |
| Can the P100 clocks be locked (`nvidia-smi -ac` / `-lgc`) below the throttle point | removes thermal drift from A/B runs | one `nvidia-smi -q -d SUPPORTED_CLOCKS` and a 10-minute steady run: 15 min |
| 2060 EXL3 prefill GEMM rate (reconstruct + `hgemm_f16acc`) at 27B shapes vs the P100 | P13 and the P10 share | microbenchmark of a 5120 x 17408 layer at m = 4096 on each card: 30 min |
| Qwen3.6 speed on the current build | whether its numbers in this log are stale | one `bench_chat` run: 10 min |
| P100 link really x4 under load | finding 4 | `nvidia-smi -q -d PCI` during a load: 1 min |

### Tooling and methodology

- **Loads.** The 27B is 15.7 GiB on a USB SATA SSD (~210 MB/s [M]): ~80 s cold. It fits the page cache of the 31 GB
  machine if Flash-Next (28 GiB of weights) is not loaded in between, so batch 27B sessions and avoid alternating models;
  a resident process (P2) removes most loads altogether. An internal disk would roughly halve the 85 s weight read of
  Flash-Next [E, Stage 0 list].
- **Noise.** Load-to-load spread is ~5% [M] and greedy text is not reproducible across loads at full reply length [M],
  so 8-load ABAB/BABA is what the last items needed. Split the metric: kernel work is judged by ms/step at fixed m in a
  resident process (switches read per call, e.g. `EXL3_GEMV_M34`, can be toggled in-process; mark each switch in
  `env_vars.md` as per-call or load-time), drafting work by tokens/step over the 22 `spec_decode.py` workloads
  (`-dstats` records per-round data). t/s is then the product, with much less noise than single runs.
- **Thermal drift.** P1, or locked clocks (table above); log clocks and temperature with every run.
- **Builds.** ccache is installed and torch's JIT puts it in front of nvcc; header edits rebuild every including unit
  (`exl3_gemv_kernel.cuh`, `exl3_moe_coop_kernel.cuh` are widely included), so keep experiments in `.cu` files where
  possible. 16 jobs only with no model loaded.

Sources: [vLLM SamplingParams](https://docs.vllm.ai/en/latest/api/vllm/sampling_params.html);
[SuffixDecoding, NeurIPS 2025](https://neurips.cc/virtual/2025/poster/115453);
[Snowflake Arctic Inference + suffix decoding](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic);
[STree, NeurIPS 2025](https://arxiv.org/pdf/2505.14969); [EAGLE-3](https://arxiv.org/pdf/2503.01840);
[LK losses (EAGLE-3 at T = 0 / 1)](https://arxiv.org/pdf/2602.23881);
[Qwen3.8-27B benchmarks](https://orcarouter.ai/blog/qwen-3-8-27b-benchmarks) and the local model card
(`models/Qwen3.8-27B-exl3-4.00bpw/README.md`); [BenchLM Qwen3.6-35B-A3B vs Qwen3.8-Flash-Next](https://benchlm.ai/md/compare/qwen3-6-35b-a3b-vs-qwen3-8-flash-next.md);
[Gemma-3 27B EXL3 KL by bpw](https://huggingface.co/WeReCooking/Gemma-3-R1984-27B-EXL3);
[exl2 TP vs pipeline on 2x 3090, 17.8 -> 24.6 t/s](https://red.127.pp.ua/r/LocalLLaMA/comments/1f5qcdl/simple_tensor_parallel_generation_speed_test_on)
(anecdotal, symmetric cards with P2P-less PCIe). No reliable llama.cpp P100 decode number for a 27B was found.

## Upstream sync (Nov 4): turboderp-org/exllamav3 v1.6.6 + LFM3 (branch `pascal-sync-v166`)

Merged `upstream/master` into the fork (one conflict, `block_sparse_mlp.py` scratch buffer: kept `torch.zeros`, not verified
that every sm_60 kernel writes every slot). Verified on the P100 (+2060 for the sm_75 kernels):

- Kernel tests `gemm + moe + attention + routing`: 1469 passed, 59 skipped, 25 failed, all 25 in
  `routing/test_routing_gemm_det.py`: the deterministic int8 routing GEMM is sm_80+ only by design
  (`routing_gemm_det_fits` returns false for cc < 8, cuBLAS serves other arches) and the test does not gate on the arch.
  Not a regression. `attention/test_fa75.py` and `test_prefill_sdpa_window.py` are sm_75 kernels: they crash on the P100
  ("unspecified launch failure") and the crash poisons the CUDA context, which made ~370 later tests fail in a full run
  (each file passes alone); on the 2060 they pass (49 passed, 8 skipped). Run them with `CUDA_VISIBLE_DEVICES=0`.
- Decode on the merged build (`bench_chat.py`, MTP `-ndt 1 -dds`, one turn, run while the prefill agent shared the GPU):
  Qwen3.6 70.90 t/s (94.8% accepted) and Ornith 67.44 t/s (84.5%). Re-measured on an idle GPU (two runs each): Qwen3.6
  73.97 / 74.05 t/s, Ornith 67.40 / 67.61 t/s. No regression against 72-73 and 65-69 before the merge.
- LFM3-1.7B quantised to 4.0 bpw (head 6 bpw): `models/LFM3-1.7B-exl3-4.0bpw`. The 2060 (5.6 GiB) ran out of memory on the
  lm_head (vocab x hidden, 6 bpw); `convert.py -w <work> -r -d 0` resumed it on the P100 in 4 minutes. `chat.py` has no LFM
  prompt mode, `-mode chatml` is the closest (`<|im_start|>`), untested.

## Qwen3.8-Flash-Next (Nov 5, baseline)

`turboderp/Qwen3.8-Flash-Next-exl3`, revision `1.66bpw_h4_ng3` (`models/Qwen3.8-Flash-Next-exl3-1.66bpw`, 46.8 GiB on disk:
28.1 GiB weights in 4 shards + 18.5 GiB n-gram embedding table). `Qwen4Exp`: 125B params / 6B active, 48 layers (36 gated delta
net + 12 Qwen Sparse Attention), 512 experts (10 routed + 1 shared, intermediate 640), gated residual (hc_count 4), n-gram embedding
(streamed from disk by default, `-ngr` loads it in RAM), MTP head. The weights do not fit the 19 GiB of the two GPUs, so routed
experts of the first N layers run on the CPU (`-mcl N`, 31 GB RAM machine).

- Blocker fixed: the QSA indexer kernel (`_dsa_indexer_fewq_kernel`) needs 64 KB of shared memory, the P100 has 48 KB, and
  `autosplit_prepare` did not catch `BCKernelTooLarge` (the runtime path does and falls back to the eager attention). Commit c2993f8d.
- `-mcl 16`: "Insufficient VRAM in split for model and cache"; `-mcl 30` loads and answers correctly.
- Baseline, `-gs 4.0,15.0 -cs 8192 -cq 4 -mcl 24`, `bench_chat.py` code prompt, 300 new tokens, greedy: **17.77 t/s** without MTP;
  **23.43 t/s** with `-mtp -ndt 1 -dds` (96.1% accepted). Load plus 300 tokens takes under 3 minutes. Re-measured in Stage 0
  with the same flags and `bench_chat.py`: 19.44 / 25.78 t/s (cause of the difference unknown).
- Not yet known: where the decode time goes (CPU expert worker vs GPU), the smallest `-mcl` that fits, `-mcs` (tail split) vs `-mcl`,
  cost of the n-gram table, prefill speed, behaviour of the eager QSA path on sm_60 at long context. Answered in Stage 0 below.

### Qwen3.8-Flash-Next Stage 0: measurements (Nov 5)

Method: one load per configuration (`flash_body.py` harness: bench-style decode of the code prompt, 300 tokens greedy, plus a torch
profiler window of 30 iterations; `EXL3_MOE_CPU_PROF=1` for the CPU worker's per-job times, steady state taken as the delta of
the cumulative counters). `-gs 4.0,15.0 -cq 4 -cs 8192` unless noted, P100 cooled to <= 62 C before each load (61-71 C during
the runs, 1328 MHz). Run-to-run spread between loads of the same config is ~5% (mcl 24: 18.40 / 19.28 / 19.44 t/s), so
single-load differences below that are noise. Machine facts that matter here: Ryzen 7 5800X (8 cores, AVX2), DRAM read
40-44 GB/s (torch sum, 1-16 threads); the model lives on `/home`, a LUKS volume on a **USB-attached Kingston SA400 SATA SSD**
(~210 MB/s measured while loading); 31 GB RAM, so the 28 GiB of weights cannot stay in the page cache between loads.

Harness pitfall (cost a sweep): a sampler thread that held `sys._current_frames()` frames during `model_init.init` kept the
loader's CUDA tensors alive and made every marginal config fail with "Insufficient VRAM"; with the frame dropped (or no
sampler) the fits below are reproducible. As a precaution (not a confirmed cause) the runner also waits until no python CUDA
process is left from the previous run before loading.

**Fit and decode speed** (`bench_chat.py` where marked, otherwise the harness; t/s over 300 tokens):

| Config | GPU alloc 2060 / P100 (GiB) | decode t/s | ms/iter | CPU job (us, per layer-token) |
| :--- | ---: | ---: | ---: | ---: |
| `-mcl 18 / 20 / 22 / 23` | - | does not fit | | |
| `-mcl 24` (7 worker threads, default) | 2.91 / 13.91 | 18.40-19.44 (`bench_chat` 19.44) | 51.9-54.3 | 620 (14.8 GB/s) |
| `-mcl 24 -mct 4` | same | 15.68 | 63.7 | 1039 (8.7 GB/s) |
| `-mcl 24 -mct 8` | same | 19.86 | 50.4 | 562 (16.3 GB/s) |
| `-mcl 24 -mct 15` (SMT siblings) | same | 18.00 | 55.5 | 783 (13.2 GB/s) |
| `-mcs 256` (= 24 x 512 experts on the CPU, warm hot/cold profile) | 3.20 / 13.95 | **22.53**, `bench_chat` **22.98** | 44.6 | - |
| `-mcs 288`, cold (no expert profile yet) / warm | 2.48 / 13.05 | 20.88 / 22.65 | 48.2 / 44.4 | 360 / 179 |
| `-mcs 320` (profile of one earlier run) | 2.54 / 11.50 | 20.09 | 50.0 | 400 |
| `-mcs 288 -mcpa` (4 promote spares per layer, warm) | 2.95 / 12.82 | 21.57 | 46.4 | - |
| MTP `-mtp -ndt 1 -dds`, `-mcl 24` | 3.90 / 14.12 | `bench_chat` **25.78** (96.1%), harness 24.58 / 21.33 profiled | - | 1077 (m = 2) |
| MTP, `-mcs 256` (warm) | 3.81 / 13.95 | harness 29.32 (94.8%), `bench_chat` **31.52** (92.3%) | 67.7 | 234 (m = 2) |

- `-mcl 24` is the smallest whole-layer offload that fits (23 fails on a clean GPU). The autosplit is greedy (the 2060 fills
  first, then everything goes to the P100), so the margin is decided by where the first expert-carrying layer lands.
- `-mcs` at equal VRAM (`-mcs 256`: 48 x 256 = 24 x 512 experts on the CPU, +0.33 GiB GPU alloc) is **+16-22%** over
  `-mcl 24`. The gain comes from the hot/cold placement more than from the CPU/GPU overlap: the dynamic map moves the
  frequently routed experts to the GPU slices, so with a saved profile (`~/.cache/exllamav3/expert_stats/`, written at unload)
  the CPU job shrinks from 360 to 179 us per layer-token (s288 cold vs warm). The first load of a model is the cold case.
- The CPU expert kernel is compute-bound, not DRAM-bound: it scales linearly with physical cores (4 -> 8 threads: 1039 -> 562
  us/job) at ~2 GB/s of trellis per core, a third of the 40-44 GB/s the RAM delivers; SMT siblings make it slower. `-mct 8`
  (all physical cores, the host thread shares one) is the better default on this 8-core part (+3-8%, inside the noise band).
- `-mcpa`: with `-mcs 288` it fits (clean load) and gave no decode gain over warm `-mcs 288` (21.57 vs 22.65 t/s, within
  noise). Prefill, single loads each: +12% on the cold 4.4k chunk at 0, +2% / +5% on the 16k prompt / the 4.4k chunk at 16k,
  so only the first clears the ~5% spread.

**Decode iteration budget**, `-mcl 24`, no MTP, profiled window 53.3 ms/iter (unprofiled 51.9). Every part is serial: the
CPU worker runs while both GPUs wait (the wait is a stream memop, not a kernel), and the two GPUs never overlap.

| ms per token | |
| :--- | ---: |
| CPU routed experts, 24 layers x ~620 us (no GPU busy: 16.5 ms incl. host gaps) | 14.9 |
| P100 (layers 24-47, 20.3 ms busy): MoE coop kernels for 10 experts + shared (`coop_f` 196 us + `a` 64 + `b` 42 per layer) | 7.2 |
| -- EXL3 5 bpw GEMV (GDN qkv/z/out, attention q/k/v/o, shared expert) / lm_head (4 bpw) | 5.4 / 1.1 |
| -- gated residual (`gr_dots_kernel2` + `gr_finalize_kernel2`, 2 sites per layer, ~43 us each) | 4.2 |
| -- GatedDeltaNet recurrence + conv / attention (6 layers) / router + top-k / hyperconnection apply | 0.9 / 0.5 / 0.6 / 0.25 |
| 2060 (layers 0-23, 16.5 ms busy): tile GEMM / sliced MGEMM for the 5 bpw projections at m = 1 | 10.6 |
| -- gated residual / GDN / attention | 4.0 / 0.7 / 0.2 |
| n-gram table lookup + PLE (host 1.6 ms, mostly hidden; 0.45 ms on the 2060) | ~0.5 |

- The gated residual reads two fp16 tables per site, (320, 4 x 2560) down and the repacked up, 13 MB per site, 1.27 GB per
  token over 96 sites (also 1.2 GiB of VRAM). At ~43 us per kernel that is ~150 GB/s on both cards: a quarter of the P100's
  HBM2 and half the 2060's GDDR6. It is 8.2 ms per token, as much as the routed MoE on the P100.
- The 2060 runs the 5 bpw projections on the tile GEMM (`exl3_mgemm_kernel` 262 us for the 25 MB qkv+z slice pair, ~95 GB/s),
  because the sm_75 GEMV is gated to K <= 4 (`exl3_gemv_cfg`, `use_mgemm`); this model's attention/GDN weights are K = 5.
  Probe `EXL3_GEMV=2 EXL3_QKV_SLICE=0` (forces the staged K = 5 GEMV everywhere and unfuses qkv+z): 2060 GEMM/GEMV time
  10.57 -> 7.89 ms, 2060 busy 16.5 -> 14.0 ms per token (K = 5 GEMV at ~56 us for 15.6 MB, ~280 GB/s); the P100 gets slightly
  worse under mode 2 (6.48 -> 6.67 ms), so a real change must be sm_75-only. Output text differs from the default after the
  first sentence (different accumulation order); needs a perplexity/test check before it is claimed.
- The P100 MoE coop kernel moves 10 x 0.9 MB of 1.5 bpw trellis in 196 us (~47 GB/s): decode-compute-bound, as on Qwen3.6.
- With `-mcs 288` (cold) the split changes shape: the 2060 holds only ~6 layers (5.1 ms busy), the P100 42 (33.3 ms), and the
  CPU wait not hidden behind the GPU is 9.0 ms per token; same kernels, so the same per-kernel levers apply.
- **MTP** (the target mode), profiled windows of 30 iterations at ~1.95 tokens per iteration:
  - `-mcl 24`: 77.4 ms/iter. The CPU worker at m = 2 (up to 20 distinct experts per job) takes ~1077 us per layer, 25.8 ms
    per iteration (no GPU busy 29.5 ms, 38%); P100 30.0 ms (MoE coop 12.0, of which `coop_f` 340 us/layer; GEMV 9.4; gated
    residual 5.2), 2060 18.0 ms (tile GEMM 10.7, gated residual 4.5).
  - `-mcs 256`: 61.9 ms/iter. The CPU work is almost fully hidden (no GPU busy 4.9 ms, CPU job 234 us); the **P100 is the
    bottleneck at 47.3 ms busy**: MoE coop 20.9 ms (`coop_f` 337 us x 37 + `a` 89 + `b` 60 per layer), EXL3 5 bpw GEMV at
    m = 2 13.2 ms (~1.5 GB of projections at ~120 GB/s), gated residual 8.0 ms, attention 1.7, GDN 1.8. The 2060 holds only
    ~7 layers (9.7 ms busy, of which gated residual 1.7).

**Long context** (`-mcl 26 -cs 24576`, prompt = repository code, prefix reuse verified: the second job processed 4433 new
tokens): decode at 20.8k tokens 16.91 t/s against ~17.3 at short context in the same load. Profiled: QSA kernels P100
0.5 -> 1.24 ms per token (`_qsa_sparse_split_kernel` 139 us x 6), 2060 0.2 -> 2.1 ms (`_dsa_indexer_fewq_kernel` 189 us
x 6 + sparse split 145 us x 6); the eager QSA path's host time in attention grows 1.3 -> 13.4 ms per token but is almost
entirely hidden behind the GPU queue. The n-gram lookup goes 0.8 -> 4.7 ms host per token on novel text: rows missing from
the page cache are random reads on the USB SSD. Total profiled iteration 54.1 -> 60.6 ms (+12%). No correctness problem.

**Prefill** (default `-chunk_size 4096`, so the 4400-token prompt runs as a 4096 + ~330 chunk pair; `-mcl 26 -cs 24576`):

| | 4421 tokens at 0 | 16405 tokens at 0 | 4433 tokens at 16.4k |
| :--- | ---: | ---: | ---: |
| default | 196 t/s (22.5 s) | 220 t/s | 225 t/s (19.7 s) |
| `-mcpa` | 219 t/s (20.2 s) | 224 t/s | 237 t/s (18.7 s) |

**Load time** (~155-170 s; stack sampler on the main thread, mcl 24): ~85 s reading 17-18 GiB of weights (I/O bound,
~210 MB/s from the USB SSD; parent CPU 34 s over that span), then ~72 s in `model.warmup`, of which 35 s are n-gram table row
lookups from disk (`row_table.lookup`, random reads for the warmup chunk), ~13-27 s the CPU MoE prefill of the warmup chunk,
the rest GPU forwards. Triton compiles hit the disk cache (~1 s); the 30 s of `_configure` seen under cProfile in the first
load were profiler overhead plus a cold cache. `-ngr` (18.5 GiB table in RAM) cannot be combined with an 11 GiB expert arena
in 31 GB.

### Qwen3.8-Flash-Next: ranked candidates (Nov 5, Stage 0 estimates, nothing implemented)

Gains are per MTP iteration of the best configuration found (`-mcs 256 -mtp -ndt 1 -dds`, 61.9 ms profiled, ~1.95 tokens,
29.3 t/s harness, 31.5 `bench_chat`), where the P100 is the bottleneck; "risk" covers numerics and effort.

| # | Candidate | Estimated gain | Risk / effort |
| :--- | :--- | :--- | :--- |
| 1 | Run `-mcs 256` instead of `-mcl 24`, with a warm expert profile (ship or pre-seed it so the first load is warm). (`-mct 8` measured with `-mcl` only, within noise) | **measured with `bench_chat`**: MTP 25.78 -> 31.52 t/s (+22%), plain 19.44 -> 22.98 (+18%); harness-to-harness +19% / +16-22% | none: configuration only |
| 2 | Gated residual kernels at bandwidth: `gr_dots_kernel2` / `gr_finalize_kernel2` read 13 MB per site at ~150 GB/s | ~6 of 9.7 ms (P100 8.0 + 2060 1.7), ~10% | low-medium: two kernels, fp32 sum order |
| 3 | P100 MoE coop kernel on 1.5 bpw experts (decode-bound, ~47 GB/s; 20.9 ms per MTP iteration) | 25-30% faster = 5-6 ms, ~9% | high effort, medium risk |
| 4 | P100 EXL3 5 bpw GEMV at m = 2 (~120 GB/s, 13.2 ms) | 2-4 ms | medium-high: already tuned on the `pascal-gemv-k5` line |
| 5 | Gated-residual tables in 8 bit (or EXL3) | halves #2's bytes again; frees ~0.6 GiB VRAM for GPU experts (fewer CPU misses) | medium: accuracy of the stream mixing |
| 6 | sm_75 K = 5 GEMV + unfused qkv/z on the 2060 (relax the K <= 4 gate in `exl3_gemv_cfg` / `use_mgemm`, sm_75 only) | measured -2.6 ms at `-mcl 24` (24 layers on the 2060), ~0.8 ms under `-mcs 256` (~7 layers) | low: kernel exists; needs a sm_75 K = 5 test and PPL |
| 7 | CPU expert kernel (AVX2, compute-bound at ~2 GB/s per core vs 40 GB/s RAM) | -5 ms or more at `-mcl 24` / cold profile; under warm `-mcs` the CPU is hidden (4.9 ms no-GPU) | high effort, medium risk |
| 8 | Load: skip or shrink the n-gram row lookups in `warmup`; move the model off the USB SSD | -35 s of ~160 s load; the disk move roughly halves the 85 s weight read | low (warmup) / user decision (disk) |
| 9 | n-gram rows: keep the hot part of the table in RAM, prefetch a token ahead | ~0.5 ms short context, ~4 ms at long context on novel text | low-medium |
| 10 | QSA at long context: the 2060's `_dsa_indexer_fewq_kernel` takes 189 us/layer at 21k against ~37 us on the P100's eager-sized config (sparse split ~140 us on both), so the sm_75 indexer configuration looks wrong rather than slow; cheap probe: the eager tile size on sm_75 | ~1-2 ms at 21k tokens, nothing below ~2k | low-medium |
| - | Ruled out: `-mct 15` (SMT siblings slower), `-mcl` < 24 (does not fit), `-ngr` (RAM), `-mcs 288 -mcpa` (no gain over warm `-mcs 288`, fit marginal) | | |

### Qwen3.8-Flash-Next: gated residual decode kernels and warmup n-gram skip (Nov 5, candidates #2 and #8)

- `EXL3_GR_MIX3` (default on for cc < 8): new `gr_dots_kernel3` / `gr_finalize_kernel3` (`hc_gr_decode3.cuh`) for
  R <= 4 on the D 2560 / rank 320 shape. Old dots used 255 registers (2 blocks/SM, 2-3 waves) and re-read 4 stream rows
  from L2 per fn pair; new dots: one stream per block in shared memory, one fn row per warp (53-84 regs). New finalize:
  R templated, up loads and prologue dots loads issued before the first barrier, epilogue spread over the lanes.
  **Bit-identical** to the old pair (`tests/kernels/hyperconnections/test_gr_mix_decode.py`, 36 cases, both GPUs).
- Microbenchmark (13 MB per site, cold tables, ext flags): P100 read peak ~450-520 GB/s (not 240).

| per site, us | P100 R = 1 | P100 R = 2 | 2060 R = 1 | 2060 R = 2 |
| :--- | ---: | ---: | ---: | ---: |
| old pair (weighted / unweighted) | 51-55 / 66-70 | 54-58 / 70-75 | 73-86 / 96-100 | 78-80 / 100 |
| new pair | 34 | 39 | 50-54 | 54-55 |

- `bench_chat` ABAB (`-mcs 256 -mtp -ndt 1 -dds`, same restored expert profile each run): new 32.33 / 32.11 t/s, old
  (`EXL3_GR_MIX3=0`) 30.47 / 30.41 t/s: **+5.9%**, acceptance 96.1% in all four, generated text byte-identical (md5).
- Tried and dropped: one-launch ticketed dots+finalize (slower on the P100, 39 -> 45 us), finalize with one warp per
  (column, stream) (no gain).
- Warmup n-gram skip: warmup lookups (`tp_warmup`) decode zero rows instead of reading the table, and the PLE prefetch is
  skipped in warmup (warmup routing is randomized anyway, so the expert profile is unaffected). Warmup 40.2-40.6 s in the
  four runs above vs ~72 s in Stage 0 (different session, not an A/B); total load 152-163 s, dominated by the weight
  read from the USB SSD. Test: `test_warmup_skips_table_reads` (real table, disk mode).
- Open: in-model profiler of the new kernels (harness window was too short for MTP, `PROFNEW` must be > 2x `PROFFROM + 30`);
  a load-time A/B against the old warmup; transposed (fewer-shuffle) finalize reduction (~1 us/site est.); 8-bit
  gated-residual tables (#5). Pre-existing, unrelated: `test_embedding_table.py::test_quantize_embedding[2-*]` fails
  on the P100 (`quantize.cu:144` invalid argument).

### Qwen3.8-Flash-Next: draft length and MTP step profile (Nov 5, measured)

New best: **`-gs 4.0,15.0 -cs 8192 -cq 4 -mcs 256 -mtp -ndt 3 -sdh 16384`**, `bench_chat` **43.42 t/s** (88.6% of
drafts accepted). `-ndt 3` without `-sdh`: `bench_chat` 40.08 t/s (82.9%). Previous best `-ndt 1 -dds`: 32.2.

Method: one load per sweep (`tmp/ndt/ndt_body.py`): a fresh `Generator` per run (cold dds calibrator, as in `bench_chat`),
greedy, code prompt, 300 tokens, `time_generate`, P100 cooled to <= 62 C before each run, one throwaway run per draft width
first. All runs produced byte-identical text (one md5), as greedy verification should. Two loads, ABAB orders.

| config | t/s (runs) | tokens / step | ms / step | conditional acceptance, positions 1 / 2 / 3 / 4 |
| :--- | :--- | ---: | ---: | :--- |
| `-ndt 1 -dds` | 35.02 (4), 34.91 (2) | 1.96 | 56.4 | 96.1 |
| `-ndt 1` | 34.97 (2) | 1.96 | 56.4 | 96.1 |
| `-ndt 2 -dds` | 36.34 (4; 37.3 without one 34.1 outlier) | 2.80 | 75.4 | 97.2 / 85.6 |
| `-ndt 2` | 37.13 (2) | 2.80 | 75.6 | 97.2 / 85.6 |
| `-ndt 3 -dds` | 40.10 (2) | 3.45 | 86.3 | 95.4 / 84.3 / 89.6 |
| `-ndt 3` | 40.63 (2), 40.23 (3) | 3.49 | 86.6 | 95.3-97.7 / 82-85 / 88.5 |
| `-ndt 3 -sdh 16384` | 43.80 (3) | 3.66 | 84.0 | 97.5 / 90.4 / 92.2 (warm token profile) |
| `-ndt 4`, `-ndt 4 -dds` | 39.90 (2), 39.95 (2) | 4.05 | 102 | 97.3 / 83.3 / 85.0 / 84.3 |

- Unlike Qwen3.6, extra draft positions pay: the MTP head keeps ~85-90% conditional acceptance at positions 2-3, and the
  step cost grows sub-linearly (m = 2 -> 4: 56 -> 87 ms for 1.96 -> 3.49 tokens, 28.8 -> 24.8 ms per token). `-ndt 4`
  costs +16 ms per step for +0.56 tokens (no gain; m = 5 also leaves the `GR_MIX3` R <= 4 path). `-dds` never helps here
  (it only truncates; acceptance is high), so drop it.
- `-sdh 16384` at ndt 3: +8.9% (harness, 40.2 -> 43.8), `bench_chat` 40.08 -> 43.42 (+8.3%). It removes the draft head's
  three full-vocab m = 1 GEMVs (1.1 ms each on the P100) and, oddly, raised acceptance (82.9 -> 88.6% in `bench_chat`). Cold
  start: the first `-sdh` run of the load (token profile from earlier Qwen3.6 runs) fell to 77.7% / 33.2 t/s; afterwards
  normal. `~/.cache/exllamav3/token_stats` now holds this model's profile.
- First generation after a load is slower: run 0 at ndt 1 was 61.9 ms/step vs 56.4 later (32.0 vs 35.0 t/s). That is why
  single-turn `bench_chat` at ndt 1 reads ~32 against 35 in the warm harness; at ndt 3 the two agree (40.1 / 40.2).

**MTP step profile** (torch.profiler, 80 decode iterations after 40 warmup iterations, 480-token job; profiled wall is ~8 ms per
step above the unprofiled one, mostly host-side). ms per step, real kernels only (spans of the CPU-expert stream wait excluded):

| | ndt 1 -dds (m = 2) | ndt 2 -dds (m = 3) | ndt 3 (m = 4) |
| :--- | ---: | ---: | ---: |
| profiled wall (unprofiled) | 57.8 (56.4) | 77.5 (75.4) | 94.9 (86.6) |
| P100 busy / 2060 busy / both at once | 43.3 / 8.6 / 0.0 | 59.1 / 11.1 / 1.3 | 66.0 / 13.3 / 0.05 |
| no GPU kernel running | 6.0 | 8.7 | 15.6 (~7 unprofiled) |
| P100 MoE coop (`coop_f<1>` per layer, x37) | 19.9 (302 us) | 24.8 (389 us) | 28.7 (468 us) |
| P100 EXL3 GEMV/GEMM, incl. lm_heads | 13.2 | 21.5 | 22.4 |
| -- 5 bpw `exl3_gemv_kernel<5,true,2,MMODE,..>` x60 / x41 | 74 / 83 us (MMODE 2) | 125 / 113 us (MMODE 1) | 126 / 113 us (MMODE 1) |
| -- main lm_head (4 bpw) / draft lm_head (m = 1, 1.1 ms each) | 1.3 / 1.1 | 2.4 / 2.2 | 2.4 / 3.3 |
| P100 gated residual (`gr_*_kernel3`) | 5.0 | 6.7 | 8.3 |
| P100 GDN / attention / router + hc | 1.8 / 1.7 / 0.9 | 2.2 / 2.1 / 0.9 | 2.5 / 2.1 / 0.9 |
| 2060 MoE / GEMV-GEMM / gated residual | 2.9 / 3.8 / 1.1 | 4.0 / 4.3 / 1.5 | 5.0 / 4.9 / 2.0 |

- The P100 5 bpw GEMV dispatch (`exl3_gemv.cu`, MMODE template parameter): m = 2 uses the direct two-row dot product
  (MMODE 2, 74 us), m = 3..8 the MMA path (MMODE 1, 125 us at m = 3 and m = 4); the four-row direct path (MMODE 3)
  exists but was measured slower than MMODE 1 at m = 3..4 earlier. So GEMV jumps +8.3 ms from m = 2 to 3, then +0.9 ms to
  m = 4; the main lm_head does the same (1.3 -> 2.4 ms). Two m = 2 calls (2 x 74 us) would be slower than MMODE 1.
- The shared expert (5 bpw) runs as `coop_a<5>` + `coop_b<5>` at m = 2 (88 + 60 us x41, 6.1 ms) and as one
  `coop_f<5, 2, false>` at m >= 3 (182 us x41, 7.5 ms): normal scaling with m, not a fallback.
- The two GPUs never overlap (0.05 ms): the 2060's 13.3 ms is pure serial time on top of the P100's 66.
- CPU expert worker (`EXL3_MOE_CPU_PROF`, separate load, per job, all phases): 207 us at ndt 1, 252 us at ndt 3, 259 us at
  ndt 4; ~13 jobs per step at ndt 1, ~22 at ndt 3, so ~2.8 / ~5.5 ms busy per step. Unprofiled, the no-GPU time at ndt 3
  is ~7 ms (profiled 15.6 minus the ~8 ms profiler inflation), so most of the CPU work now shows as GPU idle time.
  Runtime toggling with `exl3_moe_cpu_set_prof` does not work (read at worker startup), so set the env var before loading.

**Re-ranked next steps** (base: `-ndt 3 -sdh 16384`, ~84 ms per step, 3.66 tokens, 43.4 t/s `bench_chat`; target 40-55 t/s,
55 t/s = 66.5 ms per step at this acceptance):

| # | Candidate | Est. gain, ms per step | Risk / effort |
| :--- | :--- | ---: | :--- |
| 1 | P100 MoE coop `coop_f<1>` at m = 4 (468 us x37 = 17.3 ms, decode-compute-bound; union of ~2x the experts), plus the shared expert `coop_f<5>` (7.5 ms) | 4-6 for 25-35% faster (**done in part**: -4.5 ms, see "P100 MoE at m = 4") | high effort, medium risk |
| 2 | P100 5 bpw GEMV at m = 3-4: a four-row direct product (MMODE 3 rework) near the m = 2 cost per byte, ~90 us in place of 126 / 113; the main lm_head too | 2-4 (+ ~0.5 lm_head) | medium-high: kernel work, MMODE 3 measured slower before |
| 3 | Gated residual at m = 4 (8.3 + 2.0 ms): 8-bit tables (bytes halve; R = 4 still fits `GR_MIX3`) | 3-4 | medium (mixing accuracy) |
| 4 | CPU expert time at m = 4 (~5.5 ms busy, ~7 ms no-GPU): `-mct 8`, more GPU expert slots (VRAM freed by #3), or start the CPU job earlier | 1-3 | low (config) to medium |
| 5 | 2060 K = 5 GEMV (old #6; 2060 GEMV/GEMM 4.9 ms at m = 4) | ~1 | low |
| 6 | `-sdh` cold start: ship or pre-seed the token profile (first run of a load at 77.7% acceptance) | first-run only | low |
| - | Measured, no gain: `-ndt 4` (39.9 t/s vs 40.2 at ndt 3), `-dds` at any ndt. `-ndt 4 -sdh` not tested | | |

#2 + #3 + #4 (~8 ms) would give ~76 ms per step, ~48 t/s; 55 t/s (66.5 ms) also needs #1.

### Qwen3.8-Flash-Next: P100 GEMV at m = 3..4 (Nov 5, re-ranked #2, done)

- MMODE 4 (`exl3_gemv_kernel.cuh`, default on Pascal at m = 3..4, `EXL3_GEMV_M34=0` restores MMODE 1): the same arithmetic
  as the emulated MMA of MMODE 1 (same HFMA2 chain, pair sum, fp16 run sum, fold cadence), with one output (column
  lane / 4, row lane % 4) per lane and n8 fragment: 8 shuffles + 8 HFMA2 per fragment instead of 16 + 32 (MMODE 1 also
  computed the 8 zero rows g + 8). **Bit-identical** (`tests/kernels/gemm/test_exl3_gemv_m34.py`, 15 cases).
- Why MMODE 1 was slow: besides the wasted work, the wide config (n > 8192, 256 threads) uses 172 registers on sm_60
  (one block, 8 warps per SM); MMODE 4 uses 128 (two blocks). Narrow: 128 -> 108. MMODE 3 (four-row direct, not
  instantiated) measures 120 registers in the narrow config, so its earlier loss was not occupancy there.
- Microbenchmark, P100, us, m = 4 old / new (m = 2 for reference): 5 bpw 2560 x 4096 71 / 63 (56); 4096 x 2560 67 / 57
  (50); 2560 x 8192 103 / 94 (80); 2560 x 12288 185 / 109 (102); 4 bpw 2560 x 16384 187 / 121 (111). m = 3 alike.
- `bench_chat` best config, 8 loads (ABAB + BABA, expert and token profiles restored before each load so all runs
  draft the same; 216/252 accepted): old 38.54 / 41.01 / 40.81 / 38.31, new 44.06 / 39.89 / 43.00 / 42.49 t/s, means
  **39.67 -> 42.36 (+6.8%)**, ~90 -> ~84 ms per step; generated text byte-identical in all 8. Without the profile
  restore, runs differ in text and acceptance from load to load (profiles are rewritten at unload), so A/B across loads
  must restore them.

### Qwen3.8-Flash-Next: 8-bit gated-residual tables (Nov 5, re-ranked #3, ruled out)

Branch `flash-gr-q8` (not merged): `EXL3_GR_Q8=1` runs `gr_*_kernel3` on int8 fn / up tables (absmax per row and stream / per
up row, fp32 scales), same VRAM (the fp16 repacked up table is dropped, R 5..8 go to cuBLAS); plus a one-wave dots grid with
a next-row prefetch (bit-identical, kept on that branch only).
- Microbenchmark per site, P100, R = 1 / 4: fp16 35.5 / 56 us, q8 25.8 / 44 us. At R = 4 (the ndt 3 verify) both
  kernels are latency-bound, not byte-bound: dots stages 40 KB of streams (one 8-warp block per SM; 16-warp blocks were
  slower), finalize uses 168 registers (128 with int8). 
- Accuracy (one load, 4-token decode chunks, 302 positions): KL(fp16 || q8) mean 7.5e-4, max 0.029, top-1 agreement
  99.34%; greedy text diverges at ~token 240 of 300 on a near-tie.
- `bench_chat` best config, 8 loads ABAB + BABA, profiles restored: fp16 42.48 / 38.52 / 39.80 / 41.54, q8 43.73 / 42.28 /
  37.63 / 39.39 t/s, means 40.59 vs 40.76: **no gain** inside the ~5% load-to-load spread. Not worth the changed text.

### Qwen3.8-Flash-Next: P100 MoE at m = 4 (Nov 5, re-ranked #1, branch `flash-moe-m4`)

Profile (torch.profiler, 80 steps of the best config, `tmp/moem4`), P100 MoE per step: routed `coop_f<h1>` 18.87 ms
(510 us x37), shared expert `coop_f<5>` 7.47 (182 us x41, separate launch: the dual kernel had no half-rate
instance), 2 bpw routed layers `coop_f<2>` 2.04 (x4), CPU-split collect / issue 0.73 / 0.57, rotation 0.67, top-k
0.46, gaps between MoE kernels 0.88: **30.8 ms**. Routing at m = 4 (hook on `run_bszN`, 2378 calls): 36.8 active GPU
slots and 21.7 distinct GPU experts per call; runs of 1 / 2 / 3 / 4 rows = 59 / 22 / 11 / 9% of runs.
Bandwidth floor per layer: 21.7 x 0.92 MB (1.5 bpw, 3 x 2560 x 640) + 3.1 MB shared (5 bpw) = 23 MB, ~50 us at
450-520 GB/s, against ~690 us measured: far from the 15% stop rule, but the kernel is issue-bound (decode + HFMA2),
not byte-bound, so the floor is not reachable; m = 1 costs ~17 us per expert, and m = 4 is ~21 us per distinct expert.
- `EXL3_MOE_COOP_M34` (default on): multi-row runs (2..4 rows) on sm_60 take the one-output-per-lane form of the
  emulated MMA (as GEMV MMODE 4): 8 shuffles + 8 HFMA2 per fragment instead of 16 + 32 plus the A gather. Kept in
  `ch` / `acc0` (low half / `.x`), so 128 registers stay with small spills (stack 8-16 B).
- `EXL3_MOE_COOP_DUAL_H` (default on): dual persistent kernel (shared + routed in one queue) for half-rate routed
  experts (h1 / h2 with 4..6 bpw shared, `exl3_moe_coop_inst_dh.cu`): the 4-row shared expert had 20 real blocks
  per stage on 56 SMs. Both **bit-identical** (`tests/kernels/moe/test_moe_coop_m34.py`, 35 cases; dual cases added
  to `test_moe_coop_fused.py`; moe tests 353 + 2 skipped on the P100, 355 on the 2060).
- Microbenchmark, Flash-Next shapes, routed h1 + shared 5 bpw pair, us: P100 m = 4 706 -> 599, m = 2 484 -> 383,
  m = 1 270 -> 222; 2060 m = 4 475 -> 439. In the model: MoE kernels **30.8 -> 26.3 ms per step** (`d_kernel<1, 5>`
  580 us x37 in place of 510 + 182), same greedy text.
- `bench_chat` best config, 8 loads ABAB + BABA, profiles restored before each (A = `EXL3_MOE_COOP_M34=0
  EXL3_MOE_COOP_DUAL_H=0`, same `.so`): A 36.64 / 37.25 / 42.35 / 38.48, B 37.46 / 41.01 / 40.23 / 47.40 t/s, means
  **38.68 -> 41.53 (+7.4%)**, ~91 -> ~85 ms per step (wide spread this session, P100 at 67-71 C at turn start).
  Text byte-identical in 7 of 8; the odd one is an A run (diverges at line 35, the known run-to-run drift), so not
  the new kernels.
- Left: the 4 layers with 2 bpw routed experts still launch the shared expert separately (`d_r2` covers S = 2..4;
  ~0.2 ms); the dual kernels spill 24-64 B at the 128-register cap; per distinct expert the 1.5 bpw staged decode
  (`dq_dispatch`, HALF) is now the cost.

### Qwen3.8-Flash-Next: dual-kernel empty-row race fixed; full-reply ndt sweep (Nov 5, branch `flash-fix-ndt`)

- **Race (was in Known issues), real and fixed.** A token row with no active routed slot (all picks CPU-side under
  `-mcs`, or all weights zero) is written by `write_empty_row_chunk` (gated shared term only). In the dual kernel that
  ran in the rotation pre-kernel (bsz > 1) or a routed A item (bsz 1), before the same launch's shared down items:
  stale `sh_out`. `tests/kernels/moe/test_moe_coop_dual_empty.py` (15 cases, NaN-filled `sh_buf`) failed 15/15 on
  the old build. Fix: those writes are skipped when `p.f_sh_wait` is set and the last shared down item (atomic on
  `f_sh_done`) writes all empty rows. Bit-identical to the separate launches; moe tests 368 + 2 skipped (P100),
  370 (2060). Cost: none measurable, `exl3_moe_coop_pair` microbench (h1 + 5 bpw, ABAB) 689 -> 681 us at m = 4,
  223 -> 218 at m = 1.
- Greedy text is not reproducible across loads at full reply length (same config, different md5 and length):
  identical-text controls are not available here; use >= 3 loads.
- Full replies (`bench_chat -max_new 3000`, all ran to EOS; `EXL3_AUTOSPLIT_MARGIN_MB=64` needed to load since desktop
  apps now hold ~570 MiB on the 2060; profiles restored before each load; `-sdh 16384`, no `-dds`), 3 loads each:

| prompt | config | t/s | tokens / step | acceptance | reply tokens |
| :--- | :--- | ---: | ---: | ---: | :--- |
| code (quicksort go) | `-ndt 3` | 31.67 (32.2 / 30.4 / 32.5) | 3.32 | 77.2% | 898-942 |
| code | `-ndt 2` | 30.15 (30.5 / 30.4 / 29.6) | 2.70 | 84.7% | 897-905 |
| prose (Roman Republic) | `-ndt 3` | 17.48 (16.8 / 17.9 / 17.7) | 1.88 | 29.3% | 1373-1755 |
| prose | `-ndt 2` | 19.63 (19.8 / 19.6 / 19.5) | 1.81 | 40.4% | 1509-1900 |

  Prose drafts far worse than code (29-40% vs 77-85%; sparse-head miss rate 0%, so not the `-sdh` subset). Code
  favours ndt 3 (+5%), prose ndt 2 (+12%). Not measured (stopped on request): `-ndt 1`, `-dds` at any ndt,
  `-sdh 0 / 8192 / 32768`, `-dc`. With 1.8 tokens per step on prose, `-ndt 1` is the likely prose winner.
- `bench_chat` now prints tokens per step ((new - 1) / (new - 1 - accepted)) and the sparse-head miss rate / mode.

## Target (set Nov 3 by the user)

**100 t/s** with MTP (`-mtp -ndt 1 -dds`, `-gs 4.0,15.0 -cq 4`) on Qwen3.6-35B-A3B and on Ornith-1.5-35B-A3B, measured with
`examples/bench_chat.py` on the code prompt, P100 cooled to <= 62 C. Starting point (Nov 3): Qwen3.6 69.8 t/s greedy (94.2% accepted),
~67-69 with the chat.py sampler; Ornith 63.2. At ~1.97 tokens per iteration this means a 19.7 ms MTP iteration against ~27.5 ms
today. Deeper drafting (`-ndt 2/3`) is not a lever: acceptance of the extra tokens drops (82% -> 70% -> 66-70%) and the MoE verify
cost grows with the distinct experts read, so only kernel and scheduling work counts.

### Stage 0: per-iteration budget of the MTP iteration (Nov 3, measured)

torch profiler around `Generator.iterate` in the real decode (chat template, "Implement a quicksort in go", greedy, 60
iterations from iteration 150, `-gs 4.0,15.0 -cs 32768 -cq 4 -mtp -ndt 1 -dds`, P100 1328 MHz, 57-64 C). Unprofiled
900-token runs: Qwen3.6 27.70 ms/iter, 1.927 tok/iter (69.6 t/s, 92.7% accepted); Ornith 28.01 ms, 1.767 tok/iter (63.1 t/s,
76.8%). Every step of the iteration is serial: the 2060 and the P100 run concurrently for 0.01 ms per iteration.

| ms per iteration (kernel times) | Qwen3.6 | Ornith |
| :--- | ---: | ---: |
| draft: MTP layer on the 2060 (m = 1) | 0.48 | 0.47 |
| draft: lm_head on the P100 (m = 1) | 1.10 (5 bpw GEMV) | 1.74 (fp16) |
| verify, 2060 layers (m = 2) | 5.46 (10 layers) | 5.47 (9 layers) |
| -- of which MoE / projections / attention | 2.61 / 2.10 (tile GEMM, K = 5) / 0.31 | 2.11 / ~2.5 (fp16) / 0.39 |
| verify, P100 layers + lm_head (m = 2) | 19.29 | 17.19 (+1.1 unattributed) |
| -- MoE persistent kernel | 8.02 (30 x 267 us) | 6.04 (31 x 195 us) |
| -- rotation pre-kernels / router GEMV / top-k | 0.42 / 0.54 / 0.32 | 0.24 / 0.35 / 0.28 |
| -- projections (m = 2) | 5.78 (EXL3 5 bpw GEMV) | ~6.7 (fp16 hgemv) |
| -- lm_head m = 2 | 1.27 | 1.75 |
| -- attention (split 1.02, combine 0.13, rope, cache quant) | 1.34 | 1.30 |
| -- GatedDeltaNet (recurrence 0.66, conv 0.28, norms, ...) / RMS norms | 1.33 / 0.28 | 1.1 / 0.25 |
| MTP prefill of the accepted token (2060, attention part only) | 0.30 | 0.20 |
| no GPU busy (profiled / unprofiled estimate) | 1.83 / ~0.95 | 2.38 / ~1.2 |

Host side: the draft step ends with a blocking `draft_conf.cpu()` (confidence calibration of `-dds`), so the verify
launch waits for the draft head; after verify one `cudaDeviceSynchronize`, the accept walk and the MTP prefill launch
(~0.4 ms without GPU work); 620 sub-5 us gaps per iteration in graph replay (~0.4 ms profiled). Ornith launches ~900
kernels per iteration outside graphs (0.57 ms of small gaps in its verify).

Measured gates:
- All Qwen3.6 layers on the P100 (`-gs 0,15.9 -cs 8192`, fits): 28.12 ms/iter vs 27.70 split; verify GPU time 25.07 vs
  24.75 ms. The 2060 is as fast per layer as the P100 at m = 2 (~550-600 us), the hand-off itself costs ~nothing. **Ruled out.**
- Cross-GPU hand-off: no peer access between the two cards (either direction); a P100 -> 2060 -> P100 round trip of
  2 x 2048 fp16 with two tiny kernels costs 59 us, an event ping-pong without data 50 us. An expert split across the GPUs
  would pay >= 2 ms per m = 2 pass (40 layers) against an estimated <= 3 ms gross gain, and the 2060 cannot hold
  ~40% of the experts. **Ruled out.**
- Decode attention splits sized from the live length instead of the 16-page-padded block table (experiment: no
  padding): split kernel P100 127 -> 96 us, 2060 97 -> 36 us; Qwen3.6 27.65 -> 27.25 ms/iter (+1.0%). The P100 cost is
  then ~96 us for a single 32-token tile per program (255 registers): the kernel itself is the problem on sm_60.

### Stage 1: first items (Nov 3, measured; switches off by default)

| Switch | What | In-model kernel effect (P100 / 2060, profiler) | ms/iter, interleaved (unprofiled loop) |
| :--- | :--- | :--- | :--- |
| `EXL3_ATTN_LIVE_SPLIT=1` | decode attention split over the live length (device-side), 16-key tiles on sm_60 / sm_75 | split kernel 127 -> 68 us (x8) / 97 -> 27 us (x4) | Qwen3.6 28.02 -> 27.31; Ornith 28.12 -> 27.44 |
| `EXL3_MOE_ROT_PAIR=1` | shared + routed rotation pre-kernels as one launch, bit-identical | 7.9 + 6.0 -> 9.8 us per layer | Qwen3.6 28.02 -> 27.87; Ornith n/a (fp16 shared expert) |
| `EXL3_MTP_DEFER_CONF=1` | draft confidence read after verification at window 1 | - | Qwen3.6 28.02 -> 27.96; Ornith 28.12 -> 28.07 (noise level) |
| `EXL3_ROUTER_FUSED=1` | router hgemv + top-k in one launch, bit-identical | 18.1 + 10.6 -> 29.1 us per layer: no gain | Qwen3.6 +0.3 / +0.25, Ornith +0.4 / +0.66: **leave off** |

**Nov 3: `EXL3_ATTN_LIVE_SPLIT`, `EXL3_MOE_ROT_PAIR` and `EXL3_MTP_DEFER_CONF` are now on by default** (set to `0` to disable; verified without any env var: Qwen3.6 70.3 -> 73.0 / 73.7 t/s). `EXL3_ROUTER_FUSED` stays off.

Attention numerics change (split boundaries), so the generated tokens differ slightly: Qwen3.6 acceptance 92.7 -> 94.0% in the
loop, Ornith 76.8 -> 80.0% (text-dependent, not a claimed gain). All four on (round 2, ABAB): Qwen3.6 27.20 / 27.63 -> 26.38 / 26.81
ms/iter (-0.82 both pairs), Ornith 27.99 / 27.99 -> 27.73 / 27.80 (the router switch costs Ornith what attention gains).

End to end with `examples/bench_chat.py -mtp -ndt 1 -dds -gs 4.0,15.0 -cs 32768 -cq 4 -prompts "Implement a quicksort in go"
-max_new 900`, greedy, ABAB, P100 66-71 C at 1328 MHz (cooled to 62 C before each load), A = switches off, BEST =
`EXL3_ATTN_LIVE_SPLIT=1 EXL3_MOE_ROT_PAIR=1 EXL3_MTP_DEFER_CONF=1`:

| t/s | A | BEST | BEST + `-sdh 16384` |
| :--- | ---: | ---: | ---: |
| Qwen3.6-35B-A3B | 70.31 / 70.31 (94.2%) | 72.96 / 72.74 (94.8%), x1.036 | 73.33 (88.7%) |
| Ornith-1.5-35B-A3B | 64.03 / 64.42 (80.1%) | 65.16 / 65.90 (80.0%), x1.020 | 69.47 (79.5%) |

Sparse draft head (`-sdh`, existing feature; `bench_chat.py` did not pass it to the generator before this stage, so the
earlier "-sdh changes nothing" was not a measurement of it). Unprofiled loop, switches off: `-sdh 16384` Qwen3.6 27.70 ->
26.58 ms/iter (92.3% accepted, 72.3 t/s), Ornith 28.00 -> 26.34 (77.5%, 67.4 t/s); the draft head goes from 1.1 / 1.7 ms to
~0.02 ms. Cold start is a problem: the first Qwen3.6 run with an empty token profile (`-sdh 32768`) drafted from a
23-column subset and fell to 61.9% acceptance, 60.3 t/s; the profile persists in `~/.cache/exllamav3/token_stats` (one
file, shared by both models' tokenizer, written by these runs), after which acceptance is normal. In bench_chat the
Qwen3.6 acceptance with `-sdh` is 88.7% vs 94.8%, so the net gain there is small (+0.5%); Ornith gains +5.4%.

Where this leaves the target: Qwen3.6 ~26.4-26.8 ms/iter against 19.7 needed; Ornith ~27.7. Left from the Stage 0 list: MoE
m = 2 (item 3, the largest), the P100 attention split kernel still 68 us per call in the model (30 us in the microbenchmark;
a CUDA kernel for sm_60 could reach ~15), rotation inside the persistent queue (now ~10 us per layer), sm_75 GEMV for K = 5.

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

- (Nov 3) Router GEMV + top-k fused into one launch (`EXL3_ROUTER_FUSED`): same kernel time in the model (29.1 vs 18.1 + 10.6
  us per layer at m = 2), end to end +0.25..0.66 ms/iter; the top-k latency (K serial warp-argmax rounds) stays on the critical
  path. A 256-wide bitonic sort in the same selection order in its place: 27 vs 18 us for the fused launch, slower.
- (Nov 3) Deferring the `-dds` draft-confidence readback (`EXL3_MTP_DEFER_CONF`): within noise, because the drafted ids must
  reach the host anyway before verification (CPU embedding table).
- (Nov 3) All layers on the P100 and an expert split across the two GPUs (see Stage 0).
- (Nov 5) "chat.py drafts worse than bench_chat.py" on Qwen3.8-Flash-Next (`-mtp -ndt 3 -sdh 16384`, quicksort in go):
  a length confound, not a bug in the chat path. bench_chat `-max_new 300` measures only the code part of the reply;
  chat.py runs to EOS (`-maxr` capped to 4096 by `-cs 8192`, ~900 tokens) and the prose tail drafts worse. Profiles
  (`expert_stats`, `token_stats`) restored before every run, matched lengths:
  300 tokens: bench 214/258 (82.9%) 40.4 t/s, chat.py `-prompt` 213/261 (81.6%) 40.2 / 35.3 t/s, chat.py interactive
  in a pty (Rich streaming, token healing off, `stop_on_loop`) 214/258 (82.95%) 37.2 t/s;
  to EOS: bench 626/846 (74.0%) 34.3 t/s, chat.py `-prompt` 589/774 (76.1%) 35.6 t/s, interactive 625/861 (72.6%)
  31.6 t/s. Job options and the generator setup of chat.py do not change acceptance; text diverges after ~35 lines
  between runs (verification batch width, CPU MoE), so +-3 points is noise. The interactive path is ~3 t/s slower at
  the same acceptance: Rich Markdown re-rendering on the host thread (open: lower `-ups` or `-basic` to check).
  Compare with `bench_chat.py -max_new 4096` (runs to EOS) against chat.py.

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

- `tests/kernels/routing/test_routing_gemm_det.py`: 25 failures on both the P100 and the RTX 2060 (the int8 deterministic
  router GEMM needs sm_80; the test skips only builds without it). Untouched code. Stage 1 check (Nov 3): gemm + moe 672
  passed / 49 skipped on the P100, 653 / 68 on the RTX 2060, 0 failed; the rest of `tests/kernels/routing` and the new
  live-split test pass on both.

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

## Candidates towards 100 t/s (Nov 3, estimates from the Stage 0 budget, not measured unless stated)

| # | Item | Qwen3.6 est. ms/iter | Ornith est. | Confidence |
| :--- | :--- | ---: | ---: | :--- |
| 1 | Decode attention: device-side split span from `cache_seqlens` (0.40 measured) + sm_60 per-program cost (~96 us -> ~20) | 1.0-1.3 (**measured 0.71** with 16-key tiles; a CUDA kernel may add ~0.4) | ~1.1 (**measured 0.68**) | done in part |
| 2 | Router GEMV + top-k (+ rotation) in one kernel or in the persistent MoE queue (P100 42.6 us/layer + 3 launches -> ~10) | 0.9-1.2 (**measured: rotation pair 0.15, router fusion none**) | ~0.6 (**none**) | top-k latency is the floor |
| 3 | MoE m = 2 efficiency (~200 G weights/s vs 400-460 for the GEMV; latency-bound down stage) | 1.5-2.5 | 1.2-1.8 | low-medium, large effort |
| 4 | Sparse draft head (`-sdh`, exists) for the m = 1 draft head | ~0.9 (**measured 1.1 ms/iter**, but bench_chat acceptance 94.8 -> 88.7%: +0.5% t/s) | ~1.5 (**measured 1.7**, +5.4% t/s) | needs a warm token profile (cold start: 61.9% accepted) |
| 5 | Host: no `draft_conf` sync at window 1 (**measured: noise**, the ids readback stays); MTP prefill + next draft as one m = 2 pass (not done) | 0.5-0.8 | 0.5-0.8 | low |
| 6 | sm_75 GEMV for K = 5 (2060 projections run the tile GEMM, 2.1 ms) | ~0.6 | - | medium |
| 7 | P100 m = 2 EXL3 GEMV (issue-bound, 5.8 ms) | 0.6-0.9 | - | low |
| 8 | Ornith: graphs for the fp16 layers; hgemv small shapes | - | 1.0-1.5 | medium |

Sum for Qwen3.6: optimistic ~8 ms (27.7 -> ~19.7), midpoints ~6 ms (-> ~21.7 ms, ~89 t/s at 1.93 tok/iter, ~85 without
`-sdh`). 19.7 ms needs every item at its optimistic end, including the MoE rewrite. Ornith needs 17.7 ms at 1.77 tok/iter;
the list reaches ~21 ms (~84 t/s), so 100 t/s on Ornith is not reachable with these items.

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
