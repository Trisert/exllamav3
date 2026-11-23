# LFM3-1.7B 8.0bpw decode megakernel: feasibility (2026-11-23)

Starting point: lfm8-opt @ 971fceac (PR #4).
- P100 decode: 144.3 / 140.2 / 129.5 t/s at depth 512 / 8k / 32k, 6.93 ms per token at depth 512.
- RTX 2060 decode: 133.0 / 124.6 / 103.6 t/s, 7.52 ms per token at depth 512.
- 243 kernels per token, all inside one CUDA graph.

## M0: what a launch boundary costs (doc/lfm3_mega/boundary_probe.cu)

The probe runs 243 dependent full-grid steps per "token", at 256 threads per CTA. It compares three structures:
- a CUDA graph of 243 kernels;
- one persistent kernel with a sense-reversing atomic barrier between steps;
- one persistent kernel with a cooperative-groups `grid.sync()` between steps.

Each configuration is compared with the same work done with no boundary at all. Results are µs per boundary.

| card | CTAs | work (FMAs per step) | graph | atomic barrier | grid.sync |
|---|---:|---:|---:|---:|---:|
| P100 | 56 | 0 | 1.37 | 1.91 | 1.40 |
| P100 | 112 | 0 | 1.35 | 2.31 | 1.51 |
| P100 | 56 | 64 | 1.33 | 1.93 | 1.46 |
| P100 | 112 | 64 | 1.43 | 2.43 | 1.64 |
| RTX 2060 | 30 | 0 | 1.40 | 1.44 | 0.79 |
| RTX 2060 | 60 | 0 | 1.01 | 1.14 | 0.85 |
| RTX 2060 | 30 | 64 | 1.00 | 1.10 | 0.84 |
| RTX 2060 | 60 | 64 | 1.04 | 1.13 | 0.88 |

Findings:
- **The naive megakernel (one global barrier per op) is REJECTED.**
  - On the P100 a global barrier costs as much as a graph boundary or more, so it gains nothing.
  - On the 2060 it saves at most ~0.2-0.6 µs per boundary, about 1-2 % of the token.
- **Upper bound for all inter-kernel bubbles:** 243 boundaries × the measured cost.
  - P100: ~0.33 ms, 4.7 % of the token.
  - 2060: ~0.25-0.34 ms, 3-4.5 %.
- **The profilers cannot measure these bubbles.** Both torch.profiler/CUPTI and nsys `--cuda-graph-trace=node` inflate the kernels themselves.
  - nsys run: 122 t/s.
  - Kernel sum: 7.85 ms per token, against a 6.93 ms unprofiled wall.
  - So "wall − kernel sum" comes out negative.
  - Tools: `doc/lfm3_mega/nsys_decode.sh` (standalone nsys at /opt/cuda/nsight_systems; the integration copy is broken with CUDA 12.9) and `doc/lfm3_mega/gap_stats.py`.

## M1: where a dataflow megakernel could win (P100, depth 512, per token)

The nsys trace gives the call counts. The per-call phases come from the round-7 GEMV phase probe (doc/lfm8_opt/PLAN.md). Both are profiler-inflated by ~13 %.

| source | arithmetic | ms per token |
|---|---|---:|
| inter-kernel bubbles | 243 × ~1.35 µs | ~0.33 |
| GEMV input Hadamard + grid.sync | ~2.3 µs × 97 GEMV calls | ~0.22 |
| balanced GEMVs: average warp waiting for the slowest | ~8 µs × 61 large calls (gate/up, down, conv in) | ~0.49 |
| narrow 2048 × 2048 GEMVs waiting at the second grid.sync | ~9.7 µs × 24 | ~0.23 |
| latency-bound small kernels | norms 0.27, attention combine 0.14, rope 0.13, hgemv gate 0.14, short conv 0.11, quant cache 0.08, mul_sigmoid 0.04, copy2d 0.03 (partly overlappable at best) | ~0.9 |

- **Total ceiling:** ~1.3 ms (~18 %) if every idle cycle were filled. That counts bubbles, GEMV fixed phases and tails, and excludes the small kernels.
- **Realistic capture:** about half, so **~5-9 % on the P100**. The old estimate of +10-15 % was too high.
- **What a megakernel adds that nothing else can on the P100:**
  - The trellis decode is instruction-bound and does not depend on the activations.
  - So a CTA that finishes its share of a GEMV can start decoding the next GEMV's weights before the activations it multiplies are ready.
  - For that, the per-tile dependency counters must replace the global barriers.
- **RTX 2060:** the GEMVs already run at 78-90 % of copy bandwidth, and the practical ceiling is ~137-149 t/s against 133. The gain is capped at a few percent.

## Options

- **(A) Dataflow megakernel**, P100 first:
  - one persistent kernel per token;
  - a per-tile dependency counter for each op;
  - weight decode started ahead of the activations.
  - Ceiling ~5-9 % on the P100 and ~2-4 % on the 2060. Weeks of work.
  - Natively it needs everything in the graph: GEMV, mgemv, the native attention, the norm, rope, the cache quant, the short conv and the sampler's argmax. The sliding layers' Triton split kernel would need a CUDA rewrite.
  - De-risk with a two-op prototype first: gate/up → down with counters, measured against the graph.
- **(B) Targeted fusions**, each A/B'd with doc/lfm8_opt/ab_dec.sh. Each one removes a boundary plus a latency-bound kernel. Candidates:
  - rope + quant_cache_paged (11 + 11 calls);
  - mul_sigmoid into the o_proj input stage (11);
  - copy2d (11);
  - the input norm into the GEMV input stage (47.6 norm calls × ~5.6 µs);
  - the attention combine at short depth (11 × ~13 µs).
  - Roughly 2-4 % in total, days of work. These ops are also the building blocks (A) needs.

## Round H: the decode GEMV hot loop on the P100 (2026-11-23)

**Hot loop SASS** (doc/lfm3_mega/hot_loop.py: the loop with the most VABSDIFF4) of `exl3_mgemv_kernel<8,0,0,1,BAL>` (gate/up, 2.3 ms per token):
- 599 instructions per 64 weights per lane, 10.6 issue slots per weight;
- XMAD 128 + XMAD.PSL 96 (3.5 per weight);
- VABSDIFF4 64 (half rate);
- HFMA2 64 (0.5 per weight codebook scale, 0.5 per weight dot);
- HADD2.F32 + FADD 32 + 32 (fp16 → fp32 fold);
- MOV 56.

GP100 issues 2 warp-instructions per clock per SM. At ~149 G per second the loop alone accounts for ~37 of the 46.7 µs, so the kernel is issue bound.

- **H1 REJECTED** (code removed): fold every 64 terms instead of 32 in the CFG 1 (balanced) kernels.
  - The fold became a branch every other ring pass, adding 16 MOVs.
  - Microbench (ab_so_micro.sh, ctl / cand / ctl after warm-up): 2048 × 7680 46.8 / 46.7 µs, 7680 × 2048 48.1 / 47.9. Neutral.
  - It would also have changed the numerics.
- **H3a ACCEPTED**, bit-identical: the staged path stores the ring slot `pf[d]` to smem before refilling it, instead of copying it to `bw` first.
  - MOVs drop from 56 to 16, and the loop from 599 to 562 instructions.
  - The gated down kernel likewise goes to 564.
  - Microbench (ctl / H3a / ctl, µs):

    | shape | ctl | H3a | ctl | change |
    |---|---:|---:|---:|---:|
    | 2048 × 7680 | 50.7* | 45.3 | 46.8 | −3.2 % |
    | 7680 × 2048 | 52.0* | 46.5 | 48.1 | −3.3 % |
    | 2048 × 6144 | 43.5* | 39.9 | 40.9 | −2.5 % |
    | 2048 × 2048 | 27.8 | 26.1 | 27.5 | −5 % |
    | 2048 × 512 | 15.7 | 14.8 | 15.7 | −5.7 % |

    \* The first control arm runs warm-up-cold.
  - Decode A/B (ab_so_dec.sh, .so swap, t/s):

    | card | depth | ctl | H3a | ctl | change |
    |---|---|---:|---:|---:|---:|
    | P100 | 512 | 144.9 | 149.2 | 144.2 | +3.0 % |
    | P100 | 8k | 141.6 | 145.2 | 140.8 | +2.8 % |
    | P100 | 32k | 131.0 | 133.3 | 129.4 | +2.4 % |
    | RTX 2060 | 512 | 132.0 | 134.4 | 133.0 | +1.0-1.8 % |
    | RTX 2060 | 8k | 124.0 | 125.9 | 124.6 | +1.0-1.8 % |
    | RTX 2060 | 32k | 102.6 | 104.6 | 103.6 | +1.0-1.8 % |

  - Greedy oracle: max |dlogit| 0 on both cards.
  - test_exl3_gemv + test_exl3_mgemv: 190 passed on the P100, 187 on the 2060.
- **H3b REJECTED** (code removed): the first product after a fold writes the fp16 partial fresh (`hmul2`) instead of accumulating into a reset zero.
  - Hot loop 562 → 549 instructions.
  - Microbench H3a / H3b / H3a: 2048 × 7680 45.3 / 45.2 / 45.3 µs, 7680 × 2048 46.5 / 46.5 / 46.5. Neutral: the resets were already hidden.
  - Static instruction counts over-predict (H1, H3b). H3a's win comes from the earlier smem store as much as from the dropped MOVs.
- **Next:**
  - E16b (below).
  - H2: the mul1 product in 2 XMADs (`XMAD.PSL x, 0x83DC, lo`). Needs SASS-level editing; ptxas never selects it.
- **E16 (HFMA2 decode attention) re-read: the −4.7 % was that implementation, not HFMA2.**
  - It loaded q one half2 at a time from smem: 2 LDS.32 per 4 dims and head, against 1 LDS.128 in the fp32 path.
  - It converted bytes with half-rate PRMT.
  - It converted p for every key in P × V.
  - Registers grew from 94 to 118 on an occupancy-limited kernel.
  - It ran before E20.
  - **E16b plan:**
    - HFMA2 for Q·K only, with P × V kept in fp32;
    - q as uint4 (8 halves per LDS.128);
    - bytes → half2 with LOP3 | 0x6400 plus an exact HSUB2;
    - built on top of E20 (8 warps).

**DSpark draft** (rasyosef/LFM3-1.7B-DSpark): looked at and dropped at the user's request.
- It is a speculators DFlash backbone plus a Markov logit bias (rank 256), a confidence head and a 32k draft vocab with d2t.
- The download was deleted.

## E16b ACCEPTED: HFMA2 score loop in the native decode attention (2026-11-23)

`EXL3_ATTN_NATIVE_HF`, default on. The changes against E16:
- HFMA2 for Q·K only; P × V stays fp32, so there is no per-key p conversion;
- q is held as fp16 in smem, interleaved {q0, q2}, {q1, q3}, and read with one LDS.128 per 8 dims and head (E16 needed 4×);
- bytes → half2 with LOP3 | 0x6400 plus an exact HSUB2 (E16 used PRMT, half rate on sm_60);
- 94-96 registers (E16 used 118).

Results:
- test_decode_native: 72 passed on both cards.
- P100 decode A/B (ab_dec.sh, HF 0 / 1 / 0, t/s):

  | run | depth 512 | depth 8k | depth 32k |
  |---|---|---|---|
  | 1 | 149.5 / 151.1 / 149.3 | 145.6 / 147.2 / 144.9 | 134.6 / 135.9 / 132.7 |
  | 2 (repeat) | 149.5 / 151.1 / 149.3 | 145.3 / 147.0 / 145.0 | 132.8 / 135.5 / 133.2 |

  That is +1.1 % / +1.1..1.6 % / +1.7..2.4 %.
- Greedy oracle 128/128 identical (max |dlogit| 0.031); decode ppl 9.93979 in both arms.
- RTX 2060: neutral (134.3 / 134.4 / 134.6 at 512), as expected: the native kernel is sm_60-only by default.

## LFM3-1.7B 4.0bpw vs 8.0bpw on the current build (doc/lfm3_mega/bench_4bpw.sh)

| card | 8.0bpw t/s (512 / 8k / 32k) | 4.0bpw t/s | decode ppl 8.0 → 4.0 |
|---|---|---|---|
| P100 | 149.4 / 145.2 / 133.0 | 156.1 / 152.0 / 139.8 (+4.5 %) | 9.93979 → 9.97686 |
| RTX 2060 | 134.5 / 125.8 / 104.6 | 189.6 / 172.4 / 134.2 (+41 / +37 / +28 %) | 9.94076 → 9.97451 |

- **RTX 2060:** the decode is bandwidth bound, so halving the weight bytes nearly pays out in full.
- **P100:** the gain is small. The hill-climb tuned the K = 7/8 paths, and K = 4 on Pascal has no multi-matrix GEMV (use_mgemm gates K 7/8) and no staged sm_60 decode.
  - Round-4 microbench: 38.6 µs (4 bpw) vs 48.9 µs (8 bpw) per 15.7 M-weight matrix.
  - That leaves an estimated +15-20 % for a K = 4 pass on the P100.
