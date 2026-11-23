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
