# Qwen3.8-27B (dense): review Fable vs Opus (Nov 11)

Read-only reviews by two subagents (no GPU runs). Base for all work: branch `qwen38-27b-dflash2-head` (c3953c1a),
which contains `flash-opt` plus 26 commits (MMODE 6, `-dgs`, 2-blocks/SM 5/6 bpw head, DFlash2 docs). Only Opus found
this branch; Fable cited `flash-opt` and so missed MMODE 6 and the DFlash2 baseline (43.9 t/s, 131 ms/step).

## Agreement
- The model on disk is mixed 3/4/5/6 bpw with a 5 bpw lm_head, not the 4 bpw of Stage 0: re-measure first.
- Biggest lever is P100 cooling (hot 230 vs 103 ms/step; 24k prefill 88 vs 196 t/s).
- **K=3 has no MMODE 5 instance** (`SEL_GRID_M5` only K=4/5/6, `exl3_gemv.cu`): at m=9..16 about 20 K=3 calls per step
  fall to the sm_60 tile GEMM (~4.8 ms vs ~0.3 ms). Add `SEL_GRID_M5(3,2,false)` (+ MMODE 6 for K=3), extend
  `tests/kernels/gemm/test_exl3_gemv_m16.py`, then use `-hdt 12`. Derived from code, not measured.
- `-hdt 7` suffix drafting on copy-heavy agentic turns; `-sdh 16384`; K=5/6 GEMV on the 2060 (gates `exl3_gemv.cu:88`,
  `config.py:94`).
- Does not transfer: gated-residual fusion, n-gram prefetch/ramp, MoE coop, CPU-tail DtoH, async DMA, `-mcs`.

## Differences
| item | Fable | Opus |
| :--- | :--- | :--- |
| gain from -hdt | ~+40% (38 -> 55 t/s) | ~80-90 t/s vs 44 (on DFlash2 base) |
| K=3 MMODE 6 (m=5..8) | not seen | ~-3 ms/step |
| prefill 17..64-row chunks on tile GEMM | probe TTFT | not mentioned |
| 2060 clock lock, GDN m=8..16, max_chunk_size 4096 | no | yes |
| DFlash2 drafter | not seen | removed from models/, re-download 1.4 GB |

Both gain figures are estimates; neither was run. Opus's 80-90 t/s assumes `-hdt` composes with DFlash2, unverified.

## Next steps (in order)
1. K=3 MMODE 5/6 instances + tests (needs a build: 4 jobs if a model is loaded; one TU peaks ~7.5 GB RAM).
2. Cooled one-load-per-arm A/B on the P100 (<=62 C): `-hdt 0/7/12`, `-sdh 16384`, baseline `prof_m.py`.
3. `max_chunk_size` 4096 and 17..64-row TTFT probe; then the sm_75 K=5/6 GEMV.

The GPUs and RAM are shared with the Flash-Next session (`tmp/gpu.lane` flock): take the lane before any run.
