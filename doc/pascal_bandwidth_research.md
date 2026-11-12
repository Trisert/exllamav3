# P100 decode GEMV: why the bandwidth is not used (research notes, Nov 12)

Condensed from a subagent study (scripts, SASS and raw logs in `exllamav3/tmp/bw_research/`, not in the repo).
Nothing was run on a real checkpoint or on the MoE coop kernels; all numbers are P100 micro-benchmarks with
a thermal clock swing of 0.6-1.33 GHz (run-to-run noise about +-5 %).

## Finding: the m = 1 GEMV is instruction-issue-bound

- K = 4 wide config, SASS main loop: 510 executed warp-instructions per 64 weights per lane = **7.97 per weight**.
  The SM issues 2 warp-instr/clk, at the measured ~1.26 GHz this predicts ~104 us for k = 4096, n = 14336;
  measured 100-112 us (7 us fixed cost removed). The "262 GB/s peak" is this ceiling.
- Not memory-bound: replacing all global loads by cache hits leaves the time unchanged. The same access pattern
  streams at 450-533 GB/s (best sequential 594) against the 227-265 GB/s used. Not occupancy-bound either.
- Pipes: XMAD, XMAD.PSL, IADD, LOP3, HFMA2, FADD run at ~1.9-2.0 warp-instr/clk/SM. VABSDIFF4, SHF, PRMT, BFE
  run at 1.0 on a separate half-rate pipe. In the real loop: full-rate pipe ~71 % busy, half-rate ~44 %, issue ~95 %.
- Per-weight budget (7.97): 3.0 product XMADs, 1.0 VABSDIFF4, 0.5 pack, 0.5 scale/bias HFMA2, 0.5 dot HFMA2,
  0.375 SHF, 0.5 fp32 fold, 0.44 MOV, 0.31 LDG+SHFL, 0.84 control/address/predicates. Other K: K=2 8.8, K=3 9.4,
  K=5 10.0 (wide) / 10.9 (narrow).

## Second loss: grid imbalance

Wide config = 112 co-resident blocks and 64-column groups, narrow = 56 blocks and 32-column groups. A model of
per-SM issue load plus 7 us fixed cost reproduces all 16 measured rows (8 Flash-Next shapes x K = 4, 5) within
~3 %. Perfect balance would be 1.25x (K = 4) and 1.28x (K = 5) time-weighted (1.04-1.56x per shape). Wide is also
3-11 % faster than the narrow heuristic on small shapes with n <= k (e.g. K=4 4096x2560: 36.2 vs 39.5 us).

## Ranked improvements

| # | change | gain | cost |
|---|---|---|---|
| R1 | split-k load balance (units = 64-col group x k chunk, red.add fp32 scratch) + wide-config heuristic | modelled +25 % (K=4) / +28 % (K=5); heuristic part measured 3-11 % | fp32 summation order only; m*n*4 B scratch; not built |
| R2 | decode-side exact diet (peeled loop, fold every 8 slices, scale/bias folded into dot) | SASS 7.97 -> 6.33 per weight, but **measured 0..-8 % time** | exposed load latency eats the gain; NO-GO alone |
| R3 | cheaper scalar codebook product (1 or 2 XMAD) | micro-bench +16..+35 %, real kernel est. +8..15 % | requantisation, -0.1..-0.4 dB |
| R4 | pair-trellis LUT codebook (1 LDS.32 -> half2 per two weights) | micro-bench +60 %, LDS-bound | quality within noise at K <= 2, -0.3 dB K=3, unusable K >= 4; only for MoE experts (K <= 2.5) |

Rejected: smem product tables (two random LDS at ~3.8 clk vs 3 XMAD), pipe shuffling (issue-bound), deeper ring
(PF 3/4 no help), different trellis stride, decode amortisation over m > 1 (already done).

Codebook quality (Viterbi MSE on iid N(0,1), L = 16, dB; mul1: 11.74 / 17.63 / 23.43 / 29.03 at K = 2/3/4/5):
1-XMAD `S*0xc60f` 11.52 / 17.49 / 23.09 / 28.70; 2-XMAD `S*0x1_9c9f` 11.60 / 17.51 / 23.16 / 28.72.
Pair-trellis LUT (Q = 12, Lloyd-trained): K=2 11.61-11.69, K=3 17.38-17.42, K=4 22.61-22.65, K=5 26.73 (equal-n study).

## Caveats (advisor review)

- The issue-bound diagnosis is solid (SASS count x issue rate matches the time; loads removed = same time; same
  access pattern streams ~2x faster without compute).
- The +25 % for R1 is **modelled and unverified, an upper bound**: no split-k kernel exists, the model fits 16
  rows within 3 % against +-5 % noise with 3 tuned constants and a stale clock log, the gain is concentrated on small
  shapes where the fixed cost is 30-45 % and split-k adds more (scratch zeroing, atomics, extra sync), and the
  weighting is not the real call mix. The wide-heuristic deltas (3-11 %) are marginal against the noise.
- Missing: the end-to-end relevance. Everything is m = 1, K = 4/5; the experts use the MoE coop kernels (not
  instrumented). Before building anything, compute from the existing decode-window kernel reports (`tmp/w1`) the
  share of iterate wall time spent in `exl3_gemv` on the P100, which K and m it uses with `-ndt 3`, and
  share x realistic saving in tokens/s. Cheap falsification of the model: K=4 wide, k=2560, n=7168 (balanced) should
  take ~40 us against 56 us measured at n=8192.

## Recommendation

GO: R1 (split-k + wide heuristic), stop if the 8-shape suite (K = 4, 5; m = 1, 2) does not show >= 15 %
time-weighted. NO-GO: decode-only work on K = 4/5 until the load pipeline is reworked. R4 is a separate research
item for the MoE coop kernels (K <= 2.5), starting with a pair-trellis quantiser prototype on one expert tensor.

Plan: (1) torch-free baseline harness with best-of-5 interleaved runs, (2) heuristic change and re-measure on the
16 rows (0.5 day), (3) split-k kernel (2-3 days), (4) only then revisit R2 together with the load pipeline,
(5) R3/R4 only if requantisation is accepted.
