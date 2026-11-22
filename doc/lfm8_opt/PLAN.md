# LFM3-1.7B 8.0bpw decode hill-climb (RTX 2060 + Tesla P100)

Targets: batch-1 decode t/s (greedy and "user" sampler) at depth 512 and 8192; lossless only (greedy output must match
the baseline oracle). Harness: util/bench_decode.py (median of 3 x 256 tokens, wall between first and last token),
-cs 16384 -cq 8. Raw outputs in /home/nicola/exllamav3/tmp/lfm8_opt/. Ceilings in THEORY.md.

## 2026-11-22

### Baseline (flash-lead1 .so, provenance unpinned)
| card | depth 512 greedy | 512 user | 8192 greedy | 8192 user |
|---|---:|---:|---:|---:|
| RTX 2060 | 117.7 | 116.4 | 107.3 | 106.0 |
| P100 | 51.7 | - | 50.1 | - |

Profile (64 steps, depth 512, ms/token):
- 2060 (8.18 busy): mgemm<8> gate/up 3.71 (24x, ~204 GB/s), GEMV<8> 2.01 (48x) + 0.89 (24x), lm_head GEMV<6> 0.66
  (~298 GB/s), mgemm<8> qkv 0.26 (11x), glue ~0.6
- P100 (20.09 busy): no K=8 GEMV at all. mgemm<8> gate/up 8.61, gemm<8> 5.82 (48x) + 2.67 (24x), qkv mgemm 0.81, all at
  ~80-90 GB/s; the 6 bpw lm_head GEMV runs at ~356 GB/s (0.55 ms)

Gates found: exl3_gemv.cu `K >= 7 && !sm75 -> -1` (no K=7/8 GEMV on Pascal); config.py use_mgemm unfuses only for
K <= 5 (so at 8 bpw gate/up and qkv stay on the fused tile mgemm on both cards).

### Control build (lfm8-opt @ 2be183fb, clean rebuild)
2060 118.0 / 116.7 / 107.3 / 106.1; P100 51.7 / 50.1 (same as above). Greedy-128 oracles: oracle_{2060,p100}_base.json
(the two cards diverge from each other at token 90: a near-tie, fp16 order).

### Round 1 — ACCEPTED (E1 + E2 + E3), control / candidate / control, same .so, arms by env
- E2 `exl3_gemv.cu`: K = 7 / 8 GEMV allowed on Pascal (`EXL3_GEMV_PASCAL_K8`, default on). Kernel test
  tests/kernels/gemm/test_exl3_gemv.py on the P100: 46 passed.
- E3 `config.py use_mgemm`: K = 7 / 8 mul1 unfused on Pascal (gate/up and the q/k/v bundle go to separate GEMVs).
- E1 same place: K = 7 / 8 mul1 on sm_75 unfused only for n >= 4096 (gate/up; sliced qkv bundle stays fused).
  `EXL3_UNFUSE_K8=0` keeps the fused MGEMM for both.

| arm | d512 greedy | d8192 greedy |
|---|---:|---:|
| P100 ctl a | 51.7 | 50.1 |
| P100 E2 only (fused gate/up) | 67.8 | 65.0 |
| P100 E2 + E3 | **126.5** | **117.4** |
| P100 ctl b | 51.6 | 50.0 |
| 2060 ctl a | 117.8 (user 116.7) | 107.2 (user 105.8) |
| 2060 E1 | **129.3** (user 128.9) | **117.5** (user 116.2) |
| 2060 ctl b | 117.4 (user 116.3) | 106.6 (user 105.3) |

Correctness: P100 E2+E3 greedy-128 identical to the P100 base (128/128, final top-20 same, max |dlogit| 0.06). 2060 E1
diverges from the 2060 base at token 90 but is identical (128/128) to the P100 base: the base 2060 fused mgemm was on
the other side of the near-tie. Decode-mode ppl (doc/lfm8_opt/decode_ppl.py, 1536 tokens teacher-forced at m = 1):
P100 9.93801 -> 9.94133, 2060 9.93904 -> 9.94162 (+0.03 %, same size as the cross-card difference); top-1 agreement
1534/1536 for every pair, including base 2060 vs base P100. fp16-order noise, not a defect.

Profile after round 1 (ms/token, depth 512): 2060 7.34 busy (gate/up GEMV 2.89 at ~261 GB/s, other K=8 GEMV ~226 GB/s,
lm_head 0.66); P100 8.71 busy (K=8 GEMV 6.48 at ~227 GB/s average, small shapes 2048x2048 / 2048x512 at 50-185 GB/s,
glue ~1.7 ms: paged attention 0.58, norms 0.40, cuBLAS gemv2N 0.20, rope 0.12, conv 0.10).

### K=8 GEMV config microbench (doc/lfm8_opt/bench_gemv_k8.py, us per call, m = 1)
- P100 (stable to ~1 us): the balanced split wins on 2048x7680 (49 vs 58-64) and 7680x2048 (50 vs 67-73), ~neutral on
  2048x6144 (42.5 vs 41); on 2048x2048 the heuristic picks narrow (26-29) while wide is 22.7-25; 2048x512 narrow 15.2
  (wide 21). -> E4.
- RTX 2060: run-to-run noise 5-8 % (boost clocks), microbench differences of 3-4 us are not resolvable; judged end-to-end
  only (E5: balanced split off).
