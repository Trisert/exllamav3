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

### Round 2 — both REJECTED
| arm | d512 greedy | d8192 greedy |
|---|---:|---:|
| P100 ctl a (E4 off) | 126.8 | 117.7 |
| P100 E4 (wide for n == k at K >= 7) | 125.2 | 116.3 |
| P100 ctl b | 126.5 | 117.5 |
| 2060 ctl a | 131.1 | 118.3 |
| 2060 E5 (EXL3_GEMV_BALANCE=0) | 126.7 | 114.7 |
| 2060 ctl b | 130.5 | 117.8 |
| 2060 E5 repeat | 126.8 | 114.6 |
- E4: -1.2 % end-to-end although the isolated microbench had the square shape 26.4 -> 22.7 us. The in-graph cost
  does not follow the isolated one (back-to-back different shapes, L2 state); code removed.
- E5: the balanced split is worth +3 % on the 2060 at 8 bpw: stays on.

### Qualification (final build c200e416)
- tests/kernels/gemm/test_exl3_gemv.py: 46 passed on both cards.
- Needle (doc/lfm8_opt/needle.py, 16k 8-bit KV cache, chat template, greedy, reasoning on): 6/6 on both cards at
  8.0k and 13.8k tokens, needle at 10 / 50 / 90 %.
- Greedy oracle / decode ppl: see round 1.

### Round 3 — E6 ACCEPTED (hgemv split-K scratch inside graph captures)
The cuBLAS gemv2N (P100) / gemvx + splitKreduce (2060) at 11 calls per token was the headwise attention gate
(`g_proj`, LinearFP16 2048 -> 16): eager it takes hgemv (~7 us), but hgemv keeps its split-K scratch per (device,
stream) and refuses to allocate during a capture, so inside the decode graphs it fell back to cuBLAS (~18 us on the
P100). Fix in hgemv_f16.cu: a per-device scratch set, allocated beside the first eager one, used by captured launches
(`EXL3_HGEMV_GRAPH=0` = old behaviour). New test tests/kernels/gemm/test_hgemv_f16.py::test_hgemm_in_graph_takes_gemv
(98 passed on both cards; with EXL3_HGEMV_GRAPH=0 the 2048 x 16 case fails).
| arm | d512 greedy | d8192 greedy |
|---|---:|---:|
| P100 ctl a | 126.8 | 117.7 |
| P100 E6 | **127.6** | **118.3** |
| P100 ctl b | 126.5 | 117.4 |
| P100 E6 repeat | **127.4** | 117.8 |
| 2060 ctl a / E6 / ctl b / E6 | 130.9 / 130.7 / 130.6 / 130.5 | 118.5 / 117.9 / 117.9 / 117.9 |
P100 +0.7 % (smaller than the 0.11 ms the kernel times suggested), 2060 neutral (its cuBLAS gemvx was already ~2.5 us).
Greedy-128 oracles identical to round 1 on both cards (128/128). Side effect, not measured: the same capture fallback
hit the fused router top-k and every other fp16 linear at <= 4 rows inside graphs (MoE routers of QNF / Qwen3.6).

Lever 2 (one launch for k + v) cannot be done by concatenating the weights: each EXL3 tensor has its own input
Hadamard signs (suh), so it needs a multi-matrix GEMV kernel.

### Round 4 — K=8 GEMV inner loop on the P100 (kernel work)
The P100 GEMV is instruction-bound: one 15.7M-weight matrix takes 38.6 / 47.0 / 48.9 us at 4 / 6 / 8 bpw (memory
floor at 8 bpw: 29 us). SASS of the 8 bpw wide kernel's hot loop (doc/lfm8_opt/sass_hist.py): 634 instructions per
64 weights per lane (XMAD 224, VABSDIFF4 64, HFMA2 64, MOV 58, fold 64, SHF.R.U64 48, LDS 32). Measured issue rates
(doc/lfm8_opt/instr_rate.py): XMAD, LOP3, HFMA2, IADD full rate; VABSDIFF4, SHF, PRMT half rate.
- E7 REJECTED: prefetch / fold depth 4 for the wide and balanced configs: 2048x7680 46.8-53.5 -> 54.6 us, K=4 worse too.
- E8 ACCEPTED: fixed-offset staged decode for 8 bpw mul1 on sm_60 (s2 = 0 for every lane: windows at 24/16/8/0, 3 LDS
  instead of 4, one immediate funnel shift instead of three variable SHF.R.U64). Bit-identical (output hashes on 5
  shapes, greedy oracle 128/128 with dlogit 0). Microbench 2048x7680 49.0 -> 46.7 us, 7680x2048 50.2 -> 48.1.
  End-to-end P100 127.3-127.8 -> **132.9-133.0** t/s at depth 512, 118.3-118.9 -> **123.0-123.1** at 8192.
- E9 REJECTED: `#pragma unroll 2` on the k-slice loop (to drop the ring-rotation MOVs): big shapes 46.8 -> 50.8 us.
Remaining in the loop: the 3 XMAD per weight of the mul1 product (no 32-bit IMAD on Pascal) and the half-rate
VABSDIFF4 byte sum are inherent to this codebook; estimated floor ~7.4 issue slots per weight vs ~9.4 now.

### Prefill (2026-11-22)
Measurement note: the repo perf script only appends its own parent to sys.path, so a script under doc/ imported the
venv's editable install (the main checkout) until doc/lfm8_opt/prefill_perf.py inserted the tree first; the first
prefill numbers were from the main checkout (same prefill code, the control arms below agree with them).
Profile, P100, 4096-token prefill (model.prefill, 1698 ms): exl3 GEMMs (reconstruct + HFMA2 GEMM) 911 ms at ~10.6
TFLOPS (57 % of fp16 peak), **Triton `_paged_attn_prefill_kernel` 495 ms (the 5 sliding-window layers, ~0.13 TFLOPS)**,
SDPA (6 full-attention layers) 190 ms.
- E10 ACCEPTED (`EXL3_SDPA_SWA`): sliding-window layers on the dense-window SDPA path with a banded mask per q chunk.
  Test: tests/kernels/attention/test_prefill_sdpa_window.py::test_sliding_window_path_matches_triton (24 cases, both
  cards, vs the Triton kernel at 2e-3). Prefill ppl (doc/lfm8_opt/prefill_ppl.py): P100 11.03163 -> 11.03238 (4096
  tokens, top-1 4082/4096), 2060 8.61278 -> 8.61413 (2048 tokens, top-1 2044/2048). Needle 6/6 on both cards.

| prefill t/s (ctl a / E10 / ctl b) | P100 | RTX 2060 |
|---|---|---|
| 512 | 2452 / **3061** / 2397 | 3592 / **4899** / 3630 |
| 2048 | 2504 / **3284** / 2459 | 4016 / **6473** / 4078 |
| 4096 | 2427 / **3163** / 2386 | 4086 / **6628** / 4064 |
| 8192 | 2211 / **2796** / 2184 | 3877 / **6156** / 3853 |
| 16384 | 1885 / **2275** / 1861 | 3568 / **5399** / 3545 |

- E11 ACCEPTED (`EXL3_FA60_HD128`, P100): fa60 templated on head_dim (256 unchanged: same HFMA2 sequence per
  accumulator), head_dim 128 for the full-attention layers instead of SDPA's SIMT cutlass kernel. Test:
  test_fa60.py with D = 256 / 128 and 16/4 heads (87 passed). Prefill ppl 11.03238 -> 11.03115 (top-1 4077/4096),
  needle 6/6. The 2060 is unaffected (fa75 applies there, head_dim 128 keeps SDPA on HMMA).

| P100 prefill t/s (ctl a / E11 / ctl b) | |
|---|---|
| 512 | 3050 / 3074 / 2999 |
| 2048 | 3294 / **3449** / 3257 |
| 4096 | 3184 / **3504** / 3137 |
| 8192 | 2807 / **3346** / 2766 |
| 16384 | 2294 / **3074** / 2262 |

Prefill after E10 + E11 vs the start: P100 2407 -> 3504 t/s at 4096 (+46 %), 1873 -> 3074 at 16k (+64 %); RTX 2060
4075 -> 6645 (+63 %), 3557 -> 5407 (+52 %). What is left: on the P100 the exl3 reconstruct + HFMA2 GEMMs (~71 % of a
4k prefill at ~57 % of the fp16 peak); on the 2060 cuBLAS GEMMs on HMMA at ~22.6 TFLOPS average (70 %), close to
the card.

## STATUS after rounds 4-5 (final build 60280f40, re-measured)
| metric | start | now |
|---|---:|---:|
| P100 decode d512 / d8192 | 51.7 / 50.1 t/s | **132.6 / 122.8** (x2.56) |
| RTX 2060 decode d512 / d8192 | 117.8 / 107.2 | **130.7 / 118.3** (+11 %) |
| P100 prefill 4096 / 16384 | 2407 / 1873 t/s | **3504 / 3074** (+46 % / +64 %) |
| RTX 2060 prefill 4096 / 16384 | 4075 / 3557 | **6645 / 5407** (+63 % / +52 %) |

## STATUS (2026-11-22, after round 3)
| metric | baseline | now | floor (copy BW) |
|---|---:|---:|---:|
| P100 d512 greedy | 51.7 t/s | 126.5-126.8 t/s (x2.45) | ~323 t/s |
| P100 d8192 greedy | 50.1 | 117.4-117.7 | |
| 2060 d512 greedy | 117.8 | 129.3-131.1 (+10-11 %) | ~174 t/s |
| 2060 d512 user sampler | 116.7 | 128.9 | |
| 2060 d8192 greedy | 107.2 | 117.5-118.3 (+10 %) | |
All medians of 3 x 256 tokens, util/bench_decode.py, graphs on, 16k 8-bit KV cache.
Final build c200e416 re-measured: P100 126.5 / 117.5 greedy, user sampler 110.4 (d512) / 115.2 (d8192); 2060 130.5 /
118.0 greedy, 129.3 / 116.7 user. Open observation: on the P100 the user sampler costs 1.15 ms/token at depth 512 but
0.17 ms at 8192 (no P100 user-sampler baseline to compare; not investigated).
Scope: the change applies to every 7 / 8 bpw mul1 tensor on Pascal and sm_75. The other local quants (Qwen3.8-Flash-Next
1.66bpw, Qwen3.6-35B-A3B 4.00bpw) were not A/B'd; only an 8 bpw tensor in them (e.g. a head) would be affected.

What limits further gains:
- P100 (7.9 ms/token): K=8 GEMV 6.5 ms at ~227 GB/s average in-graph (42 % of copy BW; the trellis decode is
  instruction-issue bound on sm_60), small shapes (2048x2048, 2048x512) at 50-185 GB/s are fixed-latency bound,
  glue ~1.7 ms.
- 2060 (7.6 ms/token): GEMVs at ~226-261 GB/s (78-90 % of copy BW), lm_head 0.66 ms, glue ~0.6 ms. Near the practical
  ceiling (~137-149 t/s).

Remaining levers, ranked (estimated gain / cost):
1. (done, round 3: +0.7 % on the P100) per-head attention gate on cuBLAS inside graphs.
2. P100: multi-matrix GEMV for the q/k/v bundle (one launch for 2048 x (2048 + 512 + 512)) to cut the fixed latency of
   the small shapes (~0.3 ms): ~3-4 %, 1-2 days.
3. P100: K=8 GEMV inner loop (instruction count per weight, HFMA2): the big one (up to ~1.5-2x on 6.5 ms), multi-day
   kernel work.
4. Not measured: m = 2..32 (batched decode / speculative verify) on sm_75 now runs two tile GEMMs for gate/up instead of
   one fused MGEMM at 7 / 8 bpw (the m >= 2 K >= 5 GEMV is not enabled there). Batch-1 decode is unaffected.
