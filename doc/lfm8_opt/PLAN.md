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

### Follow-ups for PR #4
- Scope: QNF (Qwen3.8-Flash-Next) and Qwen3.6-35B-A3B have head_dim 256 and no sliding window, so E10 / E11 do not
  apply to them (fa60 at head_dim 256 is bit-identical, output hashes on 3 cases).
- test_fa60_long now covers head_dim 128 at 16384 / 30000 keys (passed).
- m = 2..4 (perf script -sd, it/s at context 512; ctl = EXL3_UNFUSE_K8=0):
  - RTX 2060 S=1/2/3/4: ctl a 117.3 / 101.9 / 100.7 / 99.3, default 130.1 / 104.6 / 104.1 / 102.8, ctl b
    113.7 / 99.3 / 96.7 / 98.0. No regression at m = 2..4 from the unfused gate/up (+3-5 %).
  - P100: ctl 67.5 / 62.7 / 60.1 / 60.0, default 122.6 / 97.8 / 87.6 / 87.4.

### Round 6 — long context (decode at 32k, 64k 8-bit cache)
Depth 32768: P100 99.2 t/s (10.08 ms; decode attention 3.09 ms = 28 % of the token: ~201 MB of 8-bit KV per token in
2.94 ms, ~68 GB/s), RTX 2060 90.5 t/s (11.05 ms).
- Split target 2 -> 8 / 32 x SMs (bc_attn.py splits_cap; the eager path's knob alone had no effect, the decode graph
  runs the BC attention): neutral on both cards (P100 99.5 / 100.9 / 102.3, 2060 90.6 / 91.3 / 89.6).
- Key tile 16 -> 64 (32 in between): P100 99.4 -> 103.2 at 32k but 132.8 -> 122.9 at 512 and 122.6 -> 117.7 at 8k
  (REJECTED for sm_60: one tile per graph slot, it cannot follow the context length); RTX 2060 90.4 -> 101.2 at 32k.
- E12 ACCEPTED (sm_75, head_dim <= 128: 64-key tiles, `EXL3_ATTN_DECODE_BN` pins): RTX 2060 ctl a / E12 / ctl b
  512: 130.6 / **129.4** / 130.5 (-0.9 %), 8192: 118.4 / **121.5** / 118.0 (+2.8 %), 32768: 90.6 / **101.4** / 90.5
  (+12 %). Greedy oracle 128/128 identical, decode ppl 9.94076, top-1 1531/1536 vs the E1 build;
  test_decode_live_split.py 56 passed. head_dim 256 models (QNF, Qwen3.6) keep 16.
- Long-context gate (doc/lfm8_opt/longctx_gate.sh, 64k 8-bit cache): needle 3/3 at 49.0k tokens (10 / 50 / 90 %) on
  both cards; prefill to 64k (t/s) P100 16k 3046, 32k 2617, 64k 2024; RTX 2060 16k 5287, 32k 4295, 64k 3110.
- Split kernel launch shape (warps,stages; `EXL3_ATTN_DECODE_WS`, doc/lfm8_opt/ws_sweep*.sh) — REJECTED, the
  default 4,2 stays. P100 (tile 16 unless noted), depth 512 / 32768: 4,2 133.0 / 99.4 (control after: 132.8 / 99.3),
  2,1 124.4 / 87.1, 2,2 124.5 / 87.0, 8,2 133.6 / 85.9, 4,1 132.9 / 99.6; tile 64 at 2,1 107.9 / 82.9, at 8,1
  129.7 / 100.4 (+1 % at 32k for -2.5 % at 512). RTX 2060 (tile 64): 4,2 129.8 / 101.8 (control after 129.4 / 100.9),
  2,1 123.8 / 81.3, 8,1 129.9 / 93.6, 2,2 123.7 / 80.9. Per-call profile on the P100 (prof_bn.sh): the split kernel
  takes 40 us at 512 and 267 us at 32k with 16-key tiles, 71 / 207 us with 64 — the remaining long-context gap is in
  the Triton kernel itself (scalar-FMA tl.dot on sm_60), not in its launch configuration.

### Round 7 — P100 GEMV launch structure (multi-matrix GEMV)
Profile (P100, depth 512, build 7615dfe3, profiler-inflated): 8.35 ms/token, of which the exl3 GEMVs 6.76 ms (81 %),
decode attention 0.59, norms 0.40, the rest ~0.6. Per call in the graph: gate 2048x7680 ~60 us, up ~47, down ~61,
conv in_proj ~42, q / o_proj / conv out_proj (2048x2048) ~38, k / v (2048x512) ~16 (grid 16: 16 of 56 SMs).
- mul1 product in 2 XMADs instead of 3 (doc/lfm8_opt/xmad_probe.cu): REJECTED without a run. Every PTX form (16-bit
  high product, shl + add, 32-bit multiply by 0x83DC0000) compiles to 3 XMADs or to the full 32-bit multiply;
  `XMAD.PSL x, imm, c` exists in SASS but ptxas does not select it. Backs the round 4 "inherent" note.
- Microbench was L2-warm below 4 MB (one tensor reused): bench_gemv_k8.py now cycles over copies totalling >= 48 MB
  (BG_COLD_MB). Cold P100: 2048x2048 28-30 us (heuristic narrow) / 24-26 (wide), 2048x512 15.8-17.3, 2048x1024 17.8,
  2048x3072 27.1 (wide).
- Forcing the balanced split on the small shapes (EXL3_GEMV_BALANCE=2): no gain (2048x2048 29.9 vs 30.1 us),
  2048x512 worse (19.6 vs 17.3; its output stage sums ~128 slots per group serially). REJECTED.
- Phase probe (doc/lfm8_opt/gemv_probe.cu: the kernel header built standalone with EXL3_GEMV_BAL_DBG=128, 200
  launches in one CUDA graph, cold copies; us): input Hadamard + grid.sync ~2.3 per call on every shape; 2048x2048
  narrow: main avg 12.2 / max 20.7 (64 column groups on 56 blocks: two rounds), 9.7 waiting at the second grid.sync;
  2048x7680 balanced: main 30.8 avg, 8.0 avg waiting for the slowest warp, ~6 of launch / tail outside the probes;
  128x128: 7.4 in graph (fixed floor). One launch over a wider matrix amortizes it: 2048x15360 balanced 82.7 vs
  2 x 49.6 (gate + up), 2048x3072 wide 24.9 vs 30 + 17.2 + 17.2 (q + k + v), 2048x1024 narrow 16.1 vs 2 x 17.2 (k + v).
  Estimate: ~0.4 ms (gate + up, 24 layers) + ~0.43 ms (q / k / v, 11 layers) of 7.5 ms per token.
- E13 (in progress): multi-matrix GEMV `exl3_mgemv_kernel` (exl3_gemv_kernel.cuh; the kernel body is now a device
  function shared by the single and multi wrappers, single-path SASS opcode histogram unchanged): the bundle's
  matrices as one virtual matrix of num_slices x width columns, per-source input transform, exl3_mgemm's uniform and
  sliced tables and launch signature (same graph sites), m = 1..16 with the single GEMV's row modes, balanced split
  at m = 1. Dispatched from exl3_mgemm on Pascal at 7 / 8 bpw mul1 (`EXL3_MGEMV`, 0 = off); use_mgemm keeps those
  bundles fused instead of unfusing them. LFM3's headwise fp16 attention gate no longer excludes the sliced Q / K / V
  bundle (it runs beside it as before; `EXL3_QKV_HEADWISE=0` restores the separate projections).
  Tests: tests/kernels/gemm/test_exl3_mgemv.py (uniform / sliced, m = 1..16, fp16 / fp32 out, in-graph replay, the
  kernel taken on Pascal; K / V pairs bit-identical to the single GEMV at m = 1): 144 passed on the P100; with the
  existing GEMV / sliced-mgemm / m16 / m34 suites 161 passed on the P100, 276 on the RTX 2060.
  (A first parallel run on both cards was stopped: the UPS alarmed under the combined load. Every run below is one
  card at a time.)
  **E13 ACCEPTED on the P100** (doc/lfm8_opt/ab_e13.sh, ctl = EXL3_MGEMV=0, t/s at depth 512 / 8192 / 32768):

  | P100 | 512 | 8192 | 32768 |
  |---|---:|---:|---:|
  | ctl a | 132.5 | 122.7 | 99.0 |
  | E13 | **141.3** | **130.1** | **103.7** |
  | ctl b | 132.5 | 122.6 | 99.1 |

  +6.6 / +6.1 / +4.7 % (about half the isolated-kernel estimate). Greedy oracle 128/128 identical to the E8 build
  (max |dlogit| 0.039), decode ppl 9.94133 -> 9.93979, top-1 1533/1536.
  RTX 2060 (doc/lfm8_opt/ab_e13_2060.sh): only the headwise Q / K / V bundle changes there and it runs on the tile
  MGEMM: ctl a 131.3 / 123.2 / 102.5, candidate 122.6 / 117.9 / 97.9, ctl b 130.9 / 122.6 / 95.4 (32k noisy) ->
  REJECTED for sm_75: the headwise bundle is now limited to where the multi-matrix GEMV runs
  (InferParams.qkv_headwise_bundle: Pascal, EXL3_MGEMV and EXL3_QKV_HEADWISE on). After the gate: RTX 2060
  131.2 / 122.9 (unchanged), P100 141.8 / 130.4.
- E14 balanced-split dynamic tail — REJECTED (code removed, probes kept). Per-warp %globaltimer / %smid probe
  (gemv_probe.cu, EXL3_GEMV_BAL_DBG & 256 / 512), 2048x7680: no entry skew; main loop 25.6 / 36.9 / 42.0 us min /
  median / max per warp for equal work, bimodal per block (~30 vs ~40) and uncorrelated between launches (r = 0.22).
  But every SM holds exactly two blocks and the pair is fast + slow (29.7 / 41.0, ...): the warp scheduler favours one
  block, the SM stays busy until its last block ends, and the per-SM end times are tight (37.9 / 38.9 / 42.0). The
  "sync wait" is not idle SM time; at most ~3 us is recoverable across SMs. Tried anyway: the last DYN_NUM / 16 of each
  group's k range as CL-slice chunks from an atomic counter (fixed slots, deterministic sum), probe in graph, us per
  call 2048x7680 / 7680x2048 / 2048x15360 / 2048x6144: static 46.2 / 47.9 / 77.2 / 43.1; 2/16 CL 2: 51.2 / 54.2 /
  80.9 / 44.8; 4/16 CL 1 / 2 / 4: 60.2 / 54.6 / 53.6 ...; 8/16 CL 2: 59.4 / 73.6 / 97.8 / 50.9 — every chunk restarts
  the prefetch ring with its full load latency, worse in all cases (doc/lfm8_opt/dyn_sweep.sh).
- **E15 ACCEPTED: CUDA-native decode split kernel** (`exllamav3_ext/attn_decode_native.cu`, `EXL3_ATTN_NATIVE`, sm_60):
  replaces the Triton split kernel inside the BC graph through the same TritonKernel launch site (new constructor
  from a runtime kernel, constants appended as extra args; graph nodes matched by the runtime symbol, which
  cudaGraphKernelNodeGetParams reports), partials in the Triton layout, Triton combine unchanged. One key per lane
  for the scores (fp32 dot of the dequantized 8-bit key with the group's q rows from smem), per-warp online
  softmax, one 4-dim slice per lane for P x V, warp merge in smem; q rotated by an in-warp Walsh-Hadamard butterfly
  (the first version's 32-term global-memory dot per element cost ~1 % at depth 512). Test
  tests/kernels/attention/test_decode_native.py (72: lengths 0..8000 incl. empty splits and page edges, both split
  modes, GQA 1 / 2 / 4 / 8, vs the eager Triton decode). Split target sweep (EXL3_ATTN_NATIVE_SPLITS, x SMs, t/s at
  32k / 512 / 8k): Triton 104.3 / 141.1 / 129.9, native 2: 123.0 / 140.1 / 135.8, 4: 124.7 / 139.2 / 135.3, 8:
  121.2 / 138.0 / 132.9 -> 2. A/B (doc/lfm8_opt/ab_e15.sh, cand / ctl Triton / cand):

  | P100 | 512 | 8192 | 32768 |
  |---|---:|---:|---:|
  | native a | 140.5 | 136.0 | 123.4 |
  | Triton | 141.3 | 129.9 | 103.8 |
  | native b | 140.7 | 136.4 | 123.3 |

  -0.5 / +4.8 / +18.8 %. Greedy oracle 128/128 identical (max |dlogit| 0.031), decode ppl 9.93979 unchanged,
  needle 3/3 at 49.0k tokens (doc/lfm8_opt/needle_p100.sh). At 32k the kernel is still ~3x off the HBM2 floor
  (~173 GB/s per call): instruction count (byte -> fp32 conversion + fp32 FMAs) and occupancy; an HFMA2 inner loop
  would be the next step there.
- E16 HFMA2 inner loops for the native split kernel — REJECTED (code removed). PRMT byte pairs -> half2 (1024 + b),
  HSUB2 to the exact b - 128, fp16 runs folded to fp32 per 32-dim score group and per 8 keys of P x V, the 0.5
  centering from fp32 sums (doc/lfm8_opt/native_err.py: rel max error vs Triton 8e-4, fp32 variant 5e-4). P100,
  HFMA2 / fp32 / HFMA2 (doc/lfm8_opt/ab_e16.sh): 512 140.2 / 140.4 / 140.0, 8k 134.2 / 135.7 / 134.1, 32k 117.6 /
  123.2 / 117.1 (-4.7 %); oracle identical, needle 3/3. The per-key fp32 -> half2 conversions of p (4 heads), the
  half-rate PRMTs and the folds cost more than the halved FMA count saves, and registers grew 94 -> 118.
- **E17 ACCEPTED: post-norm + next pre-norm in one launch** (`rms_norm_post_pre`, norm.cu; `EXL3_NORM_POST_PRE`).
  LFM3 has post-norms on both branches: r += norm(y) * w_post then the next sublayer's input = norm(r) * w_pre were
  two row kernels; one block now does both with the same arithmetic (the second norm reads r as stored), bit-
  identical: tests/kernels/norm/test_rms_norm_post_pre.py (144, torch.equal vs the two launches). In-block (attn
  post-norm -> MLP norm) and across blocks (MLP post-norm deferred to the next block's input norm through
  params["post_pending"], flushed by forward_ls at the end): 47 fused launches per token instead of 94. A/B
  (doc/lfm8_opt/ab_e17.sh, cand / ctl / cand, depth 512 / 8192): P100 142.4 / 139.3 / 142.7, 137.7 / 135.0 / 137.8
  (+2.3 / +2.0 %); RTX 2060 129.2 / 127.4 / 128.7, 120.9 / 119.6 / 120.7 (+1.2 / +1.0 %). Greedy oracle identical,
  max |dlogit| 0.0000.

### Round 8 — P100 prefill (2026-11-22, after E17)
Profile, 4096-token prefill (prefill_profile.py, PF_PREFILL=1, 1142 ms wall): hfma2 GEMMs 913 ms (80 %), SDPA (5
sliding-window layers, banded chunks) 57 ms, fa60 (6 full-attention layers) 55 ms, reconstruct 25 ms, the rest ~90.
- GEMM at the ceiling of its contract: tests/kernels/gemm/hfma2_bench/bench.py on the LFM3 shapes at M = 4096
  (doc/lfm8_opt/pf_gemm_bench.sh), efficiency vs the HFMA2 calib kernel: gate/up 79.6 %, conv in 79.0 %, down
  76.1 %, 2048 x 2048 75.0 %, k/v 69.8 % (cuBLAS 41-46 %, the 128-row tile 58-76 %). doc/pascal_progress.md "hfma2
  prefill GEMM" (Nov 6) puts the issue ceiling at ~86 % with the 32-term fp16 flush; beyond it needs a longer fp16
  partial (KSLICE 64, ceiling ~91 %, numerics change) — not done, offered to the user as a labelled trade.
- **E18 ACCEPTED: sliding-window prefill attention on fa60** (`fa60_fwd_window`, pascal/fa60.cu; `EXL3_FA60_SWA`):
  fa60 takes a causal window (key tiles below each 64-row query block's band skipped, keys below row - window
  masked, fully masked leading tiles weigh exactly 0), one call per sequence instead of the banded chunked SDPA.
  Tests: test_fa60.py window cases (windows 0..2000 incl. narrower than a key tile and wider than the sequence,
  which is bit-identical to the causal kernel). test_prefill_sdpa_window.py: on sm_60 the fa60 paths (sliding,
  and head_dim 128 full attention since E11) are held to 5e-3 instead of SDPA's 2e-3 (max 3.4e-3 seen; fa60's fp16
  partials, model-level KL ~1e-3 per its header), the fa75-forced cases skip below sm_75 (they faulted the context on
  the P100 and failed every later test); 384 passed on the P100, 392 on the RTX 2060. A/B (doc/lfm8_opt/ab_e18.sh,
  ctl EXL3_FA60_SWA=0 / fa60 / ctl, t/s): 512 3194 / 3257 / 3078, 2048 3546 / 3679 / 3504, 4096 3596 / 3738 / 3532,
  8192 3431 / 3571 / 3377, 16384 3148 / 3264 / 3098 (+3.9 / +4.4 / +4.9 / +4.9 / +4.5 %). Prefill ppl 11.03115 ->
  11.03210 (4096 tokens), top-1 4082/4096, needle 6/6.
- **E19 ACCEPTED for LFM3 only: 64-element fp16 partials in the hfma2 prefill GEMM** (labelled numerics change, user
  approved; `KS` template parameter: the fp16 accumulators open / flush per pair of 32-element K tiles). Microbench
  (pf_gemm_bench.sh --check, M = 4096): gate/up 79.6 -> 83.5 % of the HFMA2 peak, down 75.9 -> 76.9, conv in 79.1 ->
  83.3, 2048 x 2048 74.9 -> 77.7, k/v 69.7 -> 74.0; rel RMS vs fp32 8.5e-4 -> 1.19e-3. A/B (doc/lfm8_opt/ab_e19.sh, ctl /
  K64 / ctl, t/s): 512 3384 / 3461 / 3347, 2048 3872 / 3948 / 3794, 4096 3894 / 3993 / 3854, 8192 3712 / 3801 / 3675,
  16384 3399 / 3469 / 3359 (+2.7..3.1 %). Prefill ppl 11.03210 -> 11.02903, top-1 4078/4096, needle 6/6.
  Scope: the GEMM serves every model on the P100, so the 64-element form is a per-call flag (`hgemm_recon(..., k64)`)
  that LinearEXL3 sets from `config.prefill_gemm_k64`, True in Lfm3Config only (Lfm3VlConfig resets it: the vision
  tower is unmeasured). Qwen3.6 / QNF / Ornith keep the 32-element contract bit for bit
  (tests/kernels/gemm/test_hgemm_hfma2.py::test_hfma2_k64_flag, 53 passed). `EXL3_HGEMM_HFMA2_K64` = 0 / 1 forces it
  off / on for every model. Default run re-checked: LFM3 prefill ppl 11.02903.

### Round 9 — native decode attention at long context (2026-11-23)
- E20 ACCEPTED: 8 warps per CTA instead of 4 (same splits, so the combine is unchanged; `EXL3_ATTN_NATIVE_WARPS=4`
  restores 4). P100 4 / 8 / 4 warps (doc/lfm8_opt/ab_dec.sh): 512 143.3 / 143.1 / 143.1, 8192 138.4 / 139.4 / 138.0,
  32768 125.2 / 128.1 / 125.3 (+2.3 % at 32k). Per call at 32k ~206 -> ~156 us. test_decode_native 72 passed (both).
- E21 REJECTED (code removed): the next tile's page lookup, key words and scales loaded before this tile's P x V.
  8 warps, PF off / on / off: 512 143.0 / 142.7 / 142.9, 8192 139.5 / 139.3 / 139.1, 32768 128.1 / 127.9 / 128.4.
- Where it stands: at 32k the attention is ~14 % of the token (~1.1 ms incl. the sliding layers' Triton split and
  the combine) against a ~0.4 ms HBM2 floor; the rest is spread (P x V shuffles and dependent loads per key, the
  scalar-FMA score loop); further work there is worth at most ~6 % at 32k.

## STATUS after rounds 7-8 (final build 68d6ba38, 2026-11-23; doc/lfm8_opt/qual_final.sh, one card at a time)
| metric | start (Nov 22) | now P100 | now RTX 2060 |
|---|---:|---:|---:|
| decode d512 / d8192 / d32768 (t/s) | P100 51.7 / 50.1 / -; 2060 117.8 / 107.2 / 90.5 | **143.5 / 138.7 / 125.8** | **132.9 / 124.5 / 103.4** |
| prefill 4096 / 16384 (t/s) | P100 2407 / 1873; 2060 4075 / 3557 | **3767 / 3278** | **6646 / 5455** |
Kernel suites (gemv / mgemv / hfma2 / mgemm sliced / native decode / fa60 / sdpa window / live split / norms): 1345
passed on the P100, 1297 on the RTX 2060. Greedy oracle 128/128 identical to the round-6/7 builds on both cards
(2060 max |dlogit| 0, P100 0.031), decode ppl 9.93979 (P100) / 9.94076 (2060), needle 3/3 at 49.0k tokens on both,
chat.py one-shot (lfm3 mode, think, recommended sampling) correct Lomuto quicksort in Go at 106 t/s (P100) / 129 t/s
(2060).
Rounds 7-8 on the P100: decode +8.3 % at 512, +13 % at 8k, +27 % at 32k over round 6; prefill +5 % (4k).
Remaining (ranked): native decode attention at ~173 GB/s per call at 32k (occupancy / latency, not HFMA2: E16);
fusions toward a megakernel (act_mul into the down GEMV input stage, the input norm into the GEMV); small prefill
kernels (~90 ms of ~1050 at 4k).

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
