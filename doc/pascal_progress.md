# Pascal (Tesla P100) fork: progress log

A running record of what was done, measured and ruled out, and what is still open. Details and methodology live in the
Pascal section of `README.md`, `doc/pascal_p100.md` and `doc/env_vars.md`; this file is the index. Add new entries at
the top of "Done", and move items between "Open ideas" and "Done" as they are measured.

Reference machine: Tesla P100 16 GB (SM 6.0, passively cooled, power cap 125 W, optionally 160 W) plus RTX 2060 6 GB
(SM 7.5). torch `cuda:0` = RTX 2060, `cuda:1` = P100; `nvidia-smi` index 0 = P100. Measured like `examples/chat.py`
(`-gs 4.0,15.0 -cs 32768 -cq 4`, MTP `-mtp -ndt 1 -dds`) with `examples/bench_chat.py`, `bench_chat_ab.sh` and
`bench_prefill.py`. P100 cooled to <= 62 C before every run, builds alternated (ABAB); it throttles at 79 C.

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
  **23.43 t/s** with `-mtp -ndt 1 -dds` (96.1% accepted). Load plus 300 tokens takes under 3 minutes.
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
sampler) the fits below are reproducible. A previous run's CPU-worker child can also still hold VRAM for a few seconds after
`os._exit`; the runner waits for no python CUDA process before loading.

**Fit and decode speed** (`bench_chat.py` where marked, otherwise the harness; t/s over 300 tokens):

| Config | GPU alloc 2060 / P100 (GiB) | decode t/s | ms/iter | CPU job (us, per layer-token) |
| :--- | ---: | ---: | ---: | ---: |
| `-mcl 18 / 20 / 22 / 23` | - | does not fit | | |
| `-mcl 24` (7 worker threads, default) | 2.91 / 13.91 | 18.40-19.44 (`bench_chat` 19.44) | 51.9-54.3 | 620 (14.8 GB/s) |
| `-mcl 24 -mct 4` | same | 15.68 | 63.7 | 1039 (8.7 GB/s) |
| `-mcl 24 -mct 8` | same | 19.86 | 50.4 | 562 (16.3 GB/s) |
| `-mcl 24 -mct 15` (SMT siblings) | same | 18.00 | 55.5 | 783 (13.2 GB/s) |
| `-mcs 256` (= 24 x 512 experts on the CPU, warm hot/cold profile) | 3.20 / 13.95 | **22.53** | 44.6 | - |
| `-mcs 288`, cold (no expert profile yet) / warm | 2.48 / 13.05 | 20.88 / 22.65 | 48.2 / 44.4 | 360 / 179 |
| `-mcs 320` (profile of one earlier run) | 2.54 / 11.50 | 20.09 | 50.0 | 400 |
| `-mcs 288 -mcpa` (4 promote spares per layer, warm) | 2.95 / 12.82 | 21.57 (one load also failed to fit) | 46.4 | - |
| MTP `-mtp -ndt 1 -dds`, `-mcl 24` | 3.90 / 14.12 | `bench_chat` **25.78** (96.1%), harness 24.58 / 21.33 profiled | - | 1077 (m = 2) |
| MTP, `-mcs 256` (warm) | 3.81 / 13.95 | **29.32** (94.8%) | 67.7 | 234 (m = 2) |

- `-mcl 24` is the smallest whole-layer offload that fits (23 fails on a clean GPU). The autosplit is greedy (the 2060 fills
  first, then everything goes to the P100), so the margin is decided by where the first expert-carrying layer lands.
- `-mcs` at equal VRAM (`-mcs 256`: 48 x 256 = 24 x 512 experts on the CPU, +0.33 GiB GPU alloc) is **+16-22%** over
  `-mcl 24`. The gain comes from the hot/cold placement more than from the CPU/GPU overlap: the dynamic map moves the
  frequently routed experts to the GPU slices, so with a saved profile (`~/.cache/exllamav3/expert_stats/`, written at unload)
  the CPU job shrinks from 360 to 179 us per layer-token (s288 cold vs warm). The first load of a model is the cold case.
- The CPU expert kernel is compute-bound, not DRAM-bound: it scales linearly with physical cores (4 -> 8 threads: 1039 -> 562
  us/job) at ~2 GB/s of trellis per core, a third of the 40-44 GB/s the RAM delivers; SMT siblings make it slower. `-mct 8`
  (all physical cores, the host thread shares one) is the better default on this 8-core part (+3-8%, inside the noise band).
- `-mcpa`: with `-mcs 288` it fit on one of two loads (the promote spares cost VRAM) and gave no decode gain over warm
  `-mcs 288` (21.57 vs 22.65 t/s, within noise); in prefill it gives +5-12% (below).

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
29.3 t/s), where the P100 is the bottleneck; "risk" covers numerics and effort.

| # | Candidate | Estimated gain | Risk / effort |
| :--- | :--- | :--- | :--- |
| 1 | Run `-mcs 256` (+ `-mct 8`) instead of `-mcl 24`, with a warm expert profile (ship or pre-seed it so the first load is warm) | **measured** +14-19% with MTP (24.6-25.8 -> 29.3 t/s), +16-22% without | none: configuration only |
| 2 | Gated residual kernels at bandwidth: `gr_dots_kernel2` / `gr_finalize_kernel2` read 13 MB per site at ~150 GB/s | ~6 of 9.7 ms (P100 8.0 + 2060 1.7), ~10% | low-medium: two kernels, fp32 sum order |
| 3 | P100 MoE coop kernel on 1.5 bpw experts (decode-bound, ~47 GB/s; 20.9 ms per MTP iteration) | 25-30% faster = 5-6 ms, ~9% | high effort, medium risk |
| 4 | P100 EXL3 5 bpw GEMV at m = 2 (~120 GB/s, 13.2 ms) | 2-4 ms | medium-high: already tuned on the `pascal-gemv-k5` line |
| 5 | Gated-residual tables in 8 bit (or EXL3) | halves #2's bytes again; frees ~0.6 GiB VRAM for GPU experts (fewer CPU misses) | medium: accuracy of the stream mixing |
| 6 | sm_75 K = 5 GEMV + unfused qkv/z on the 2060 (relax the K <= 4 gate in `exl3_gemv_cfg` / `use_mgemm`, sm_75 only) | measured -2.6 ms at `-mcl 24` (24 layers on the 2060), ~0.8 ms under `-mcs 256` (~7 layers) | low: kernel exists; needs a sm_75 K = 5 test and PPL |
| 7 | CPU expert kernel (AVX2, compute-bound at ~2 GB/s per core vs 40 GB/s RAM) | -5 ms or more at `-mcl 24` / cold profile; under warm `-mcs` the CPU is hidden (4.9 ms no-GPU) | high effort, medium risk |
| 8 | Load: skip or shrink the n-gram row lookups in `warmup`; move the model off the USB SSD | -35 s of ~160 s load; the disk move roughly halves the 85 s weight read | low (warmup) / user decision (disk) |
| 9 | n-gram rows: keep the hot part of the table in RAM, prefetch a token ahead | ~0.5 ms short context, ~4 ms at long context on novel text | low-medium |
| 10 | QSA at long context: 2060 indexer (189 us/layer) and sparse split | ~2 ms at 21k tokens, nothing below ~2k | medium |
| - | Ruled out: `-mct 15` (SMT siblings slower), `-mcl` < 24 (does not fit), `-ngr` (RAM), `-mcs 288 -mcpa` (no gain over warm `-mcs 288`, fit marginal) | | |

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
