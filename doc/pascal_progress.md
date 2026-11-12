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

### Qwen3.8-Flash-Next: prefill under the agentic workload (Nov 7, branch `flash-ngram-prefetch`)

Context: the user reopened Flash-Next for general agentic use, mainly code, so prefill and long context matter more than the
first-300-token decode. All runs `-gs 4.0,15.0 -cs 24576 -cq 4 -mcs 256`, P100 at 1328 MHz, 54-76 C.

- **Prefill baseline** (`examples/bench_prefill.py`): 2k 170 t/s, 8k 199, 16k 183 (spread 11-16% between repeats). A 10-load
  sweep of `EXL3_MOE_STREAM_T` (4/16/32), `EXL3_MOE_STREAM_BATCH_EXPERTS` (12/48), `EXL3_MOE_CPU_STAGE_THREADS` (2/8) and
  `-mcpa` against two defaults found **no knob that beats the default**: the first load of a session ran 175/188 t/s (cold),
  every later load, including a repeat of the default, 224-259 t/s at 4096/8192 tokens. Only `STAGE_THREADS=2` was slower
  (176/199). The 170-199 t/s baselines of the earlier measurements were cold first loads.
- **Decode by context depth** (`-mtp -ndt 3 -sdh 16384`, greedy, repository code + a request): the step cost does not grow
  with context up to 22k (93-116 ms per step); the t/s drop from 33 (short prompt, 86% drafts accepted) to 21-28 comes from
  the acceptance (48-54%) on prose-like answers, not from depth. Prefill 205 t/s at 8k, 210 at 16k, 184 at 22k (120 s);
  a second turn on the same prefix re-processed 1553 of 22033 tokens in 8 s.
- **Where the prefill time goes** (torch.profiler trace of a 4096-token prefill, 23.5 s): the two GPUs run kernels together
  for only 13.1 s; 8.2 s have no kernel on any device, in two gaps (4.5 s at the start, 3.2 s at the start of the second
  2048-token chunk). Stack sampling of the main thread puts both in `row_table.lookup` waiting on the n-gram row prefetch
  (random 62-byte reads from the USB SSD; ~2 ms per token), i.e. 34% of the prefill. Per-device kernel time (P100):
  `exl3_moe_kernel<1,..,16,true>` 3.3 s, hfma2 GEMM 2.3 s, pinned->device copies 1.1 s, hgemm/sgemm fallbacks ~1.2 s.
- **Prefetch of the next chunk's n-gram rows** (`358cbdf7`): `Job.prefill` passes the ids of the following chunk in
  `params["ple_prefetch_next"]` and `PLELayer.prefetch` queues them right behind the current chunk's, so they are staged
  while the current chunk runs (`EXL3_NGRAM_PREFETCH_AHEAD`, default on). Generated text identical. **Erratum (Opus review, checked
  against `ab_ahead.log` / `ab_ahead2.log`): these were six off-then-on pairs, always in that order, each mode on its own
  prompt (offsets 400000.. in steps of 10000), not ABBA; the pairs come from two loads and the second load re-used the first
  load's offsets (400000-450000), so about half of the "cold" prompts had their rows in the page cache already. The prefetch
  stats show `hit` unchanged at 2 and `retired` 1 in every "on" run: the staged entry for the next chunk is never the one the
  next forward asks for (the real chunks are 2048 / 1792 / 255 tokens, the prediction used 2047), so the gain, if real, is the
  page-cache warming of the rows the two chunks share, not a used prefetch. Treat the -10% as unproven (n = 6, confounded).**
  Six cold pairs
  (4096-token prompts, `ab_ahead.py`): off 22.06 / 22.11 / 22.60 / 22.42 / 23.39 / 25.09 s,
  on 19.53 / 18.80 / 18.93 / 22.22 / 22.64 / 21.27 s; **mean 22.95 -> 20.57 s (-2.4 s, -10%)**, better in 6 of 6 pairs but
  from -0.2 to -3.8 s. The wait of the second chunk goes from 2.4-3.1 s to 0.00 s in the runs that gain; in two runs the
  time saved on the wait did not show in the total (CPU contention with the expert worker is the suspect, not checked).
  The first chunk still stalls ~3.3-4 s, nothing to overlap it with.
- `EXL3_NGRAM_FADVISE_ALL=1` (WILLNEED on every row before a large gather): -12% lookup time in isolation (1.29 -> 1.14 s,
  12 trials each, inside the noise); left off. `EXL3_NGRAM_DEBUG=1` prints the wait of every large lookup.
- Ruled out: progressive first chunks (first chunk 256 or 512 tokens, doubling, so the lookup of chunk k+1 overlaps chunk k).
  Generated text identical, but 4 rounds x 3 modes of cold 4096-token prompts (distinct offsets, page table reset) give mean
  base 20.4 s, 256-first 20.3 s, 512-first 20.8 s; cold-prompt spread for one mode is 16.6-22.2 s, larger than any difference.
  Code reverted (not committed).
- `EXL3_MOE_FUSED_ROWS` sweep (P100 uses the plain value, wide tiles need cc >= 7.0), 4096-token cold prompts, 3 per load,
  two loads per value: 64 -> 269 / 234 t/s, 128 (default) -> 225 / 257, 256 -> 255 / 248 (warm-up load 226). The spread
  between two loads of the same value (up to 35 t/s) exceeds the difference between values: default kept.
- Decode GEMV, 5 bpw mul1 on the P100 (`util/bench_gemv_decode.py`, m = 1 and 4 are the same speed): 28-146 us per call,
  i.e. 112-150 GB/s of trellis bytes against ~480 GB/s achievable. 2560x2048 still takes 28 us for 3.3 MB, so a fixed cost of
  ~20 us per call (cooperative launch, in-kernel input Hadamard) dominates the small shapes; 2560x10240 is the worst
  (112 GB/s). Wave quantisation alone does not explain a 3-4x gap; next is to split fixed cost from streaming time.
- GEMV fixed cost vs streaming (k sweep at fixed n, m = 1): fixed cost ~9 us (k = 128: 9.7-10.4 us; two grid syncs plus the
  Hadamards), not 20; asymptotic streaming 170 GB/s (n = 2560, k 2048 -> 8192: 172 GB/s; n = 10240, k 2048 -> 4096: 170 GB/s).
  Same shape 4096 x 10240 at K = 2 / 3 / 4: 121 / 134 / 122 us, i.e. time per weight is constant while bytes per weight
  double: the kernel is decode/ALU bound (~350 Gweights/s = ~4.7 weights per clock per SM), not memory bound. K = 5 is
  slower again (162-226 us across runs, thermal spread). The lever is instructions per weight in the decode + HFMA2 path,
  not the grid or the launch.
- Next: count instructions per weight in the m = 1 5 bpw path (SASS / ncu if available) and look for a cheaper decode.

### Qwen3.8-Flash-Next: decode, where the time goes (Nov 7)

- GEMV 5 bpw on the P100: ~9 us fixed + decode/ALU bound (~170 GB/s, ~11.7 SASS instr per weight; time per weight constant over
  K = 2..4); MoE `coop_d` runs at the same weights/s. Step with MTP (m = 4): `coop_d` 34%, GEMV ~16%, gated residual (`gr_*`) 13%.
  gr pair on the P100 (cold tables): 35 / 41 / 49 / 57 us at R = 1..4, near the bandwidth floor at R <= 2; fusing launches would
  gain < 3%. The P100 has ~1000 kernels per step but gaps < 20 us total only ~1.7 ms; its ~26 ms/step of idle is the RTX 2060's layers
  (~15 ms, the two GPUs run in pipeline, the 2060 is slower per layer) plus ~11 ms of host work.
- Decode-time n-gram reads: one lookup per verification step, ~64 rows, 3-9 ms of the main thread (~10% of the step). The SSD
  serves random reads one at a time, ~0.27 ms each even with 64 threads (rand_read2.py), so only the number of cold rows matters.
  `posix_fadvise` 30 ms ahead cuts a cold decode lookup from 12 ms to 0.23 ms in isolation. Wired into the generator
  (`EXL3_NGRAM_ADVISE=1`: leading token after a verification, each drafted token as its pinned copy lands): lookups drop to 0.32 ms
  in the runs that use it, but the time per verification round does not improve (94.4 vs 97.1 ms inline with fadvise on the
  main thread, 97.5 vs 92.3 with the syscalls on a worker: hints land late there) and an inline advise call costs 0.6-1.9 ms.
  **Default off.** Also learned: two identical runs (same prompt, greedy, same settings) already differ in text and acceptance
  (the CPU expert placement is dynamic), so single t/s numbers move +-10%; compare ms per round over >= 6 runs.
- CPU side: `-mcs` 256 -> 352 (CPU experts 50% -> 69%) leaves decode unchanged, so the AVX2 MoE kernels are not on the decode
  critical path (the 5800X has no AVX-512; the AVX2 tier is a dedicated implementation, not scalar).

- Hybrid drafting (`-hdt N`, suffix-match continuation on top of MTP) on a copy-heavy agentic turn (a ~7000-char source excerpt in the
  prompt, "reproduce it with one function renamed", 500 greedy tokens, `-mtp -ndt 3`, `bench_rewrite.py`, one load, modes alternated,
  4 rounds): baseline 42.5 t/s (MTP alone already accepts 373/381 drafted tokens there, capped by the 3-token window);
  `-hdt 3` 41.7 (same window), `-hdt 5` 41.7 (30.8 on its first use of a new row count, 43.5-47.9 after), **`-hdt 7` 49.0 (39.6 on
  the first use, then 49.6-53.8, about +22%)**; `-hdt 12` 38.3 (more than 8 rows per verification leaves the m <= 8 GEMV path).
  Free-form chat prompts (no copying, acceptance ~70%): baseline 34.3, hdt 3 35.1, hdt 5 33.9, hdt 7 32.8 t/s, all inside the +-10% run
  noise, so no visible cost. Recommended for code-editing workloads: `-hdt 7` (8 rows per verification). A larger gain needs
  the GEMV/MoE paths to take m = 9..16 rows (EXL3_GEMV_MAX_M is 8).

### Qwen3.8-Flash-Next: long context (Nov 7)

- KV cost is small (12 attention layers x 2 KV heads x 256 dims, 4 bit: ~6-7 KB/token): 131k ~0.9 GB, 262k ~1.8 GB. The limit is
  VRAM left on the P100 after the weights: `-gs 4.0,15.0 -cs 131072` and `-gs 4.0,15.8 -cs 131072 -mcs 256` fail ("Insufficient VRAM",
  the last layer is ~160 MiB short); `-gs 5.0,14.0` / `4.5,14.5` also fail (the P100 needs >= 15 GB; CUDA device 0 is the RTX 2060).
- Loads and decodes at the same speed as the default (bench_chat, two 400-token turns): `-gs 4.0,15.8 -cs 131072 -mcs 272` 42.2 t/s,
  `-mcs 288` 44.0, `-cs 262144 -mcs 320` 44.0, `-mcs 352` 44.0 (default `-cs 24576 -mcs 256`: 39.2 / 43.1 / 44.8 over three loads).
- `-gs 4.0,15.8 -cs 262144 -mcs 320 -cq 4 -mtp -ndt 3 -sdh 16384` (bench_depth, code corpus, greedy): prefill 65k 332 s (197 t/s),
  131k 719 s (182 t/s), 250k 1429 s (175 t/s); decode 18.3 / 21.3 / 17.6 t/s with draft acceptance 58 / 56 / 51% (short chat 46.6 t/s, 82%);
  re-asking at 250k reuses the prefix (3.2 s prefill, 20.9 t/s). P100 reaches 78-79 C and 1189 MHz during the long prefills.
  Decode is flat from 65k to 250k, so the drop against short chat comes from acceptance/content and the heat, not from context length.

## Qwen3.8-27B (dense) stage 0

Nov 5, branch `dense27-stage0` (= `flash-fix-ndt` c5145bcb, same `.so`; only `bench_chat -cool` added). Model
`models/Qwen3.8-27B-exl3-4.00bpw`: 64 layers (48 GDN + 16 full attention, head_dim 256, 4 KV heads), hidden 5120, MLP
17408, 4 bpw body, 6 bpw lm_head (954 MB, 248320 x 5120), 1-layer MTP head. `-gs 4.0,15.0 -cs 32768 -cq 4`: 17 layers
plus the MTP layer on the 2060 (3.61 GiB), 47 layers plus the lm_head on the P100 (9.91 GiB), embedding on the CPU.
Harness `tmp/d27s0/prof_m.py` (one load, a fresh Generator per draft width, code prompt, 400 tokens timed, torch.profiler
window of 40 steps). Every number is labelled **greedy** or **card** (model-card sampling, non-thinking: `-temp 0.7
-topp 0.8 -topk 20 -minp 0 -presp 1.5`).

**The main result is thermal.** Cooled (P100 <= 62 C before each run) the P100 holds 1189-1252 MHz. Back-to-back runs
without cooling (as the user asked from 22:40) keep it at 78-80 C with thermal slowdown in 74-83% of the nvidia-smi samples
and a mean SM clock of 523-595 MHz (min 405). The P100 kernels get 1.9-2.7x slower (4 bpw down GEMV 189 -> 353 us at m = 1,
294 -> 789 us at m = 4); the 2060 kernels do not change. Decode t/s roughly halves. The sampler costs nothing measurable
(argmax/sample kernels 0.1 ms per step, host gaps unchanged), so greedy-cooled vs card-hot differ by heat, not by sampling.

### Decode by draft width (m = rows per verify step)

| m (config) | greedy, cooled: ms/step, tok/step, t/s, acc | card, hot (3 reps, one load): ms/step, tok/step, t/s mean (min-max), acc |
| :--- | :--- | :--- |
| 1 (no MTP) | 66.9, 1.00, 14.95 | 113.1, 1.00, 8.87 (8.49-9.60) |
| 2 (`-ndt 1`) | 80.4, 1.94, 24.13, 94.2% | 146.9, 1.89, 12.89 (12.05-13.42), 89.0% |
| 3 (`-ndt 2`) | 97.4, 2.74, 28.11, 87.0% | 213.8, 2.78, 13.01 (12.87-13.09), 89.2% |
| 4 (`-ndt 3`) | 102.7, 3.38, **32.89**, 78.7% | 232.6, 3.13, 13.48 (11.72-14.52), 70.6% |

Card-hot, two more loads of m = 3 / 4 (2 reps each, the M34 A/B below): 12.45 / 12.89 and 14.56 / 14.73 t/s, acceptance
81.6-87.5% / 77.8-78.7%. Under card sampling acceptance at ndt 3 moves by ~8 points between reps (59.9-80.6%).

`bench_chat` full replies (`-max_new 1000`, two turns per load: code "Implement a quicksort in go", then prose "Explain in
detail how the Roman Republic became an empire"), one load each, **greedy**: no MTP 15.50 / 15.15 t/s (no cooling between
turns; prose turn ended at 885 MHz); `-ndt 1 -dds` code 24.64 t/s (97.6%, 1.98 tok/step), prose 22.22 t/s (79.2%, 1.79
tok/step) with the P100 cooled to 62 C before each turn, prose only 15.64 t/s without that cooling (506 MHz at the end).
The code reply is short (317-328 tokens).

**Missing (stopped on request, not measured):** the card-sampling `bench_chat` sweep (3 loads x {no MTP, ndt 1 -dds,
ndt 2, ndt 3}, ndt 2/3 with `-dds`), so there is no card number for full code/prose replies yet; prefill at 2k / 8k / 24k
with kernel shares (harness `ctx.py` ready); decode at 8k / 24k / 64k context (`ctx.py -cs 73728`); the `-gs 2.0,15.5`
split probe; a matched A/B for `-sdh`. Nov 4 data from an older build (before MMODE 4), greedy, cooled, for reference
only: prefill 226 t/s at 2k and 4k, 223 at 8k, 204 at 16k (P100 at 80 C by the end); hot back-to-back 4k / 8k / 16k fell
to 182 / 110 / 93 t/s (thermal 59%).

### Where a step goes (greedy, cooled, torch.profiler, ms per step)

| | m = 1 | m = 2 | m = 3 | m = 4 |
| :--- | ---: | ---: | ---: | ---: |
| profiled wall (unprofiled) | 67.4 (66.9) | 80.8 (80.4) | 99.7 (97.4) | 105.8 (102.7) |
| P100 verify (47 layers + lm_head) | 45.2 | 52.2 | 65.3 | 66.2 |
| -- 4 bpw GEMV MLP gate+up / down | 16.4 / 8.9 | 19.4 / 10.0 | 22.9 / 13.7 | 23.1 / 13.8 |
| -- 4 bpw GEMV GDN qkv / z / out | 4.0 / 2.5 / 2.9 | 4.7 / 3.0 / 3.2 | 5.7 / 3.5 / 4.3 | 5.7 / 3.5 / 4.3 |
| -- 4 bpw GEMV attention q / k+v / o | 1.8 / 0.7 / 1.0 | 2.1 / 0.8 / 1.1 | 2.4 / 1.0 / 1.5 | 2.4 / 1.0 / 1.5 |
| -- lm_head (6 bpw) | 2.6 | 3.2 | 3.9 | 3.9 |
| -- GDN (conv, recurrence, gates) / attention / norms | 2.1 / 1.1 / 0.7 | 2.5 / 1.2 / 0.7 | 3.1 / 2.3 / 0.7 | 3.5 / 2.4 / 0.7 |
| 2060 verify (17 layers, 4 bpw GEMV ~95%) | 20.7 | 21.3 | 22.0 | 22.5 |
| draft head: target lm_head at m = 1 on the P100, per draft token 2.6-2.9 | - | 2.6 | 5.7 | 8.6 |
| draft MTP layer on the 2060 (+ MTP prefill of the accepted token) | - | 1.3 + 0.6 | 2.6 + 0.6 | 3.9 + 0.6 |
| recurrent rewind (P100 + 2060) | - | 0.07 | 0.2 | 0.4 |
| no GPU busy (host, syncs, CPU embedding round trips) | 1.5 | 5.9 | 3.7 | 6.5 |

- GEMV dispatch on the P100 (`exl3_gemv_kernel<4, fp32, cb 2, MMODE, cfg>`): m = 1 MMODE 0, m = 2 MMODE 2, m = 3..4
  MMODE 4. gate/up (n 17408) take the wide config (112 blocks x 256), down (k 17408) the narrow one (56 x 512); per call
  gate/up 171 / 202 / 239 / 241 us, down 136 / 152 / 207 / 208 us for m = 1 / 2 / 3 / 4. On the 2060 m >= 2 runs MMODE 1
  (MMA): flat in m (gate/up 254 us, down 170 us).
- **No jump at m = 4 on a cool GPU**: m 3 -> 4 costs +5.3 ms (one more draft lm_head +2.9, MTP layer +1.3, GDN, rewind,
  gaps); the GEMV is identical at m = 3 and 4 (MMODE 4 serves both). The expensive step is m 2 -> 3 (+17 ms: GEMV MMODE 2
  -> 4 +11 ms on the P100, one more draft step +4.4, paged attention 70 -> 154 us x12 +1 ms). The user's 161 ms/step at
  `-ndt 3` matches the hot P100 (card-hot m = 4: 230 ms/step), so it is most likely thermal throttling, not a dispatch
  path; not reproduced in a cool run.
- `EXL3_GEMV_M34=0` (MMODE 1 at m = 3..4), greedy cooled one load: m = 3 125.3 ms (21.83 t/s), m = 4 132.3 ms (25.52 t/s)
  against 97.4 / 102.7. Card-hot ABAB (2 loads each): on 211.0 / 213.1 ms (m = 3), 231.1 / 229.9 (m = 4); off 263.0 /
  273.5, 286.3 / 285.8. **MMODE 4 is worth +24-29% at m = 3..4 on this model** (larger than on Flash-Next).
- `-sdh 16384`, card-hot, one load (not matched, indicative): m = 2 134.9 ms (13.96 t/s, 88.2%), m = 3 198.2 ms (13.71
  t/s, 86.1%) vs 146.9 / 211-214 without: -12 to -15 ms per step hot, consistent with removing the m = 1 draft lm_heads.

**Floors.** Bytes per verify: P100 35 GDN x 192.5 MB + 12 attention x 186.3 MB + lm_head 954 MB = 9.93 GB; 2060 3.25 GB.
At the P100's measured read peak (~450-520 GB/s, Flash-Next microbench) and an assumed ~300 GB/s on the 2060 (spec 336,
not measured here) that is 20.7 + 10.8 = **31.5 ms** per verify at any m <= 4 (the two GPUs run serially, so the floors add),
plus 2.0 ms per draft lm_head. Measured: P100 GEMV at m = 1 runs at 236-262 GB/s, at m = 4 at 150-187 GB/s (decode/ALU
bound, as on Flash-Next); the 2060 at 155-180 GB/s at every m. GEMV at m = 4 is 63 ms on the P100 against a 20.7 ms floor.
Per layer at m = 1 the P100 is faster (0.96 vs 1.22 ms), at m = 4 they are equal (1.32 ms), so moving layers to the P100
(`-gs 2.0,15.5`, not measured) would help m = 1..2 only (~2.6 ms at m = 1).

### Ranked candidates (estimates; base greedy-cooled `-ndt 3` 102.7 ms/step, 3.38 tok/step, 32.9 t/s)

| # | Candidate | Est. gain | Risk / effort |
| :--- | :--- | :--- | :--- |
| 1 | Keep the P100 below 79 C (airflow / fan on the passive card; or a lower power cap, untested whether the cap clock beats the thermal 500-600 MHz) | hot card -> cool: 230 -> ~103 ms/step, ~13.5 -> ~30 t/s (card) | hardware/config, no code |
| 2 | `-sdh 16384` (removes the per-draft-token full lm_head on the P100, 2.6-2.9 ms each, and the CPU embedding round trip) | ~8 ms/step at ndt 3 cool (-8%, ~35.5 t/s); hot probe -12..15 ms | config; needs matched A/B and acceptance check (Nov 4: cold token profile cost acceptance) |
| 3 | P100 4 bpw GEMV at m = 3..4 (MMODE 4) closer to the m = 2 cost per call (gate/up 240 -> ~205, down 208 -> ~155, GDN/attn proj ~-20%) | ~10-12 ms/step (~-11%) | kernel work, medium-high |
| 4 | P100 4 bpw GEMV efficiency at every m (236-262 GB/s at m = 1 vs ~480 peak; trellis decode bound) | up to ~20 ms at m = 4 if 1.5x | high |
| 5 | 2060 GEMV at ~170 GB/s (half its bandwidth), 22 ms at every m | ~5-8 ms | medium |
| 6 | Host gaps / syncs in the MTP loop (no-GPU 4-6.5 ms profiled): CPU embedding lookup per draft step, per-step syncs | 2-4 ms | low-medium |
| 7 | Paged attention decode at m >= 3 (154 vs 70 us x12) and GDN recurrence at m = 4 (63 us x35) | ~1.5 ms | low-medium |
| 8 | Split `-gs 2.0,15.5` | ~2.6 ms at m = 1, ~0 at m = 4 | config, unmeasured |

Carries over from Flash-Next: GEMV MMODE 4 (in use, +24-29% here), `-sdh` (the MTP draft uses the target lm_head),
`bench_chat` tokens/step. Does not apply: MoE coop / dual kernels, gated-residual `GR_MIX3`, CPU experts and `-mcs`, n-gram
tables, QSA, the K = 5 GEMV work (this model is 4 bpw with a 6 bpw head; the 6 bpw head lives on the P100 so the sm_75
K <= 4 gate does not matter). Harness pitfalls: `prof_m.py` overwrites `trace_m*.json` per run (the greedy traces above
were overwritten by the card-hot ones; the tables come from the greedy analysis made before).

### 27B prefill profile

Nov 6, branch `prefill27-profile` (= `dense27-stage0`, same Nov 5 `.so` with MMODE 4, no code change). `-gs 4.0,15.0 -cs
32768 -cq 4`, no MTP. Cold prompt (repo-source corpus) through the `Generator` (`max_chunk_size` 2048, as chat/Tabby), one
token out, greedy. torch.profiler CUDA kernel time per kernel group, % of the GPU span (both cards, which run serially).
Harness `tmp/p27/`: `pf27.py` (timed and profiled generator prefill), `pfchunk.py` (one 2048-token chunk on a P-token prefix
with the generator's paged params; attention cost depends only on lengths, GDN not on P), `an27.py` / `comb27.py` (tables).

**24k could not run end to end within the 5-minute rule**: the P100 reaches 79 C within ~40 s of prefill and then sits in
thermal slowdown (405 MHz in 86 of 162 samples of the 24k attempt, which hit the 330 s timeout). The 24k column is built
from 12 chunks: per-chunk non-attention cost (constant) plus attention fitted linearly in the prefix. "cool" uses the 2k/8k
generator traces (P100 65-78 C, nvidia-smi 2 s log of that run: mean SM clock 1128 MHz, thermal slowdown in 4 of 96
samples); "hot" uses chunks measured at P = 0, 0, 12288, 22528, 22528 with the P100 at 78-79 C (log: mean 709 MHz,
thermal slowdown in 54 of 123 samples; the before/after snapshots of ~1190 MHz in the log lines are between throttle
dips; every P100 kernel 2.3-2.6x slower than cool, the 2060 unchanged). Hot is the sustained state for long agent prompts
on this box. The cool attention slope comes from the 8k profiled run, which warmed to 75-78 C, so the cool 24k attention
share is more likely an upper bound. The EXL3 tile GEMM (`exl3_gemm`) is 0% at every length: at >= 64 rows everything goes
through reconstruct + hfma2 (P100) or reconstruct + cuBLAS (2060).

| kernel group | 2k | 8k | 24k cool (est.) | 24k hot (meas. chunks) |
| :--- | ---: | ---: | ---: | ---: |
| GEMM hfma2 (P100, `hfma2::gemm_kernel<*, 64>`) | 72.2% | 72.5% | 63.2% | 76.7% |
| GEMM cuBLAS (2060, tensor cores) | 10.7% | 9.8% | 9.4% | 4.6% |
| GEMM reconstruct (`reconstruct(_had)_kernel`, both) | 6.3% | 5.0% | 5.5% | 3.9% |
| attention core (fa60 P100 / fa75 2060) | 1.2% | 4.5% | 13.5% | 9.1% |
| GDN h (`..._fwd_kernel_h_blockdim64`) | 2.2% | 2.1% | 1.9% | 1.6% |
| GDN o (`chunk_fwd_kernel_o`) | 2.2% | 2.1% | 1.9% | 1.7% |
| GDN kkt_solve / recompute_w_u / conv / gates+norm | 0.9 / 0.4 / 0.3 / 0.2% | 0.8 / 0.4 / 0.3 / 0.1% | 0.8 / 0.4 / 0.2 / 0.2% | 0.6 / 0.3 / 0.2 / 0.1% |
| GDN l2norm / cumsum | < 0.05% | < 0.05% | < 0.05% | < 0.05% |
| RMSNorm / act_mul / hadamard / rope | 0.4 / 0.3 / 0.3 / 0.1% | 0.3 / 0.3 / 0.1 / 0.1% | 0.3 / 0.3 / 0.3 / 0.1% | 0.2 / 0.3 / 0.0 / 0.1% |
| GEMV, other elementwise, memcpy | 1.5% | 1.0% | 1.3% | 0.4% |
| no GPU busy (gaps between kernels, host) | 0.7% | 0.5% | 0.6% | 0.2% |
| **all GEMM** | **89.3%** | **87.4%** | **78.1%** | **85.1%** |
| **GDN total / GDN h+o** | 6.2 / 4.4% | 5.9 / 4.3% | 5.4 / 3.8% | 4.6 / 3.4% |
| **GDN + attention** | 7.3% | 10.4% | 18.9% | 13.7% |

Prefill speed (unprofiled, `time_prefill`): **2k 228.4 t/s** (P100 65 -> 67 C), **8k 221.3 t/s** (68 -> 75 C);
profiled 225.6 / 197.1 (8k profiled ran hotter, 75 -> 78 C). **24k: ~196 t/s cool (est., 125 s), ~88 t/s hot** (279 s,
from the chunks; consistent with the 24k attempt not finishing in ~250 s of prefill). Single hot chunks: 94-98 t/s at P = 0,
92 at P = 12288, 89 at P = 22528. 64k not measured (cool ~11 min end to end, beyond the run limit).

Per card (share of that card's own kernel time):

| run | P100 busy: GEMM / attention / GDN / rest | 2060 busy: GEMM / attention / GDN / rest |
| :--- | :--- | :--- |
| 2k (generator, cool) | 7624 ms: 92.4 / 1.3 / 4.3 / 2.0% | 1428 ms: 78.2 / 0.8 / 16.4 / 4.6% |
| 8k (generator) | 35434 ms: 89.9 / 4.8 / 4.1 / 1.2% | 5895 ms: 76.9 / 2.8 / 16.9 / 3.4% |
| chunk P = 0 (hot) | 19494 ms: 93.8 / 1.3 / 4.1 / 0.9% | 1430 ms: 78.9 / 0.7 / 17.4 / 3.0% |
| chunk P = 22528 (hot) | 21371 ms: 76.3 / 19.5 / 3.3 / 0.8% | 1679 ms: 68.4 / 13.8 / 15.0 / 2.8% |

On the 2060 (tensor-core cuBLAS GEMMs) GDN is 15-17% of its time, but the 2060 is only 7-16% of the prefill, so GDN
there is ~1-3% of the whole.

Findings:
- **Prefill is GEMM-bound on the P100.** At 2k cool the P100 is busy 7.65 s, the 2060 1.44 s (47 vs 17 layers: 163 vs
  84 ms per layer per 2048 tokens). The hfma2 GEMM does ~73 TFLOP of P100 linears in 6.6 s = **~11 TFLOPs, ~64% of the
  fp16 HFMA2 peak** (~17.2 TFLOPs at 1.2 GHz), consistent with the 50% HFMA2 instruction share in its SASS. `pick_tile`
  only ever selects the 64-row tile (`MIN_BLOCKS_64` gate, else cuBLAS); the 128-row variant is not used on this path.
- **GDN is small**: all FLA Triton kernels together 4.6-6.2% at every length, h + o 3.4-4.4%, below the 8% gate for
  Triton launch tuning, so the register-spill fix (step 2) was not attempted: removing the spills completely would save at
  most ~2-3% of prefill. kkt/recompute/conv/l2norm/cumsum are < 1% each.
- **Attention grows to 9-14% at 24k** (fa60 on the P100, per 2048-token chunk: 96 ms at P = 0 + ~107 ms per 1000 prefix tokens cool; hot fit, both cards, 279 ms +
  163 ms per 1000 prefix tokens per chunk). GDN + attention at 24k is 13.7% hot (measured chunks, below the 15% stop
  line) and 18.9% cool (estimate, above it, likely an upper bound): the stop rule depends on the thermal state. Step 2
  was skipped because of the separate gate (GDN h + o < 8% at every length), not because of the stop rule.
- **The two cards never overlap**: both busy at the same time 1.5 ms at 2k, 19 ms at 8k. The 2060 holds the first layers
  (cuda:0 in `-gs` order), so it could start chunk N + 1 while the P100 runs chunk N (each layer's KV pages and
  recurrent state stay on its own card). Upper bound: the 2060's share of the span, ~16% at 2k cool, ~7% hot; after
  that, `-gs` could be rebalanced towards the 2060 (84 vs 163 ms per layer per chunk, VRAM permitting). Estimate only.
- Gaps between kernels are < 1% (host overhead is not a factor in prefill); reconstruct (weight dequant before every
  GEMM, per 2048-token chunk) is 4-6%.
- **Heat dominates long prompts**: the same work runs 2.2x slower once the P100 is at 79 C (24k 196 -> 88 t/s).

Next steps (estimates, not measured): (1) P100 cooling / airflow is worth more than any kernel at >= 8k (up to 2.2x);
(2) hfma2 GEMM efficiency: 64% -> ~80% of the HFMA2 peak would be ~-15-20% prefill time (the 72-77% group): raise the
HFMA2 share of the inner loop (larger per-thread tile / fewer LDS per HFMA2, 128-row tile for M >= 2048), check the fp16
partial / fp32 add overhead; (3) pipeline the two cards across chunks (2060 on chunk N + 1 while the P100 runs chunk N; <= 7-16%), then rebalance
`-gs`; (4) fewer reconstructs: a 4096 chunk halves reconstruct (~-2-3%), costs scratch memory; (5) fa60 at long prefix
(9-14% at 24k); (6) GDN Triton tuning last (<= 3%).

### hfma2 prefill GEMM

Nov 6, branch `hfma2-prefill-gemm` (from `prefill27-profile`). Goal: raise the efficiency of `hfma2::gemm_kernel`
(63-77% of 27B prefill) without changing the numerics. Result: **new kernel, bit-identical output, +7-8% GEMM
throughput on the 27B prefill shapes** (time-weighted over a 4-layer group: **+8.3% at M = 2048, +7.3% at M = 4096**).

Harness `tests/kernels/gemm/hfma2_bench/`: `vars.cu` (variants side by side, JIT-built with the extension's nvcc
flags; v0 = the old kernel verbatim), `bench.py` (P100 only, ABAB per launch, `--check` = rel RMS vs fp32 matmul and
bit equality vs the first variant), `weigh.py` (time-weighted speedup), `loopmix.py` (SASS op mix of the whole
function and of its main loop), `pf27ab.py` (end-to-end, see below). **Clock normalisation**: the P100 throttles
within seconds (79 C, 405-1300 MHz from one sample to the next), so a post-hoc nvidia-smi clock does not describe a
30-150 ms GEMM (it gave "101% of peak"). Each GEMM launch is bracketed by a register-only HFMA2 kernel (`calib`,
~3.6 ms); efficiency = GEMM rate / mean calib rate. Calib reaches 18.4 TFLOP/s at 1328 MHz = 96.6% of the nominal
HFMA2 peak (56 x 64 x 4 flop/clk); "TFLOP/s @1328" below = efficiency x 19.03 (nominal peak at 1328 MHz), i.e.
what the kernel does on a cool card. Outputs in the module dtype (gate/up, q, k/v fp16; down, GDN qkv/z, o fp32).

Final run (cool card: 62-66 C at M = 2048, median clock 1227 MHz; 65-70 C at M = 4096, 1189 MHz), 3 reps x 4 launches,
efficiency vs calib (per-rep spread +-1%):

| M | shape (K x N) | old 64-row | new 64-row | x | old 128-row | new 128-row |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 2048 | gate/up 5120 x 17408 | 75.4% (14.4 T) | 80.6% (15.3 T) | 1.066 | 70.7% | 77.1% |
| 2048 | down 17408 x 5120 | 72.1% (13.7 T) | 78.6% (15.0 T) | 1.091 | 69.9% | 75.6% |
| 2048 | GDN qkv 5120 x 10240 | 72.5% (13.8 T) | 78.6% (15.0 T) | 1.084 | 71.4% | 77.7% |
| 2048 | GDN z 5120 x 6144 | 71.8% (13.7 T) | 78.6% (15.0 T) | 1.089 | 71.3% | 76.8% |
| 2048 | GDN out / attn o 6144 x 5120 | 70.6% (13.4 T) | 79.2% (15.1 T) | 1.121 | 69.9% | 75.1% |
| 2048 | attn q 5120 x 12288 | 73.7% (14.0 T) | 79.4% (15.1 T) | 1.071 | 70.1% | 76.5% |
| 2048 | attn k / v 5120 x 1024 | 57.8% (11.0 T) | 71.4% (13.6 T) | 1.234 | 54.5% | 58.7% |
| 4096 | gate/up | 77.5% (14.8 T) | 82.0% (15.6 T) | 1.065 | 73.1% | 77.2% |
| 4096 | down | 74.9% (14.3 T) | 81.6% (15.5 T) | 1.075 | 74.3% | 80.3% |
| 4096 | GDN qkv | 74.2% (14.1 T) | 81.0% (15.4 T) | 1.092 | 72.1% | 77.6% |
| 4096 | GDN z | 72.5% (13.8 T) | 78.9% (15.0 T) | 1.088 | 71.3% | 77.1% |
| 4096 | GDN out / attn o | 73.0% (13.9 T) | 78.8% (15.0 T) | 1.081 | 71.5% | 77.8% |
| 4096 | attn q | 75.6% (14.4 T) | 80.5% (15.3 T) | 1.066 | 72.1% | 76.9% |
| 4096 | attn k / v | 68.7% (13.1 T) | 69.8% (13.3 T) | 1.016 | 65.4% | 70.6% |

(T = TFLOP/s at 1328 MHz equivalent. Raw TFLOP/s at the run's 1100-1328 MHz, e.g. GDN out / attn o at M = 2048: old
11.6-12.8, new 13.0-14.4; k / v: old 9.6-10.7, new 12.0-13.2.) The doc's earlier "64% of peak" assumed 1.2 GHz; per clock the old kernel was already at 72-78% of calib.

What was tried, in the task's order (each ABAB vs v0, gate/up + down + o at M = 2048 first):
- **(a) 128-row tile**: it is never picked by design (`pick_tile` returns 64 or 0; the 64-row tile was measured within
  2% of it on large shapes). On the 27B shapes it is **3-6% slower** than the 64-row tile for both the old and the new
  kernel (one 256-thread block per SM vs 2-3 128-thread blocks). Not enabled.
- **(b) more HFMA2 per LDS (v1)**: the old kernel stored A duplicated into half2 and loaded per k two LDS.128 of A and
  two LDS.64 of B for 32 HFMA2. v1 stores A transposed as plain halves (two rows packed per 32-bit STS, row pairs
  XOR-swizzled by k / 8: conflict-free stores and aligned LDS.128 reads) and lets the HFMA2 operand selector
  (`R.H0_H0` / `R.H1_H1`) do the broadcast: **one LDS.128 of A (8 rows) + one LDS.128 of B (8 columns) per 32 HFMA2**,
  thread tile rows ty*8 + {0..7}, cols tx*8 + {0..7}. Same arithmetic per element. +4-10%. A larger per-thread tile
  (8 x 16) does not fit: 64 half2 + 128 fp32 accumulators alone are 192 of the 255 registers.
- **3 blocks per SM (v2)**: v1 has 156 registers (3 blocks fit at <= 168) but 24 KB of shared memory (2 stages); one
  shared stage (12 KB, a second barrier per tile, global loads still prefetched into registers) +
  `__launch_bounds__(128, 3)`: ~ +1-2% over v1.
- **(c) address / loop overhead (v3/v4)**: K loop unrolled by two tiles, pointer-increment global loads: in-loop
  non-HFMA2 ALU ops per tile ~82 -> ~40. v4 (= v2 + this) is the new kernel's default (`STAGES = 1`). +1-2% over v2.
- **(d) double buffering**: was already there (two shared stages + register prefetch, one barrier per tile). The new
  kernel keeps both forms as a template: one stage / 3 blocks per SM by default, two stages / 2 blocks per SM (= v3)
  when the grid fills its last wave clearly better at 112 resident blocks than at 168 (`pick_stages`: fill ratio >
  1.3). Without it v4 lost on the 320-row tail chunk of the 27B down projection (200 blocks = 1.19 waves: 0.89x of the
  old kernel; v3 1.10x).
- **Grouped raster** (GROUP_M = 8 / 16, B stripes shared across row stripes in L2, as the f16acc kernel): no gain
  (-1 to +1%). Not kept.
- **(e) instruction mix** (`cuobjdump -sass -arch sm_60` of the built `.so`, `loopmix.py`; fp16-out 64-row kernel):

| | whole function | main loop | loop ALU ops besides HFMA2 per 1024 HFMA2 |
| :--- | ---: | ---: | ---: |
| old (JIT build, same flags as the extension) | 2022 instr, **HFMA2 50.6%** | 1387, 73.8% (LDS 128, HADD2.F32 64, FADD 64, XMAD 29, STS 20, ...) | ~209 |
| new (extension `.so`, `STAGES = 1`) | 4368 instr, **HFMA2 70.3%** | 2567 per 2 tiles, 80.0% (LDS 128, HADD2.F32 128, FADD 128, ...) | ~165 |

  The whole-function share is mostly prologue / epilogue that runs once (the 2x unroll alone moves it past 65%); the
  loop share is what counts. LDS / STS / LDG go to the load-store unit; what competes with HFMA2 for issue is the
  32-term flush (HADD2.F32 + FADD per output element per 32 k: 128 per 1024 HFMA2, fixed by the contract) plus ~40
  integer ops: issue ceiling ~86% of calib, the new kernel reaches 79-82% on the large shapes. The rest of the headroom
  needs a shorter flush, i.e. a different numerical contract (KSLICE > 32), which was out of scope.

Smaller grids and the `pick_tile` reference shapes (B x M x K x N; same method, card at 74-77 C / ~1300 MHz;
`ext` = the built extension with `pick_stages`), ratio vs the old kernel:

| shape | old | ext (stages) | v3 (2 st.) | v4 (1 st.) |
| :--- | ---: | ---: | ---: | ---: |
| 1 x 64 x 5120 x 17408 | 46.7% | 62.5% x1.35 (1) | x1.13 | x1.34 |
| 1 x 320 x 5120 x 17408 | 68.7% | 71.9% x1.04 (1) | x1.06 | x1.05 |
| 1 x 64 x 17408 x 5120 | 26.9% | 29.2% x1.09 (2) | x1.09 | x1.08 |
| 1 x 192 x 17408 x 5120 | 40.7% | 58.0% x1.43 (1) | x1.21 | x1.42 |
| 1 x 320 x 17408 x 5120 | 65.6% | 72.3% x1.10 (2) | x1.10 | x0.89 |
| 1 x 832 x 5120 x 1024 | 66.9% | 71.8% x1.07 (2) | x1.07 | x1.05 |
| 1 x 2048 x 512 x 2048 | 64.5% | 68.4% x1.06 (1) | x1.04 | x1.06 |
| 1 x 645 x 2048 x 4096 | 57.4% | 57.6% x1.00 (1) | x1.06 | x1.01 |
| 8 x 150 x 512 x 2048 (batched) | 47.3% | 51.0% x1.08 (1) | x1.05 | x1.07 |
| 1 x 2048 x 5120 x 17408 | 75.0% | 80.8% x1.06 (1) | x1.05 | x1.07 |
| 1 x 4096 x 5120 x 1024 | 68.7% | 69.3% x1.01 (1) | x1.06 | x1.02 |

The new kernel is never slower than the old one; v3 would be 4-5% better on two 512/352-block grids where the fill
rule keeps one stage (2048 x 512 x 2048, also 512 blocks, prefers one stage), left as is.

**Sustained hot throughput** (`hot.py`: gate/up M = 2048, back-to-back launches, ABAB blocks, wall clock): the card
power-caps at ~150-157 W at ~1050-1140 MHz, then thermal-throttles at 79 C. Cool / power-capped, 20 s blocks: old 11.53,
new 12.22, old 11.08, new 11.70 TFLOP/s (+6%, both at ~150 W and similar clocks). In the 79 C steady state, 11 s blocks
(blocks 2-11 of the second run): old 4.41 / 3.91 / 3.70 / 3.57 / 3.67, new 4.70 / 4.05 / 4.20 / 4.08 / 4.16 TFLOP/s:
**new ahead in every pair, +11% on the mean** (the new kernel does not throttle harder; the per-clock gain survives).

Correctness: every variant is **bit-identical** to the old kernel on all 7 shapes at M = 2048 (fp16 and fp32
outputs, both tiles; `bench.py --check`), rel RMS vs fp32 `torch.matmul` 8.4-8.7e-4 (same as before, limit 1e-3).
`tests/kernels/gemm/test_hgemm_hfma2.py`: new `test_hfma2_matches_fp32_matmul` (the seven 27B K x N at M = 130, plus odd
K tile counts K = 96 / 160 and M = 3 / 67, both tiles, vs fp32 matmul at rel RMS 1e-3); 45 passed (test_hgemm_f16acc
skipped: sm80+ only).

End-to-end (Qwen3.8-27B 4 bpw, `-gs 4.0,15.0 -cs 32768 -cq 4`, 2048-token cold prompt through the Generator):
**not conclusive**. Model load takes 75-123 s (its dummy forward runs on the throttled card), which leaves one or
two prefills per 3-minute run, and the P100 sat at 79 C in thermal slowdown for the whole afternoon. `pf27ab.py`
loads the model once and routes `ext.hgemm_recon` (P100, covered shapes) to v0 or v4 per prompt. Runs: A1 (old `.so`,
separate process) 219 / 202 / 168 t/s as it heated; in-process pair A then B: 192.4 (old, 1189 MHz) then 167.2 t/s
(new, second and hotter, 1189 -> 974 MHz); four new-kernel runs (two separate-process with the new `.so`, two
in-process B-first) started at 79 C / 405-1037 MHz and ran 95-105 t/s, timing out before a second prompt. These
numbers track the clock, not the kernel (the hot ABAB above shows the kernel itself ahead at the same thermal state). Expected from the microbenchmark: GEMM
hfma2 is ~72% of prefill, so +8% on it is ~ -5% prefill time (228 -> ~240 t/s cool at 2k). To measure: a cool card,
`pf27ab.py -pairs 1 -start 0` and `-start 1` alternately.

Recommendation: keep the new kernel (no numerical change, faster on every 27B shape, on the small / batched reference
grids and on both tiles, also hot; the 64-row tile remains the right pick). Further GEMM gains on the P100 need a contract change (longer fp16 partials, e.g. 64 terms:
halves the flush, ceiling ~91%) or better cooling, which is worth far more at >= 8k (2.2x).

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

## Qwen3.8-Flash-Next: prefill expert-stream DMA and the blocking host readbacks (Nov 8, branch `flash-prefill-gpu`)

Task: cut the prefill's GPU/kernel time. Config for every run here: `-gs 4.0,15.0 -cs 24576 -cq 4 -mcs 256`, cold
corpus chunks from `util/bench_flash.py`, one load per measurement, P100 temperature/clock logged before and after each
run. Hardware relevant to this section: the P100 trains at PCIe Gen3 **x4** (2.75-2.83 GB/s measured on this work), the
RTX 2060 at x16 (12.96-12.98 GB/s measured).

**What was changed** (`e2bca597`, Python only, `exllamav3/model/moe_cpu_host.py`): the streamed-expert VRAM-ring DMA, the
pinned-arena (`-mcpa`) DMA, the CPU-tail per-job issue and readback copies and the pinned->device bandwidth probe now
call `ext.cuda_memcpy_async` (a raw `cudaMemcpyAsync` on the current stream) instead of torch's `copy_`. Motivation:
`cuda_host.h` documents that torch does not always recognise `cudaHostRegister`'ed memory as pinned and may stage such
transfers through a pageable buffer, and the profiler attributed 13.3 s of CPU time to `cudaMemcpyAsync` on one 4096-token
prefill. The change is correct by construction: same bytes, same stream order, and the flag waits/writes that gate ring and
slot reuse stay ordered behind the copies on the same stream; only contiguous same-size views take the raw path (anything
else keeps `copy_`). `EXL3_MOE_ASYNC_DMA=0` restores the old path (read once at import).

**The traces say the 13.3 s is not the expert stream.** Method: `bench_flash.py -run prefill -lengths 4096 8192 -repeat 2
-profile_prefill -record_shapes -trace <prefix>`, the torch.profiler chrome trace parsed per device, per memcpy kind, with
each `cudaMemcpyAsync`/event runtime call correlated to its GPU activity. On the profiled 4096-token run (s1 below, 21.88 s
window): `cudaMemcpyAsync` 13.25 s of CPU time, of which **13.16 s is 355 `aten::copy_` calls to
`Memcpy DtoH (Device -> Pageable)`** - 0.03 s of actual GPU time, one 1.57 MB call and the rest a median of **2.1 kB**,
each taking 20-150 ms because the copy only runs when the compute stream reaches it (plus 10 `cudaStreamSynchronize` calls,
0.39 s, and 2 `cudaDeviceSynchronize`, 0.03 s). Those are host-side planning values read back in the middle of the layer
loop, and they are the real serialization: the host blocks ~13 s while the P100 is busy only ~10.5 s. The expert DMA is
**not** the blocker and was already asynchronous before this work: the handoff segment and the arena are pinned
(`is_pinned() == True` on the `wviews`/`arena_views` views), a 32 MiB pinned H2D `copy_` returns after 0.05 ms of CPU
(0.03 ms with `ext.cuda_memcpy_async`), DMA 2.59 ms on the 2060 (12.98 GB/s), and in the traces the pinned->device copies
carry negligible CPU time in both builds. **The async-copy patch is therefore performance-neutral in the warm-placement
case; it is kept as the correct/cleaner path (and it does remove the host-side staging for the cold case's large handoff
DMA). No t/s win is claimed for it.**

**The s1/s2 t/s delta is a placement confound, not a copy-path result.** Two loads were compared with
`EXL3_MOE_ASYNC_DMA=1/0`, and the load logs differ: s1 was cold (`CPU split experts (worker, dynamic): ... [256..512) of
512`), s2 loaded `placement from saved profile` (warm). Cause: the expert-stats file key is
`sha1(abspath(model_dir) | num_experts)`, and a run started from a worktree resolves `models/...` under the worktree, so
it hashes to a different file than a run from `/home/nicola/exllamav3`; the restore step wrote the wrong file and the first
load of the session ran cold. The two placements stream very different amounts of expert weights:

| | window | P100 kernels | P100 copies | 2060 kernels | 2060 copies | expert H2D bytes | P100 / 2060 H2D rate |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s1 (cold) | 21.876 s | 10.493 s | 2.639 s | 1.459 s | 0.222 s | **9.58 GB** (6.79 / 2.79) | 2.75 / 12.97 GB/s |
| s2 (warm) | 16.745 s | 9.974 s | 0.526 s | 1.375 s | 0.115 s | **2.42 GB** (1.03 / 1.39) | 2.83 / 12.96 GB/s |

Both sit at the link rate, and the byte count is decided by placement: cold, the CPU holds arbitrary experts, every CPU
expert clears the stream threshold and 9.58 GB crosses the links; warm, the CPU holds the cold experts, only 2.42 GB is
streamed and the CPU tail does the rest. Measured t/s of the two loads (not an A/B, listed with thermals for the record):
4096 tokens 187.6 / 200.4 t/s (s1, 54-60 C, 1328 MHz) vs 245.2 / 261.9 t/s (s2, 67-73 C, 1328 MHz); 8192 tokens 240.0 /
249.3 (s1, 60-69 C) vs 264.2 / 275.4 (s2, 72-79 C, last run 1303 MHz). The documented cold->warm placement gain is 10-20%
on this model by itself, and s2 ran hotter, so the +27-31% at 4096 cannot be credited to the copy path. Lesson for future
A/B on this model: make every side load from the same cwd/model path (or pass `EXL3_MOE_CPU_SPLIT_STATS`), then verify the
`placement from saved profile` line in both logs - otherwise the expert placement differs and the t/s comparison is void.

**Where the rest of the prefill goes** (s1 trace, cold 4096; P100 kernels 10.493 s of the 21.88 s window):
`exl3_moe_kernel<1,128,2,16,true>` 2.429 s / 163 calls (14.9 ms each), `<1,128,2,32,true>` 1.936 s / 643 calls (3.0 ms),
`hfma2::gemm_kernel` 0.91 + 0.89 s (571 + 911 calls), `aten::mm` (cuBLAS) 0.845 s / 582, `_qsa_sparse_split_kernel`
0.797 s / 36 (22 ms each), the `<2,128,2,*>` MoE instances 0.285 + 0.217 + 0.210 + 0.130 s, GDN kernels 0.614 s,
`reconstruct_had_batch` 0.169 s + `reconstruct_had` 0.108 s. On the 2060 (s2 trace): `exl3_moe_kernel<2,128,2,64,false>`
0.196 s / 37, `_qsa_sparse_split_kernel` 0.107 s / 3 (35.8 ms each), GDN 0.215 s, and **cuBLAS fp16 GEMMs ~0.23 s**
(`turing_fp16_s1688gemm*`, `turing_s1688gemm*`) because `hgemm_f16acc` is gated to Ampere+, so the 2060's reconstruct GEMMs
run SIMT. The P100's cuBLAS fallbacks (`hgemm_32x32x64_NT_vec` 0.469 s / 162, `hgemm_128x128x8_NT_vec` 0.266 s / 162,
`hgemm_32x64x32_NN_vec` 0.056 s / 80, `hgemm_32x32x32_NT_vec` 0.051 s / 162 = 0.84 s, plus `aten::mm` 0.845 s) are the
shapes `hfma2::covered`/`pick_tile` reject: gates/ups of the MoE reconstruct tiers have N = 640 and small M, so their
64-row-block grid stays under the 96-block threshold and they go to cuBLAS. Not tuned in this session; a
shape-microbenchmark sweep (`ext.hgemm_hfma2` vs cuBLAS per shape) is the way in.

**Open lead (the biggest remaining prefill lever): the blocking pageable DtoH readbacks.** 355 `aten::copy_` calls,
13.16 s of CPU time against 0.03 s of GPU time, median 2.1 kB, 20-150 ms each, on the compute stream: the host asks for
small planning tensors (CPU-tail routing/slot handoff values) in the middle of the layer loop and pays the full
compute-stream latency for each. Options, in likely order of payoff: (a) move those readbacks to a pinned ring with
`cuda_memcpyAsync` and consume the values one layer later (double-buffered host-side planning, the producer keeps running
on period i while the host uses the values of period i-1); (b) restructure the layer pipeline so layer i's CPU tail is
collected after layer i+1's GPU work is enqueued (the decode path already defers its collects this way; prefill collects
the tail inline); (c) find the exact call sites - `bench_flash.py -copy_sites` prints the Python stack of every distinct
`Tensor.copy_` pattern (added, but the verify loads that would have run it were cancelled for lane time). Expected gain:
up to several seconds per 4096-token chunk, i.e. the difference between a ~16 s and a ~11-12 s cold 4096 prefill if the
host can run one layer ahead of the GPU. This is the item to pick up first for prefill work on this model.

**ngram-io lane, for the record** (lead's report, not my measurement): the n-gram row-table ramp reduced the exposed prefill
stall from 7.8 s to 0.4-1.2 s; the 4096-token t/s in their ABBA was thermally confounded, so the final cooled
merged-vs-baseline battery will quantify the wall-time gain.

**Harness additions used here** (`bench_flash.py`, append-only): `-check_text N` (md5 of 48 greedy continuation tokens);
`-check_text_toggle 1,0,...` (same-process rounds at one cold offset that flip `moe_cpu_host.TUNING.async_dma`, an
identity gate for a copy-path change); `-trace <prefix>` (chrome trace of the profiler windows); `-record_shapes`;
`-copy_sites`; `-abba_async 1,0,1,0` (within-load A/B of the copy path, one mode per prefill run - the only design that
avoids the expert-placement confound above). Trace tooling: `tmp/trace_analyze2.py` (per-device kernels/copies/gaps,
memcpy bytes and GB/s per kind), `tmp/trace_blocked.py` (blocked runtime calls correlated to their GPU activity and to the
torch op that issued them), `tmp/copy_hist.py` (copy size histogram), `tmp/micro_copy.py` (is_pinned + copy_ vs raw
`cudaMemcpyAsync` timing on both devices).
### Qwen3.8-Flash-Next: the n-gram row stall is USB Bulk-Only latency; progressive first chunks (Nov 8, branch `flash-ngram-io`)

**Where the stall is.** Re-measured on `flash-opt` + util (no profiler, `-mtp -ndt 3 -sdh 16384`, `-gs 4.0,15.0 -cs 24576 -cq 4 -mcs 256`, one load, cold
4096/8192-token prompts at distinct corpus offsets, P100 71-78 C): a 4096-token prefill takes **21.43 s (191.1 t/s)** and a 8192-token
one **36.08 s (227.1 t/s)**. Per-lookup accounting (hooks + `EXL3_NGRAM_DEBUG=1`) puts the *exposed* n-gram wait at **5.43 s** (4096; 25 %
of the prefill) and **4.48 s** (8192): one whole first gather in each run (U 21.1 k / 17.5 k unique rows, 5.13 / 4.18 s on the prefetch
worker) plus 0.39 s on the final 255-token chunk. The later 2048/1792-token chunks are already hidden: their prefetch runs on the worker
while the previous chunk computes. Prefill-phase `_gather_rows`: 10 calls, 24.3 s total over the two runs (mean 2.4 s) - the table touches
**9.7-11.5 unique rows per token** (16 lookups, the hash dedups the rest), i.e. ~50 k rows (~10 s of device time) per 4096-token prompt at
170-240 us/row.

**The device: queue depth 1, nothing to parallelize** (`util/bench_ngram_io.py`, `/sys/block/sdc/stat`,
`/sys/block/sdc/device/queue_depth`). The model sits on a Kingston SA400 behind an ASM1051E USB-SATA bridge. The device is Bulk-Only
(`bInterfaceProtocol 0x80`; the bridge cannot do UAS), so `sdc` has **queue_depth 1** and `nr_requests 2`: one command at a time, no depth
from threads. A cold random 62-byte row costs **280-340 us** (it pulls one 4-K page). Eight threads = one thread; buffered 4 K reads,
`O_DIRECT` 512-byte-aligned reads, `O_DIRECT` 4-K reads and `POSIX_FADV_RANDOM` are all within noise of the plain 62-byte path (~300 us/row,
~4.1-5.8 KB of block-layer traffic per row; sequential read of the same file: 197 MiB/s). Rows are hash-scattered, so run coalescing is
negligible (4054 reads for 4096 rows). The 128 shards are back-to-back in `ngram_embedding.safetensors` (`util/bench_ngram_io.py` prints the
contiguity check); `RowTable.open` merges them into one handle, so a lookup issues **one** `ngram_gather_cpu` call for the whole table - the
per-shard-call hypothesis is not the problem. Net: the gather costs a fixed ~2.2-2.3 ms of device time per token, and the only lever left is
to overlap it with compute (or not read the rows).

**Look-ahead fix (`Job._next_chunk_end`).** The prefill loop cuts a chunk that would straddle the last full page,
`(len(sequence_ids) - 1) // PAGE_SIZE * PAGE_SIZE`, so the final partial page is a forward of its own (a recurrent checkpoint at the generation
boundary). `params["ple_prefetch_next"]` predicted `(next_start + chunk_size) // PAGE_SIZE * PAGE_SIZE`, which ignores that cut: for a
4096-token prompt it predicted 4095 while the loop ran 3840, so the staged set was never the one the next lookup asked for - the erratum of
Nov 7 ("the prefetch entry is never hit") is exactly this. The prediction now mirrors the loop (page rounding, prompt end, last-page cut) and
is pinned by `tests/test_ngram_ramp_plan.py`. Why the old bug looked cheap: the mispredicted ids are a *superset* of the next chunk's,
so the wasted gather still warmed the page cache of every row that was needed and the next lookup's own gather cost 32 ms instead of
3.4 s (Nov 8 old-code trace with MTP: 5.16 s mispredicted gather, then a 31.8 ms gather on the worker, 0.000 s wait). It only removed the
overlap for real: the wasted gather occupied the single worker for 5.2 s in the 4096-token run, and the misprediction is what the Nov 7
erratum saw as "the staged entry is never taken" (`hit` stayed at 2, `retired` at 1).

**Partial-chunk prefetch (`PREFETCH_MIN_TOKENS` 256 -> 64).** The last chunk of a prompt is the leftover page(s), 255 tokens for a 4096- or
8192-token prompt. prefetch() declined it by a single token (255 positions < 256), so its 2.4-2.8 k rows were gathered inline at the PLE
layer: 0.39 s of exposed wait in every run. The threshold exists to keep *decode* steps inline (4-16 positions, where the worker hop costs more
than the gather); 64 keeps that and lets a one-page prefill chunk be staged.

**Progressive first chunks (`EXL3_NGRAM_RAMP`, default on; `EXL3_NGRAM_RAMP_FIRST` 256 = one page, `EXL3_NGRAM_RAMP_GROW` 2.0).** The first
gather of a cold prompt has nothing to overlap, and its cost is linear in the rows the chunk touches, so a dedicated prefill starts with one
page and grows by whole pages up to `max_chunk_size`. 4096-token prompt: 256/512/1024/2048/255; 8192: 256/512/1024/2048/2048/2048/255. The token
order, page alignment and per-chunk forward semantics are unchanged (the model already forwards page-aligned chunks); only the boundaries of a
*cold, dedicated* prefill move. Gated off when prefilling shares the GPU with generating jobs (`chunk_size` given), for multimodal models
(span-exact chunking), for TP (no row prefetch on that path) and when `max_chunk_size` is already one page. With the first wait reduced to one
page's rows (~2.5-2.9 k, ~0.5 s), the rest of the prompt's row I/O (2.2-2.3 ms/token) fits inside its compute (3.8-4.1 ms/token) and runs
behind the GPU.

**A/B (ABBA, 2 loads per arm, cold 4096 and 8192 token prompts, distinct corpus offsets, no profiler, no MTP,
`-gs 4.0,15.0 -cs 24576 -cq 4 -mcs 256`, one `util/bench_flash.py -run prefill -lengths 4096 8192 -repeat 1 -hooks` load per measurement,
`EXL3_NGRAM_DEBUG=1` on both arms):** | arm | 4096 t/s | 4096 exposed wait | 8192 t/s | 8192 exposed wait |
| :--- | ---: | ---: | ---: | ---: |
| A, ramp off | 207.4 / 198.9 | **3.79 / 2.65 s** | 218.5 / 194.0 | **4.03 / 3.47 s** |
| B, ramp on | 195.2 / 189.6 | **0.19 / 0.67 s** | 189.7 / 194.8 | **0.24 / 0.56 s** |

The exposed wait drops 5-20x (**3.2-3.8 s -> 0.2-0.7 s**); every lookup in the B arm takes its staged set
(5/5 and 7/7 HIT prefetched per run, waits 0.000 s after the first chunk), and the first chunk's gather
falls from 21 k to 3.5 k rows. The wall t/s of this batch is *not* a clean comparison: the B loads landed
on a throttling P100 (79 C, 1189-1164 MHz) while A1 ran at 1328 MHz, so B is 5-7 % down on clock alone;
subtracting the exposed wait and scaling the remaining compute to 1328 MHz gives A 16.0 / 17.4 s and B
19.7 / 19.8 s of compute at 4096, i.e. the ramp also costs real compute - see the next paragraph. The
batch was taken with two other agents' CUDA builds running (load 9-13).


**The small chunks cost compute.** The 256/512/1024-token ramp chunks run at ~60 % of the per-token
efficiency of a 2048-token chunk with the CPU expert worker as loaded as it was here (+1.7 ms/token for
the 1792 small-chunk tokens of a 4096-token prompt, ~+3-4 s). That roughly cancels the ~3.2-3.8 s of
stall they remove, which is why the wall t/s in the batch above does not improve: the win is in exposed
wait, not (yet) in wall time. The schedule is therefore too aggressive - 44 % of a 4096-token prompt
runs in sub-max chunks. The knobs for a cheaper schedule are in the code (EXL3_NGRAM_RAMP_GROW,
EXL3_NGRAM_RAMP_FIRST): one 512-token first chunk and then straight to max_chunk_size should keep ~1.2 s
of exposed wait (3x better than the baseline) with only 512 tokens in a small chunk. The efficiency
figure is measured under two concurrent builds, and small chunks lean on the CPU expert worker
disproportionately, so the clean number may be better; a cooled pair (P100 <= 62 C before each load and
the table's page cache dropped between loads) is the next measurement.

**Greedy identity.** Not measured: the greedy text with different chunk plans was not compared (out of time). Only the chunk *boundaries* of a dedicated prefill move - page-aligned, same tokens and order, same per-chunk forward - the change is I/O-only, and the model's normal chunking already varies chunk sizes per prompt (2048/1792/255), so differing boundaries are exercised by design. Whoever enables the ramp should run the bit-identical-text check first: one load, greedy, compare the md5 across the two chunk plans.

**Cooled confirmation pair (gentle schedule).** One pair of loads with the P100 at <= 62 C before each and the table's page cache dropped between them (cold rows in both arms): A = ramp off, B = `EXL3_NGRAM_RAMP_FIRST=512 EXL3_NGRAM_RAMP_GROW=4` (two pages, then straight to `max_chunk_size`).

| arm | 4096 t/s | 4096 exposed | 16384 t/s | 16384 exposed | P100 |
| :--- | ---: | ---: | ---: | ---: | :--- |
| A, ramp off | 190.8 | 5.27 s | 242.8 | 4.05 s | 1328 MHz (70-72 C / 72-79 C) |
| B, gentle ramp | 185.3 | 3.91 s | 240.0 | 2.76 s | 1328 MHz (69-72 C / 72-79 C -> 1265) |

B has less exposed wait at both lengths (1.3-1.5x) but is 0.6-0.8 s *slower* in wall time: one more forward and the less-efficient small chunks cost more than the stall they remove. The gentle schedule is also the wrong *shape* - its second chunk (2048 tokens, ~4.5 s cold gather) cannot hide behind the first 512-token chunk's ~2 s of compute, so part of that gather is exposed anyway. A gradual (doubling) schedule hides better: the hot-batch hooks below show 0.19-0.67 s of exposed wait against 3.2-4.0 s for the baseline (5-20x less), but that batch's compute penalty and thermal throttling could not be separated from its build load.

**Why this does not contradict the Nov 7 entry "progressive first chunks do not help Flash-Next (ruled out)".** That experiment (4 rounds x
3 modes on cold 4096-token prompts, one load per mode, cold-prompt spread 16.6-22.2 s) changed only the chunk sizes while the look-ahead was
still broken, so every chunk still gathered inline at its own PLE layer and the total disk time stayed a floor under the prefill; it also
compared single loads inside a spread larger than the effect. Here the chunk ramp and the corrected look-ahead land together, the exposed wait
is accounted per lookup with the row-table hooks, and the comparison is 2 loads per arm ABBA on distinct cold offsets.

**Decision (measured, Nov 8): `EXL3_NGRAM_RAMP` ships default OFF.** The stall is real and the ramp removes most of it (5-20x in the hooks), but in the controlled pair the wall t/s did not improve: the extra forwards and the ~60 %-per-token efficiency of the small chunks cost about what the 3-5 s of stall costs. Shipping a measured wash as default-on would not be honest. What ships on: the exact next-chunk look-ahead (`_next_chunk_end`, the erratum fix - the staged set is now the one the next lookup takes) and the partial-chunk prefetch (a 255-token final chunk is staged instead of gathered inline, worth ~0.39 s in every cold prefill). The ramp stays available behind the switch.

**Open.** (1) The doubling schedule's exposed wait (0.19-0.67 s) is the right target; what it needs is less per-forward overhead for small chunks (or a machine without the CPU MoE worker contending), not another schedule tweak. (2) Stage the first chunk at `Job` *enqueue* time, so a request that arrives while another job is generating hides its first gather behind that job (the benchmark harness enqueues immediately before prefill, so this cannot be measured there). (3) The decode lookup (~64 rows/step = ~19 ms of device time cold) is the same latency; the advise() path moves the same commands earlier rather than removing them. (4) The blocking DtoH readbacks of the CPU expert tail are the real remaining prefill lever now (prefill-gpu's section); with them gone the exposes measured here are what is left.
### Qwen3.8-Flash-Next: m = 9..16 verification rows for hybrid drafting (Nov 8, branch `flash-decode-draft`)

**Problem.** Hybrid drafting on copy-heavy agentic turns: `-hdt 7` (8 rows per verification) gave 49.0 t/s against 42.5
at m = 4 (Nov 7), but `-hdt 12` (13 rows) collapsed to 38.3 because every dense projection left the small-m GEMV at
`size_m > EXL3_GEMV_MAX_M` (8, `exl3_gemv.cu`) and ran the block-pipelined tile GEMM, which on sm_60 is several times off
the roofline. This section makes the 9..16-row verification fast.

**The rows were already in the fragments.** MMODE 1 loads only the first row group of the m16n8k16 fragments: `a01[1]` /
`a23[1]` (A rows lane/4 + 8) are zeroed and the second half of `FragC_h` (C rows lane/4 + 8) is discarded, so rows 8..15
were computed for free and thrown away. **MMODE 5** turns the second group on: two more A half2 loads per k-slice, 8 more
shuffles per k-slice on sm_60 (`gather_a_rows16`, the emulated MMA), a second fp32 fold accumulator (`acc1`) and 16
instead of 8 smem rows in `sh_red` (32 KB static). Row r's arithmetic is unchanged (the MMA has no cross-row terms, the
fold cadence and the cross-warp reduction are the same), so MMODE 5 is bit-identical to MMODE 1 for the same rows.

**Dispatch and instances** (`exl3_gemv.cu`, `exl3_gemv_kernel.cuh`):
- `EXL3_GEMV_MAX_M` stays 8, so every existing path is untouched; `EXL3_GEMV_MAX_M16` = 16 and MMODE 5 only when
  9 <= size_m <= 16.
- Taken on Pascal and Turing only, integer mul1 rates only. 12 instances: `SEL_GRID_M5(bits, cb, smem)` x {cfg 0/1} x
  {fp16/fp32} for (4, mul1, shuffle extraction) = lm_head, (5, mul1, staged words) = attention / GDN projections,
  (6, mul1, staged words) = MTP fc_hidden / fc_embedding. Every other width, codebook, half-integer rate and m > 16 keeps
  the tile-GEMM fallback it had.
- `EXL3_GEMV_M16=0` disables the path (read per call, for A/B). The cfg heuristic is unchanged.

**What did not need changing.**
- **MoE coop**: runs are capped at ROWS = 8 *slots of one expert*, not at 8 verification rows, so m = 13 only means more
  runs (130 slots vs 80 at m = 8) - the cost is linear in the number of distinct experts, which is inherent to verifying
  more rows (each expert's trellis has to be decoded once per run). `MAX_BSZN` = 8 is the *bsz* dimension (1 here) and
  `MAX_SLOTS` = 256 covers 16 rows x top_k 10, so no row template or scratch change.
- **Gated residual**: R 5..32 already takes `gr_dots_kernel2` / `gr_finalize_kernel2` (GR_MAX_R = 32); only R <= 4 uses
  `gr_mix3`, which cannot be widened (10.2 KB of smem per row, R = 4 is already 41 KB of the 48 KB block limit). So
  9..16 pays the ~1.5-2x slower generic pair, no cliff.
- **Host**: nothing. `model_init.py` sizes the cache `max_history` from max(ndt, hdt) (hdt 15 -> 16 rows is legal),
  `max_q_size` = `max_draft_tokens` + 1, and `row_limited` / `draft_max_rows` is off by default.

**Kernel tests** (`tests/kernels/gemm/test_exl3_gemv_m16.py`, new; modelled on `test_exl3_gemv_m34.py`). Contract: a
single m-row call (m = 9 / 12 / 13 / 16) must equal rows 0..7 in one call plus rows 8..m-1 zero-padded to 8 rows in a
second call, bit for bit (that second call is MMODE 1, i.e. exactly the arithmetic MMODE 5 has to reproduce); plus the
float64-reference tolerance (rel RMS < 2.5e-3, max < 4e-3), zero-padded-row invariance, repeat-launch identity and
`EXL3_GEMV_M16=0` making m > 8 ineligible.
- P100 (cuda:1): 11/11 passed. RTX 2060 (cuda:0): 3 passed, 8 skipped (K = 5 / 6 are Pascal-only, K = 4 runs).
- Regression `test_exl3_gemv.py` + `test_exl3_gemv_m34.py` + `test_exl3_gemv_tail.py`: 52 passed, 1 skipped on both
  devices.

**Registers / stack / shared memory** (`cuobjdump -res-usage`, MMODE 1 -> MMODE 5, per sm_60 cubin):

| bits, cfg | regs | stack | smem |
| :--- | ---: | ---: | ---: |
| 4, narrow | 127 -> 128 | 0 -> 0 | 16384 -> 32768 |
| 4, wide | 217 -> 220 | 0 -> 0 | 16384 -> 32768 |
| 5, narrow | 128 -> 128 | 0 -> 0 | 22528 -> 38912 |
| 5, wide | 172 -> 165 | 0 -> 0 | 21504 -> 37888 |
| 6, narrow | 127 -> 128 | 0 -> 0 | 22528 -> 38912 |
| 6, wide | 174 -> 164 | 0 -> 0 | 22528 -> 38912 |

No spills anywhere (STACK 0), occupancy unchanged (sm_60 narrow is 512 thr x 128 regs = the whole 64K register file for
MMODE 1 already; wide 165-220 regs at 256 thr = 1 block/SM), staged smem 38.9 KB < 48 KB per block. The structural cost
is the +16 KB of smem and, on sm_60, the 8 extra A-gather shuffles per k-slice.

**End to end** (`util/bench_rewrite.py`, the ~7000-char copy-heavy rewrite turn, greedy, `-gs 4.0,15.0 -cs 24576 -cq 4
-mcs 256 -mtp -ndt 3 -sdh 16384 -hdt 15 -hdts 0 7 12 15 -rounds 2 -max_new 500`; `-hdt 15` at load time only sizes
`max_history`; the widths are switched per turn, m = 4 / 8 / 13 / 16):

| load | thermal | hdt 0 (m 4) | hdt 7 (m 8) | hdt 12 (m 13) | hdt 15 (m 16) |
| :--- | :--- | ---: | ---: | ---: | ---: |
| a1, first load of the session (cold n-gram page cache) | cooled to 64 C, 1328-1177 MHz | 37.7 | 47.4 | **49.4** | 48.7 |
| a2, warm cache | 69-79 C, 1328-784 MHz (throttling) | 50.3 | 65.4 | 43.0 | 46.4 |
| b1, warm cache | 75-79 C, 1328-1025 MHz (throttling) | 50.6 | 63.2 | 37.3 | 38.0 |

a1, per verification row: 26.1 / 20.3 / 18.6 / 17.8 ms per row (ms per step 104 / 162 / 241 / 285, tokens per step
3.96 / 7.80 / 12.17 / 14.26, acceptance 98 / 98 / 98 / 97%). The m > 8 cliff is gone: at m = 13 the step is now
*cheaper per row* than at m = 8, where on Nov 7 `-hdt 12` (the same 13 rows, no MMODE 5) gave 38.3 t/s, i.e. 300-320 ms
per step at the same acceptance, ~23-25 ms per row - i.e. **+29% at hdt 12 in the same cooled load** (49.4 vs 38.3 t/s),
hdt 7 inside the noise
(47.4 vs 49.0) and hdt 0 lower by the cold-cache/placement noise (37.7 vs 42.5).

Caveats, in order of size:
- **Thermal**: the P100 throttles to 784-1189 MHz during the later turns of a load, and the widths run in a fixed order
  within a round, so a2/b1 cannot compare widths (their hdt 12/15 turns ran at 784-1160 MHz against 1328 for hdt 7).
  Only a1 (cooled, first round at 1328 MHz, all four widths within 1303-1328 MHz) compares widths fairly; a future A/B
  must either interleave widths or cool between turns.
- **Page cache**: a1 is the first load of the session and pays the cold n-gram table reads (hdt 0: 37.7 against 50.3 in
  the warm loads at the same clock). The cold penalty is per step, so it flatters the wide widths slightly in a1; the
  per-row comparison above is corrected for it only qualitatively.
- **The M16=0 arm was not measured.** b1 was intended as it, but `bench_rewrite.py` cleared `EXL3_GEMV_M16` before its
  warm-up turn (only the `-m16ab` path should do that), so all three loads ran with the path on - the JSON's `m16`
  field records 1 for every row. Fixed; a valid pair is the next measurement.
- **Exactness**: the kernel-level bit-identity test is the strong evidence. In the model, the greedy text was identical
  across all four draft widths of the same load for the same prompt (k = 100: a2 and b1 both 717ed118ca; k = 101 in b1
  all four widths 82af09ba66; a1's k = 100 all four widths 711f6fbdff, its k = 101 text was not accumulated by the old
  harness). One outlier: a2 k = 101 hdt 7 differed from hdt 0/12/15 - that is m = 8, where MMODE 5 is not used, so it is
  the known run-to-run drift, not the new path.

**Verdict / next.** With MMODE 5 the 13/16-row verification no longer pays the tile-GEMM penalty, so `-hdt 12` is back
in play as the copy-heavy code setting (49.4 t/s against 47.4 at `-hdt 7` in the same cooled load; +29% over the Nov 7
`-hdt 12` number), and the dense part of the step is now flat in the number of verification rows. What is left of the
per-row cost at m = 13 (18.9 ms/row) is the MoE (distinct experts grow with the verification width) and the R > 4
gated-residual pair, not the GEMV. Remaining work: one valid cooled ABBA pair with `EXL3_GEMV_M16=0` (the harness is
fixed), and, if the wide widths are to be the default, a `-hdt` recommendation that accounts for the extra MoE time per
row on non-copying turns.

## Qwen3.8-Flash-Next: decode and prefill optimization round - integration (Nov 8, branch `flash-opt`)

Three workstreams developed in parallel by three agents in separate worktrees, each measured against the same tree, then
merged here: `flash-opt` = `flash-ngram-prefetch` 89fe340e (which already contains `pascal-27b-run` and the Nov 7
Flash-Next work) + the three branches below. End-to-end config for every run: `-gs 4.0,15.0 -cs 24576 -cq 4 -mcs 256`,
plus `-mtp -ndt 3 -sdh 16384` when decoding; greedy; P100 temperature and clock logged before/after every run.

Hardware facts measured this round (not datasheet): the P100 trains at PCIe Gen3 **x4** (2.75-2.83 GB/s measured; the RTX
2060 is x16 at 12.96-12.98 GB/s); the model disk is a USB **Bulk-Only** bridge (ASM1051E, protocol 0x80) with device
`queue_depth 1`, so a cold random 62-byte n-gram row costs 280-340 us regardless of thread count and the 19.8 GB table is
latency-bound, not bandwidth-bound; the host has 31 GB of RAM (a model load plus a CUDA build do not coexist: one TU peaks
at ~7.5 GB).

### What changed
1. **N-gram row table (prefill I/O)**, branch `flash-ngram-io`: exact next-chunk lookahead (`Job._next_chunk_end`
   mirrors the prefill loop's own bounds including the recurrent last-page split - the Nov 7 erratum, where the staged set
   predicted 4095 tokens and the loop then ran 3840), `PREFETCH_MIN_TOKENS` 256 -> 64 (`EXL3_NGRAM_PREFETCH_MIN`) so the
   255-token final chunk is staged instead of gathered inline (~0.39 s per cold prefill), and progressive first chunks
   (`EXL3_NGRAM_RAMP`, default **off** - see below).
2. **Streamed CPU-expert copies (prefill)**, branch `flash-prefill-gpu`: per-job issue/readback and VRAM-ring DMAs use a
   raw `cudaMemcpyAsync` (`EXL3_MOE_ASYNC_DMA`, default on) instead of `Tensor.copy_`, which does not recognise
   `cudaHostRegister`'ed memory as pinned; correct by construction, performance-neutral in the warm-placement case. The
   traces taken for it found the real remaining serialization (open lead 1 below).
3. **Decode**, branch `flash-decode-draft`: GEMV MMODE 5 for m = 9..16 verification rows (`EXL3_GEMV_M16`, default on;
   the m16n8k16 fragments already carry rows 8..15, so the second group costs one more A load per k-slice, one more fold
   accumulator and 16 instead of 8 shared-memory rows; bit-identical to two m <= 8 GEMVs; 11/11 new tests on the P100,
   3 pass + 8 Pascal-only skips on the 2060, gemv/m34/tail regressions 52 passed/1 skipped on both; no spills, 1 block/SM,
   staged smem 38.9 KB < 48 KB). This removes the reason `-hdt 12` used to collapse: at m = 13 the dense projections used
   to leave the GEMV for the sm_60 tile GEMM.

### Measured results (same protocol: cold prompt, n-gram page cache dropped, P100 cooled before the load, 1328 MHz)

| measurement | before (`flash-opt`) | after (this branch) |
| :--- | ---: | ---: |
| cold prefill 4096 tok | 194.5 t/s (21.06 s), 73-75 C | 206.5 t/s (19.84 s), 66-68 C |
| cold prefill 16384 tok | 236.3 t/s (69.34 s), 75-79 C, 1088 MHz at the end | 238.7 t/s (68.64 s), 68-75 C, 1328 MHz |
| decode, code prompt, 300 greedy tok | 40.21 t/s, 81.6% accept, 3.48 tok/step | 40.94 t/s, 84.3% accept, 3.56 tok/step |
| copy-heavy agentic turn, 500 tok, `-hdt 0` | 37.7-42.5 t/s | 37.7 t/s (same load as the rows below) |
| copy-heavy agentic turn, `-hdt 7` | 47.0-47.4 t/s | 47.4 t/s |
| copy-heavy agentic turn, `-hdt 12` | 38.3 t/s (Nov 7, no MMODE 5, tile-GEMM fallback) | **49.4 t/s** (+29% over Nov 7; same load as `-hdt 7`) |
| copy-heavy agentic turn, `-hdt 15` | n/a | 48.7 t/s |

Reading the table honestly: (a) the two end-to-end prefill rows were run on the after-tree 6-8 C cooler at load start, so
the 4096 difference (+6%) is partly thermal and the 16384 one (+1%) is inside run-to-run noise: **the prefill wall time is
unchanged in this round**, while the n-gram stall itself is mechanically 5-20x smaller (exposed first gather 3.2-4.0 s ->
0.2-0.7 s, 5/5 and 7/7 lookups HIT prefetched) and the remaining serialization is now identified and quantified (open lead
1). (b) The copy-heavy rows come from the cooled load `a1` (P100 64 C at start, 1328 MHz; hdt 0/7/12/15 in ONE load,
98% acceptance, 3.96 / 7.80 / 12.17 / 14.26 tokens per step, 26.1 / 20.3 / 18.6 / 17.8 ms per row): at 13 rows the step
is cheaper per row than at 8, and greedy text was identical across the four widths in the same load. (c) The standard
decode row is inside the +-5-10% run-to-run spread: MMODE 5 does not touch m <= 8, so no change is expected there.

### Committed defaults and recommended run config
- Defaults: `EXL3_GEMV_M16=1`, `EXL3_MOE_ASYNC_DMA=1`, `EXL3_NGRAM_PREFETCH_MIN=64`, `EXL3_NGRAM_RAMP=0`.
- Agentic code / copy-heavy turns: `-mtp -ndt 3 -sdh 16384 -hdt 12` (`-hdt 7` when a turn may not be copying: a long
  draft that is not accepted costs MoE time).
- Free-form chat / prose: no `-hdt` (acceptance there is 30-40%, extra draft positions do not pay).
- Long context unchanged: `-cs 131072 -mcs 272` .. `-cs 262144 -mcs 320`.

### Open leads (measured, not done, ranked)
1. **355 blocking pageable DtoH readbacks** in the CPU-tail handoff: 13.16 s of host-blocked CPU time per 4096-token cold
   prefill (median 2.1 kB each, 20-150 ms per call waiting on the compute stream) against ~10.5 s of P100 busy. A pinned
   ring with the copy consumed one layer later, or deferring the collect across layers, should recover several seconds per
   chunk (16 s -> ~11-12 s estimated).
2. **P100 small-grid reconstruct GEMMs fall back to cuBLAS**: 0.845 s `aten::mm` + 0.84 s `hgemm_*` on one 4096-token
   prefill; `hgemm_f16acc`'s `pick_tile` rejects < 96 blocks of 64 rows. The 2060's reconstruct GEMMs are excluded from
   `hgemm_f16acc` by an Ampere+ gate (0.23 s).
3. **MoE prefill kernel** is still the largest GPU item: 4.9 s of a 14 s GPU budget on a cold 4096-token prefill
   (2.43 s in `exl3_moe_kernel<1,128,2,16,true>`).
4. **Ramp**: the stall is mechanically hidden but the CPU contention eats the gain; re-measure at 16k-64k prompts after
   lead 1 frees the CPU tail.
5. **M16 = 0 arm**: the intended off-arm of the decode A/B never ran (a harness env reset) and the second pair was
   thermally unusable (784-1316 MHz); the +29% at `-hdt 12` is against the Nov 7 cross-session number. One cooled 2-load
   pair with the fixed harness is the outstanding check.
6. **Chunk plans**: greedy text across differing chunk plans was not compared; whoever enables the ramp should run the
   one-load md5 check first.

### Qwen3.8-Flash-Next: coverage curve of a hot-row cache for the n-gram table (Nov 11, offline, CPU only)

Question: how many of a prompt's cold n-gram row reads would a pinned cache of the most frequent rows remove? The first
gather of a cold 4096-token prefill is the largest single stall of the trace taken on `flash-opt` (Nov 10: 2060 idle 5.1 s
and P100 idle 5.9 s at the start of a 20.8 s window; 21 k unique rows at ~240 us each, see the ngram-io section).
`util/ngram_cov.py` reads only the three hash tensors from the table header (`head_offsets`, `head_vocab_sizes`,
`layer_multipliers`: 16 heads x 20 M rows, 62 B each, 320 M rows, one table under `layers.1.ple`), tokenizes a training corpus,
computes the row ids with the reference hash (`NGramEmbedding.compute_ngram_ids`), counts for each row in how many 4096-token
training prompts it appears, and reports the share of a held-out prompt's unique rows covered by the N most frequent rows.
Train: python files of `.venv/site-packages` (12 M / 59 M tokens); test: `exllamav3/` (549 prompts, 28.3 k unique rows per prompt:
11.7 k bigram heads, 16.6 k trigram heads) and `doc/*.md` (21 prompts, 44.2 k unique rows).

| cache rows | size | code, 12 M train tokens | code, 59 M train tokens | doc prose, 12 M |
| ---: | ---: | ---: | ---: | ---: |
| 1 M | 62 MB | 28.7% | 29.2% | 19.1% |
| 4 M | 0.25 GB | 38.7% | 40.0% | 27.8% |
| 8 M | 0.5 GB | 43.3% | 46.0% | 32.3% |
| 16 M | 1.0 GB | 53.3% | 54.2% | 42.9% |
| 32 M | 2.0 GB | 53.3% (corpus saturated) | 60.2% | 42.9% |

Bigram heads are covered much better than trigram heads (59 M train, 0.5 GB: 62% against 35%). Returns diminish: 5x more training
data adds 2-7 points. Estimate, not measured end to end: a 0.5-1 GB pinned hot pack would remove about half of the cold reads, so
the exposed first gather would drop from ~5 s to ~2.5-3 s on a cold page cache (about -2.3...-2.8 s of a 20 s prefill); the
effect shrinks once the kernel page cache is warm (later turns), and the corpus here is code (real agentic traffic has more
prose and JSON, which covers worse). The rows are copied byte for byte, so outputs stay bit-identical. Building the pack needs
one sequential scan of the 19.8 GB file (~100 s at 197 MiB/s; the hash-scattered rows cannot be fetched by random reads).
The model dir already sits on the Kingston, so the pack would too.

**Emulation of the hot-row cache (Nov 11, measured; `util/bench_flash.py -warm_ab`).** One load, the same cold 4096-token prompt in every run
(first token replaced by a distinct id so the prefix cache never hits; the rows of the rest are identical), the n-gram table evicted from
the kernel page cache (`posix_fadvise DONTNEED`) before each run, then a random fraction of the prompt's 38.9 k unique rows pre-read
(0 / 46 / 100 %). ABBA order, 13 runs (the first of them is the 0 % arm and is kept), P100 52-75 C at 1328 MHz, `-gs 4.0,15.0 -cs 24576 -cq 4 -mcs 256`:

| rows warm | wall (4 or 5 runs) | t/s | vs cold |
| ---: | ---: | ---: | ---: |
| 0 % | 19.59 s (19.37 / 19.59 / 19.64 / 19.76) | 209.1 | |
| 46 % | 17.43 s (17.19 / 17.35 / 17.45 / 17.56 / 17.60) | 235.0 | -2.16 s (-11 %) |
| 100 % | 14.61 s (14.51 / 14.50 / 14.70 / 14.72) | 280.5 | -4.98 s (-25 %) |

Within-arm spread is +-0.15 s, well below the differences. The gain is linear (~0.5 s per 10 % of rows warm) and the 100 % arm
matches the ~5 s exposed first gather of the Nov 10 trace. With the coverage table above: a 0.5 GB pack (~46 % on code) is worth about
-2.2 s, 1 GB (~54 %) about -2.6 s, 2 GB (~60 %) about -3.0 s of a cold 4096-token prefill. Page-cache hits stand in for hot-pack hits
(both are memory reads); the pack adds an index lookup and a copy per unique row, not measured. Valid for a cold page cache only.

### Agentic A/B, Flash-Next vs Qwen3.8-27B through Pi + TabbyAPI (Nov 11, one run each, thinking off)

Setup: TabbyAPI main with the fork (its minimum compute capability patched to 6.0), Pi 1.5.2 as the agent (custom provider in `models.json`),
sampling 0.7 / 0.8 / 20, presence penalty 1.5. Flash-Next `-gs 4.0,15.0 -mcs 256`, cache Q4, MTP ndt 3; 27B 4.00bpw, cache Q8 (Q4 in the
first Python run), MTP ndt 3. Four tasks per language (fix three bugs, implement a documented function, rename a function across files,
read-only questions), checked by the project tests and by file hashes.

| suite | Flash-Next | 27B | wall (Flash-Next / 27B) |
|---|---|---|---|
| Python | 4/4 | 4/4 | 591 s / 500 s (Q8; 549 s with Q4) |
| C | 4/4 | 4/4 | 774 s / 1036 s |
| Rust | 4/4 | 4/4 | 497 s / 798 s |

Decode ~29 t/s against ~15 t/s, prefill 110-133 t/s against 70-100 t/s (short contexts, 2-14k). The two models are indistinguishable on
these tasks; the first Python t1 "failure" of both was a checker bug (it ran the whole suite, including tests of the not yet implemented
function).

**Long context (119k).** Synthetic package of 45 modules (105.8 k tokens with the model tokenizer), every module carries the same eight
constants with different values. Pi reads every file with the read tool, answers ten "value of X in the Y module" questions without
reopening anything (needles at 2 to 95 % depth), then adds a currency across three distant files. Pi compaction off, context 131072.

| | Flash-Next (`-gs 4.0,15.8 -mcs 288 -cs 131072`, Q4) | 27B (Q8, 131072) |
|---|---|---|
| peak prompt | 119 025 tokens | 72 585 (stopped, invalid) |
| needles | 10 / 10 | not reached |
| edit task | tests pass, tests untouched | not reached |
| wall | 1013 s | stopped after 3144 s |
| prefill | 142 t/s average (129-147 at 73-85k) | ~64 t/s up to 65k |
| decode | 27.4 t/s above 60k, 32.6 below 30k | 10-15 t/s |

The 27B session was healthy up to 65k tokens, then every read of ~2.4 k new tokens stalled for ~8 minutes (Pi times out and retries, the
retry hits the prefix cache). The P100 was at 100 % utilisation with `sw_thermal_slowdown` active (405-974 MHz at 79 C). Not separated:
thermal throttling against a software cliff near 65536. Flash-Next showed no such collapse to 119k.

Decision (user, Nov 11): continue with Flash-Next. Harness (fixtures C / Rust / long, runners, checkers) lives in the job directory, not in the repo.

### Flash-Next prefill at depth: where the time goes (Nov 11, measured; `util/bench_flash.py -incr_depths ... -incr_profile -incr_chunks`)

Config `-gs 4.0,15.8 -cs 131072 -cq 4 -mcs 288 -mtp -ndt 3 -sdh 16384`, repo-code corpus (2.47 M tokens, novel text for every increment), P100 at 79 C and
900-1330 MHz (sw thermal slowdown) during every run except the first minutes.

- **Log-mined model** (159 Tabby requests of the agentic sessions, 2-119k depth): seconds = 1.8 s per request + 5.56 ms per new token + 14 us per cached token per
  2048-chunk. At depth 0 that is 180 t/s, a 2048-token chunk at 100k costs 12.7 s (161 t/s, -11 %).
- **In-process increments on a cached prefix** (median of 4): 2048 new tokens 172 / 146 / 144 t/s at depth 0 / 8k / 82k, 2400 new 158 / 148 / 151, 4096 new 179 / 174 / 166,
  1024 new 130 / 119 / 126, 256 new 107 / 94 / 94. A single-shot prefill of 82k tokens: 204 / 189 / 187 t/s (three runs, starting at 1328 MHz and ending throttled).
  Tabby serves the same shapes at the same speed, so the server adds nothing.
- **Request shape**: with a recurrent cache the prefill loop cuts a chunk at the last full page and ingests the partial page as a forward of its own (`_next_chunk_end`), so
  2400 new tokens are three forwards (2048 + 256 + ~95) and pay the per-layer fixed cost three times: 2048 new tokens (one chunk) 172 t/s against 158 t/s at 2400.
- **Per-device kernel time of one 2304-token chunk** (torch profiler; cuda:0 = RTX 2060, cuda:1 = P100): depth 8k: P100 8.1 s, 2060 1.1 s, wall 12.8-14.0 s; depth 82k: P100 9.1 s,
  2060 1.3 s, wall 15.5-15.7 s. Neither device is busy for 28-34 % of the wall. P100 top kernels: MoE `exl3_moe_kernel<1,128,2,16,true,false>` 2.25 s, `hfma2::gemm_kernel` ~2 s,
  `_qsa_sparse_split_kernel` 0.96 s (+0.26 s on the 2060, the same at 8k and 82k), `_dsa_indexer_kernel` 0.50 s at 82k (+0.15 s on the 2060, ~0 at 8k), GDN chunk kernels ~0.4 s.
  Attention plus indexer is ~17 % of P100 kernel time; the depth-dependent part is only the indexer (+0.65 s per chunk at 82k).
- **Host side** (CPU op table of the same chunk): 92 % of the 12-13 s of CPU time is `cudaMemcpyAsync` blocking (~4800 calls: the per-layer syncs and streamed copies), identical at 8k and
  82k; nothing on the host grows with depth.
- **Chunk size 4096 against 2048** (same load, ABBA, n = 4): 2400 new tokens 14.73 vs 15.28 s (depth 0) and 14.65 vs 15.90 s (8k), 4096 new equal, 1024 new +-5 %. No VRAM failure at 4096.
  Within the thermal noise (a 2400-token request ranges 13.2-17.7 s with the P100 between 898 and 1328 MHz): not conclusive.

Levers left, each estimated below 10 % of the prefill: the CPU-tail waits (~1.1-1.7 s per chunk), the QSA sparse kernel (~1.2 s fixed per chunk), the extra small forwards per request, chunk size 4096.

### Flash-Next n-gram hot-row pack: implemented, exactness measured, timing gain NOT yet established (Nov 11)

`util/ngram_pack.py` builds a dense pack of the most frequent rows (document frequency over 4096-token training prompts, one sequential scan of the
19.8 GB table, ~2 min; 8 M rows = 0.5 GB, trained on 12 M tokens of `.venv/site-packages`). `RowTable.load_hot_pack` (`EXL3_NGRAM_PACK=<file>`, off by default) keeps the
pack in RAM and `_gather_rows` serves the rows it holds by a sorted-id `searchsorted` before any disk read; the misses take the unchanged disk path.

Measured: on a held-out 4096-token increment of the repo corpus (33 463 unique rows) the pack served 11 536 rows (34.5 %, below the 43-46 % of the offline curve: the
increment mixes code and prose) and the gathered rows were identical to the disk rows (`PACKCHECK ... identical: True`, `bench_flash.py -incr_pack_toggle`).

Not established: the timing gain. A 30-job natural-warming run (table evicted once, 30 novel 2400-token increments, no eviction between jobs) kept only 100-600 MB of the table
resident (RAM available with the model loaded: ~9.6 GB; `ngram_ram` needs 19.8 GB, excluded) and the job time stayed at 16.3 s (cold 16.9 s, fully warm 12.7 s): the page cache
does not do the pack's job. The pack on/off ABBA `packab2` was confounded (the two arms of one text always ran off-then-on, so the second found the rows in the page cache,
-7 % not usable); `packab3` (alternating order, table evicted before every job) was stopped on request before finishing. Upper bounds measured with evicted-vs-pre-read rows:
-12 % (50 % of the rows warm) and -22 % (all warm) at 2400 new tokens, a fully cold table, not a session gain.

### exl3 vs GGUF for Qwen3.8-27B, and llama.cpp on P100 + RTX 2060 (Nov 11) — closed, decision: continue with Flash-Next (QNF)

Pi agent suites (py/c/rs, 4 tasks each), same harness. exl3 27B 4.00bpw, cache Q8: py 4/4 500 s, c 4/4 1036 s, rs 4/4 798 s. llama.cpp (build 12302) GGUF UD-Q4_K_M, KV q8_0,
embedded MTP + `-md` MTP gguf, n-max 3, `-ts 27,73`: py 3/4 748 s, c 3/4 1325 s, rs 4/4 1188 s (t2 failed in py and c; cause not established; P100 at 79 °C / 544 MHz during these, so timings are thermally inflated).
GGUF decode ~9-10 t/s (21 on one short task), draft acceptance 65-85 %, prefill 60-65 t/s steady.

`llama-bench` pp512 / tg128 (27B UD-Q4_K_M, `-ts 27/73`, KV q8_0, full clocks, P100 52-60 °C, one pass each): mainline 12302 172.8 / 13.16; v0.6.0 + 43 shinbunbun P100 patches
167.7 / 16.19 (+23 % decode, -3 % prefill, older base so partly confounded); fork `pascal-opt` flags off 173.2 / 12.99. `GGML_CUDA_GRAPHS_PASCAL=1` (CUDA graphs on cc 6.0): 13.06 / 12.72 vs 12.93 off = no effect.
nsys decode: >90 % `mul_mat_vec_q` (Q5_K, IQ4_XS, Q4_K, Q6_K); the patches cut IQ4_XS -42 %, Q4_K -34 %, Q5_K -24 %, Q6_K -4 %; P100 ~66 % busy, ~270 GB/s effective of 732.
Tensor-parallel (`-sm tensor`) and the P100 mmvq kernel work were NOT started. Kinetic (Veloresearch) read: not usable here (CUDA from sm_75, single GPU, no EXL3).
Decision: QNF for long-context agentic work (119k needle 10/10); the 27B was not compared with QNF on the same suites. llama.cpp experiments removed from disk (models-gguf, p100 worktree/patch clone, builds).

### QNF decode: expert-placement hit rate and where the host main thread spends its time (Nov 12)

Agentic py suite (4/4 pass, 564 s, decode 28-33 t/s, draft accept 62-68 %), `-mcs 288 --cpu-moe-pinned-arena` (promote-on-miss) from a cool start, `EXL3_MOE_CPU_SWAP_DEBUG=1`, 25 windows of 128 steps (3200 steps):
hit rate mean 77.7 % (min 57.0 %, max 90.4 %; first third 74.2 %, last two thirds ~79 %), ~1458 routed experts/step of which ~329 on the CPU, ~5.9 promotions/step, hook 3.66 ms/step (3.32-3.75).
Main-thread stack sample (in-process sampler, 10 ms, 20 s window, one 1500-token decode at 27.8 t/s, draft 904/1797 accepted; py-spy was refused by ptrace_scope=1 even on a child process, so a `sys._current_frames()` sampler was used via `sitecustomize`):
62.8 % of MainThread samples sit in `torch.cuda.synchronize` called from `generator.py:iterate_gen:2045` (host waiting for the GPUs: the ~95 % core in htop is mostly a CUDA sync spin-wait, not work); Python `forward` (kernel launches) 30.9 % inclusive;
n-gram row gather (`row_table._gather_rows`) 9.2 %; MTP draft loop 6.7 %; hyperconnections 6.5 %; `_split_promote_step` / `cpu_split_submit` 3.9 %; `cpu_split_combine` 1.8 %.
Not established: what the host waits for. `synchronize` only says the device queue is not drained; in the split path the collect's flag waits sit on the stream after the GPU experts, so a late CPU worker (~329 experts/step on the CPU) looks the same as a busy P100. The discriminator is one decode on the 128k config in `dynamic` mode with `EXL3_MOE_WAIT_PROF` and the per-device busy profile (P100 busy, 2060 busy, CPU flag wait, nothing busy). `_gather_rows` (9.2 %) is not shown to be hidden: the rows depend on the just-sampled tokens, so it may be serial time (what `EXL3_NGRAM_PACK` targets). 5.88 promotions/step against `_PROMOTE_BUDGET = 6` means the promote path ran at its cap for all 3200 steps (churn, not a converged placement), and production runs `dynamic`, which has no hit-rate counter. The sampler sees only Python-level frames (a C loop that holds the GIL would be under-counted).
Wall split of the same suite (from `results_hit_py/t*.summary.json`, 564 s): prefill 230 s (41 %; t1 14.4k new tokens at 148 t/s = 97 of 220 s), decode 329 s (58 %, 28-33 t/s, draft accept 62-68 % at `-ndt 3`), everything else ~1 %.
Both phases matter about equally for the agentic loop; the decode lever must be chosen after the device-wait split above, not from the main-thread sample.
