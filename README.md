
<p align="center">
  <img src="doc/logo.png" width="640" alt="Llama 3.1 8B Instruct quantization benchmark across bits per weight">
</p>

[Installation](#installation) · [Pascal / P100 Support](#pascal-sm-60--tesla-p100-support) · [Supported models](#architecture-support) · [Examples](#examples) · [Quantization](#exl3-quantization) · [Community](#community)

ExLlamaV3 is an inference library for running local LLMs on modern consumer GPUs, with flexible quantization and parallel inference.

- **Pascal / Tesla P100 (SM 6.0) Support** - Native emulation of Tensor Core operations via GP100 FP16x2 (`__hfma2`), L2 cache-coherent loads, and MoE support for 16 GB Tesla P100 cards.
- **Quantization** - [EXL3](#exl3-quantization), based on QTIP, plus 2–8 bit cache quantization.
- **Parallel inference** - Flexible tensor-parallel and expert-parallel inference for consumer hardware setups.
- **CPU offloading** - Allows large MoE models to run with limited GPU resources. AVX2 and AVX512 support.  
- **Generation** - Continuous, dynamic batching, speculative decoding, multimodal support.
- **Integrations** - Broad [HF model support](#architecture-support), a [Transformers plugin](examples/transformers_integration.py), and an OpenAI-compatible API via [TabbyAPI](https://github.com/theroyallab/tabbyAPI/).

> [!TIP]
> **Looking for a server?** [TabbyAPI](https://github.com/theroyallab/tabbyAPI/) is the official and recommended backend server. It provides an OpenAI-compatible API for local or remote inference, HF model downloading, embedding model support, and HF Jinja2 chat templates. Its startup script manages and installs prerequisites to help you get started.

<p align="center">
  <img src="doc/qb_kld.png" width="640" alt="Llama 3.1 8B Instruct quantization benchmark across bits per weight">
</p>

## Installation

Start by making sure you have the appropriate version of [PyTorch](https://pytorch.org/get-started/locally/) installed (CUDA 12.4 or later) since the Torch dependency is not automatically handled by `pip`. Then pick a method below:

### Prebuilt wheel · recommended

Pick a wheel from the [releases page](https://github.com/turboderp-org/exllamav3/releases), then e.g.:

```sh
pip install https://github.com/turboderp-org/exllamav3/releases/download/v0.0.6/exllamav3-0.0.6+cu128.torch2.8.0-cp313-cp313-linux_x86_64.whl
```

### Install from PyPI

```sh
pip install exllamav3
```
Note that the PyPI package does not contain a prebuilt extension and requires the CUDA toolkit and build prerequisites (i.e. VS Build Tools on Windows, gcc on Linux, `python-dev` headers etc.).

### Build from source

<details>
<summary>Source installation with uv or pip</summary>


`exllamav3` declares a minimum `torch` version (>= 2.6.0) and CUDA version (>= 12.4), but beyond that the user is free to select a version of `torch` that is compatible with their environment.

`torch` can be installed in three ways (from least to most effort):
1. **with `uv`, setting only `--extra cuXXX`** installs `torch` automatically with the specified CUDA version, `torch` version is selected by `uv` from compatible versions in the specific index associated with the chosen CUDA version (options 1 and 2)
2. **with `uv`, creating a thin project that depends on `exllamav3[cuXXX]` and pins a specific `torch` version** — like (1) but `torch` is pinned in the thin project's `pyproject.toml`, see [pinning a specific PyTorch version (optional)](#pinning-a-specific-pytorch-version-optional) for details
3. Manually with `uv pip` or `pip` (options 3 and 4)

The flavor extras (`--extra`) are `cu124`, `cu126`, `cu128`, `cu129`, `cu130`, and `cu132` — pick the one matching your installed CUDA build. Both `uv sync` and `pip install .` build the package in an isolated environment where your `torch` is not visible, so they install the extension sources and compile them at first import (JIT, a few minutes once per torch version). For a precompiled install run `pip install --no-build-isolation .` in an environment that already has `torch`, or use the release wheels. Selecting a flavor installs the matching CUDA build of `torch`.

**Option 1 — Working in the cloned repo directly (`uv sync`):**

```sh
git clone https://github.com/turboderp-org/exllamav3
cd exllamav3
# (Optional) switch to dev branch for latest in-progress features
git checkout dev

uv venv
uv sync --extra cu130
# add --extra examples and/or --extra eval for those extra dependencies
```

**Option 2 — Using `exllamav3` as a dependency from another project (`uv add`):**

```sh
# `uv add` works inside an existing project (a directory with a pyproject.toml).
# `uv init` creates one if you're starting a new project, if integrating into
# an existing project skip `uv init`.
uv init my-project
cd my-project

# local checkout
uv add 'path/to/exllamav3[cu130]'               # non-editable
uv add 'path/to/exllamav3[cu130]' --editable    # editable

# straight from GitHub
uv add 'git+https://github.com/turboderp-org/exllamav3.git[cu130]'                 # default branch
uv add 'git+https://github.com/turboderp-org/exllamav3.git[cu130]' --branch dev    # specific branch
```

**Option 3 — Bring your own `torch` and let `uv` pick the backend automatically:**

```sh
uv venv            # or: uv venv --python-preference only-managed
source .venv/bin/activate
uv pip install torch --torch-backend=auto
uv pip install .
```

`--torch-backend=auto` inspects your system and installs the matching PyTorch CUDA build; see [Automatic backend selection](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection).

**Option 4 — With `pip`:**

On Windows, you also need the `triton-windows` package (declared as a dependency in `pyproject.toml`); the attention, cache and recurrent kernels are Triton and ExLlamaV3 does not import without it.

```sh
# install a CUDA-enabled torch first so it matches your setup, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install .
```

</details>

<details>
<summary>Pinning a specific PyTorch version (optional)</summary>

#### Pinning a specific PyTorch version (optional)

The flavor extra picks the *index*, but by default torch resolves to the latest version on that
index that satisfies `>=2.6.0`. To pin a specific torch version while developing on `exllamav3`,
create a **"thin" project** that consumes your local checkout as an editable install and declares
the exact `torch` version itself. This keeps the pin out of the `exllamav3` pyproject, so
you can change the torch version freely without touching the repo.

```
my-exllamav3-dev/          # thin project (uv init)
├── pyproject.toml
└── src/                  # package sources (auto-generated)
```

In `pyproject.toml`:

```toml
[project]
name = "my-exllamav3-dev"
version = "0.1.0"
description = "Dev environment for exllamav3"
requires-python = ">=3.10.11"
dependencies = [
    "exllamav3[cu130]",   # select correct CUDA version
    "torch==2.13.0",      # pin the exact torch version you need
]

[tool.uv.sources]
exllamav3 = { path = "../exllamav3", editable = true }
```

Adjust `../exllamav3` to point at your local checkout, then a plain `uv sync` sets up an
environment with the correct PyTorch index (routed via the `cuXXX` extra),
the pinned version of `torch` from that index (as long as it exists), and an editable install of `exllamav3` so code
changes apply immediately. Switch CUDA flavors by changing the extra (`exllamav3[cu124]`,
`exllamav3[cu128]`, …) and/or the torch pin in the thin project.

Or, if you're installing torch manually with `uv pip install torch` (e.g. as in Option 3 above),
specify the version directly, e.g. `uv pip install "torch==2.11.0" --torch-backend=auto`.

</details>

After installing with one of the options above, you should be able to run the conversion, eval and 
example scripts from the main repo directory, e.g., `uv run python convert.py -i ...` or, for manual
installations once the venv is active, `python convert.py -i ...`

**Build environment variables**

- `MAX_JOBS`: by default ninja may launch too many processes and run out of system memory for 
compilation. Set this to a reasonable value like 4 in that case.
- `EXLLAMA_NOCOMPILE`: set to install the library without compiling the C++/CUDA extension. Torch
will build/load it at runtime instead.
- `EXLLAMA_EXT_LINEINFO`, `EXLLAMA_EXT_COMPRESS`: see [doc/env_vars.md](doc/env_vars.md).

## Pascal (SM 6.0 / Tesla P100) Support

This fork introduces full native support for **NVIDIA Pascal GP100** architectures (**SM 6.0**, specifically **Tesla P100 16GB PCIe/SXM2** and Quadro GP100), enabling high-throughput inference for EXL3 quantized models on budget 16 GB enterprise accelerators with high memory bandwidth (HBM2 ~732 GB/s).

### Architectural Highlights
- **Native FP16x2 (`__hfma2`) Tensor Core Emulation**:
  Pascal GP100 does not have hardware Tensor Cores (`mma.m16n8k16`). We implemented an emulated matrix-multiply accumulate kernel in PTX using GP100's native 2-way FP16 execution units (`__hfma2` / `hfma2.f32`) and cross-lane warp shuffles (`__shfl_sync` / `exl3_mma_gather`).
- **Interleaved 4-Way FMA Pipeline Scheduling**:
  Pascal GP100 instructions have a 6-cycle read-after-write (RAW) latency. In `exl3_mma_m16n8k16_f16`, the dot-product accumulation is fully interleaved across four independent `half2` accumulators (`d00`, `d01`, `d10`, `d11`), eliminating pipeline stalls and maximizing dual-issue throughput.
- **Vectorized 32-bit Shared Memory LDSM**:
  Replaced 64-bit generic pointer shuffles in `exl3_ldsm4` with 32-bit shared-memory conversions (`__cvta_generic_to_shared` / `__cvta_shared_to_generic`), eliminating half the warp shuffle instructions per tile.
- **Native Vector Half-Accumulator (`EXL3_GEMM_H_ACC`) for SM 6.0 & SM 7.5**:
  Enabled native FP16 MMA accumulation in the inner GEMM loops for Pascal (SM 6.0) and Turing (SM 7.5), folding into FP32 accumulators once per k-slice and bypassing intermediate scalar FP32 conversions.
- **Turing SM 7.5 Accumulator Chaining Fix**:
  Fixed an upstream chaining bug in `ptx.cuh` where the second half of `mma.sync.aligned.m16n8k8.f16.f16.f16.f16` overwrote rather than accumulated into the first sub-tile.
- **Inter-SM Texture Cache Coherency Bypass (`.cg`)**:
  On Pascal GPUs, `const __restrict__` pointers are loaded by default through the read-only Texture Cache (`LDG.E.CI`), which is not coherent between different SMs. In multi-block GEMM reductions, Hadamard transforms, and MoE scheduling, inter-SM updates caused stale reads and NaN corruptions. We implemented inline PTX coherent global loads (`ld.global.cg`) that bypass the texture cache and read coherently through L2.
- **MoE Routing & Fast Decode**:
  Fixed barrier lock stride allocations in `exl3_moe_kernel.cuh` to prevent overlapping lock spaces between thread groups, and added robust atomic fences and ticket resets for SM 6.0.
- **Direct dot-product GEMV (m = 1)**:
  At m = 1 the shuffle-emulated `m16n8k16` wastes 15 of 16 A rows. The GEMV and the single-row MoE coop GEMV tile now take the dot product directly with `__hfma2` and a quad shuffle reduction (-33..43% GEMV kernel time, -9% decode GPU time on the 35B MoE).
- **GEMV everywhere on Pascal, including 5 bpw**:
  The block-pipelined tile GEMM is 10-50x off the memory roofline on Pascal at small m, so every eligible shape (m <= 8) uses the GEMV. 5 bpw tensors (lm_head, sensitive layers) got a GEMV through staged shared-memory tile words (lm_head 5.7 -> 2.1 ms), and fused multi-matrix projections (K <= 5) are unfused on pre-Volta so each projection takes the GEMV.
- **Heterogeneous Multi-GPU Execution**:
  Enables split-model inference across mismatched architectures (e.g., RTX 2060 6GB + Tesla P100 16GB).

### Verified Performance Benchmarks

Tested on a hybrid dual-GPU setup (**RTX 2060 6GB + Tesla P100 16GB PCIe 3.0**):

| Model | Quant | Config | Generation (Decode) | Prefill (Context) | MTP Draft Acceptance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen 3.6 35B-A3B** | 3.00 bpw | Baseline (No MTP, fp16 cache) | **21.34 t/s** | 63.55 t/s | N/A |
| **Qwen 3.6 35B-A3B** | 3.00 bpw | Native MTP (`-mtp -ndt 1 -dds -cq 4`) | **31.48 t/s** | 62.81 t/s | **85.19%** (69/81) |
| **Qwen 3.5 35B-A3B** | 3.09 bpw | Native MTP (`-mtp -ndt 2 -dds`) | **22.65 t/s** | 63.92 t/s | **65.38%** (85/130) |


**Thermals and power matter more than any kernel.** On the reference machine the P100 runs at its minimum power cap (125 W of 250 W TDP, `nvidia-smi -pl`), which together with passive cooling holds the SM clock between roughly 560 and 1330 MHz depending on load history; a hard sustained load on both GPUs even rebooted the host once. The P100 is passively cooled and, without a fan, throttles under sustained load (SM clock drops from 1328 MHz to 700-1190 MHz at ~78 C; `nvidia-smi -lgc` is not supported on this card). The same build measured 21.3 t/s cold and 7.5 t/s hot in `chat.py`. Compare builds interleaved, against unchanged control kernels (torch profiler per-kernel device times), or with the SM clock sampled alongside.

Decode (35B-A3B 3.00 bpw, `-gs 4.0,15.0`), measured as interleaved base/worktree runs with the torch profiler on the P100, normalising EXL3 kernel time by the time of the non-EXL3 kernels in the same step (those are identical in both builds, so the ratio does not depend on clock or temperature): EXL3 kernel time relative to the control falls from 11.2-11.8 (base branch, 3 runs) to 6.3-7.6 (this branch, 3 runs), about 1.8x less EXL3 GPU time per token. Absolute GPU time per token moves with the SM clock (about 33 ms at 1075 MHz, 52 ms at 560 MHz for the whole `chat.py` iteration), so compare builds only this way. Microbenchmarks (GPU time): the 5 bpw lm_head GEMV takes 2.1 ms against 5.7 ms for the best tile-GEMM shape; the m = 1 direct dot-product path makes the GEMV 33-43% faster. `chat.py` with `-cs 32768 -cq 4 -mtp -ndt 1 -dds` (300 tokens, three interleaved base/branch pairs): equal on a cool card (33.07 vs 32.95 and 33.27 vs 32.07 t/s), 26.5 vs 19.6 t/s (+35%) once the card is throttled, so the kernel savings show up when the GPU, not the host, is the limit. Without MTP a single cool run showed no difference (21.6 vs 21.9 t/s). The ~15 ms/token gap between `chat.py` (~45 ms) and a bare `Generator.iterate` loop (~30 ms) is unexplained (not the Rich streamer under `-basic`, loop detection, `torch.cat` or `-no_think`). Reconstruct + cuBLAS beats the tile GEMM from about 64 rows on the P100 (the constant is 144 rows), and fused vs plain reconstruct is within noise; neither is worth changing for a 2048-row prefill chunk. (Since revised: with the HFMA2 GEMM behind the reconstruct path the sm_60 threshold is 64 rows, see the prefill section below.)

MTP (`-mtp -ndt 1`): a verify step runs the GEMVs at m = 2, which used to fall back to the shuffle-emulated MMA (1.3-1.8x the m = 1 cost). A two-row direct dot-product path brings m = 2 to 1.07-1.11x (projections) / 1.2x (lm_head) of m = 1. Per-iteration `Generator.iterate` timing with MTP (profiler script, interleaved base/branch pairs, ~1.9 tokens per iteration in both): 59.1 -> 40.8 ms (32.6 -> 46.8 t/s) at 70-74 C, and 147.6 -> 61.2 ms (12.5 -> 31.2 t/s) at 77-78 C; a third pair at heavy throttling (14.2 vs 13.8 t/s) is not usable. End-to-end `chat.py` runs at that point were all throttled to ~11 t/s on both builds and are not conclusive. Four-row direct products (m = 3..4) measured slower than the MMA path and are not used. Part of the MoE cost at m = 2 is inherent: two tokens activate up to 16 distinct experts instead of 8.

Sustained regime (continuous generation of a long text in a bare `Generator.iterate` loop, this branch, both GPUs, `-gs 4.0,15.0 -cs 32768 -cq 4`, 300 s per run, P100 held at 79 C by the driver's thermal slowdown with the SM clock at ~1000-1130 MHz, windows of 60 s): plain decode 30.3-32.1 t/s (last-3-window means 30.9 and 31.2 in two runs); `-mtp -ndt 1 -dds` 31.3-41.9 t/s (last-3 means 32.3 and 38.5). The MTP gain over plain decode is therefore +4% to +23% when throttled and depends on the text (draft acceptance), while plain decode is stable. Unlike an fp16 matmul, which falls to about 46% of its cool-card rate in the same state, real decode loses far less; the reason is not established (memory clock was not sampled).

Cheaper mul1 decode on sm_60 (`vabsdiff4` byte sum instead of the emulated `dp4a`, and a running prefetch offset instead of a 64-bit `(i + PF) * slice_stride` product in the GEMV / MoE coop loops): SASS instructions per decoded weight in the GEMV inner loop fall from 16.9 to 11.0 (K3, narrow) and 15.6 to 9.9 (K3, wide); results are bit-identical (727 raw outputs compared, same 10 pre-existing test failures on the P100). Real model, `-gs 4.0,15.0 -cs 32768 -cq 4`, P100 kernels only, 3 interleaved pairs: EXL3 kernel time 19.45 -> 14.81 ms per decode step (0.76x) with the non-EXL3 kernels unchanged at 3.06 ms. Sustained generation, 180 s per run, 2 interleaved pairs per mode, P100 at 79 C: plain decode x1.12 and x1.22 (35.2 -> 39.5 and 31.3 -> 38.2 t/s), `-mtp -ndt 1 -dds` x1.17 and x1.24 (35.1 -> 41.2 and 34.2 -> 42.5 t/s). One of the previous-build windows ran at 746 MHz, so the larger ratios are probably inflated by throttling; the conservative figure is about +12% plain and +17% with MTP. The K=2..5 mul1 GEMV kernels gain 25-36%, the mcg codebook only about 3%.

`examples/bench_chat.py` reproduces a `chat.py` session non-interactively (same arguments, prompt format and per-turn statistics, greedy sampling so two bit-identical builds generate the same tokens and the draft acceptance matches) and prints the P100 temperature and SM clock; `examples/bench_chat_ab.sh <dirA> <dirB> [rounds] [bench args]` alternates two builds. Quicksort-in-Go/Rust prompts, 600 new tokens, `-mtp -ndt 1 -dds`, 95.1% draft acceptance in every run: previous build 50.5 / 46.0 t/s, this branch 58.9 / 53.5 t/s in two interleaved rounds (+17% and +16%) while the P100 warmed from 61 to 79 C.

Against the unmodified `pascal-p100` branch (commit `549b7eff`), same benchmark, 3 interleaved rounds, P100 warming from 54 to 79 C: 35.3 / 34.0 / 32.8 t/s on `pascal-p100` against 60.0 / 59.6 / 52.9 t/s on this branch (x1.70, x1.75, x1.62), and prompt processing of the 645-token turn 191 / 187 / 181 t/s against 253 / 249 / 232 t/s. Draft acceptance is 97.0% on `pascal-p100` and 95.1% here: the dot-product paths accumulate in a different order than the shuffle-emulated MMA, so the generated tokens are not identical between the two (they are identical between this branch before and after the vabsdiff4 change).

Hardware and instruction-set notes for the P100 (FP16 rates, what sm_60 lacks, limits) are collected in [doc/pascal_p100.md](doc/pascal_p100.md).

Known Pascal limitation: the Triton MLA absorb kernel (`_mla_absorb_kernel`, used by MLA models such as DeepSeek-style and Ling) needs 64 KB of shared memory at `num_stages=2` and fails on sm_60's 48 KB.

**Prefill on the P100.** A per-kernel torch-profiler breakdown (P100 kernels, 2048-token prompt in one chunk, before these changes, 2.81 s of P100 kernel time) put the routed MoE at 62% (per-expert graph-path tile GEMMs 0.90 s, cuBLAS behind the batched reconstruct tier 0.49 s, index_select / index_add / mul 0.23 s, reconstruct 0.10 s), the GatedDeltaNet Triton kernels at 15% (the `h` recurrence alone 0.37 s), the dense projections (cuBLAS behind reconstruct) at ~14% and attention at 3.5%. The `cc >= 7.0` gate kept the P100 off the fused MoE kernel although that kernel runs and passes `tests/kernels/moe` on sm_60. Four changes, each measured on its own:

- Fused `exl3_moe` prefill kernel on sm_60 (`EXL3_MOE_FUSED_SM60`, default on). Interleaved A/B, 3 rounds: 645-token prompt +12% (chunk 2048) / +19% (chunk 512), 2048-token prompt in 512-token chunks +22%, neutral for a single 2048-token chunk (large experts favour the reconstruct tier there).
- An HFMA2 GEMM for the reconstruct-path GEMMs (`hgemm_f16acc.cu`, namespace `hfma2`; `EXL3_HGEMM_HFMA2=0` keeps cuBLAS). cuBLAS runs fp16 x fp16 GEMMs with fp32 accumulation at the fp32 FMA rate on this card (6-7 TFLOPS); the kernel keeps 32-term fp16 partials and adds them into fp32 registers (the contract of the tensor-core `f16acc` kernel) and reaches ~10 TFLOPS: 1.2-2.1x cuBLAS for grids from 96 64-row blocks. 645 tokens 498 -> 538 t/s, 2048 tokens 867 -> 974 t/s (2 cool-gated runs each).
- 16-column V tiles for the GatedDeltaNet `h` kernel on sm_60: in the model 365 ms -> 34 ms of P100 kernel time per 2048-token prefill (microbenchmark: 21 ms -> 1.6 ms per call at T = 2048), bit-identical outputs.
- Tile-GEMM / reconstruct threshold 64 rows on sm_60 (`EXL3_RECON_THRESHOLD_SM60`): the 132-row tail chunk of a 645-token prompt now reconstructs; 487 -> 512 t/s.

End to end (`examples/bench_chat.py`, `-gs 4.0,15.0 -cs 32768 -cq 4 -mtp -ndt 1 -dds`, a 2880-token first turn and a ~260-token second turn, 3 interleaved rounds against the parent commit, the P100 cooled to 62 C before every run): first-turn prompt processing 486-488 -> 720-727 t/s (x1.49) on a cool card (not measured throttled; a hot interleaved run of the fused MoE change alone was neutral for a single 2048-token chunk), second turn 211-221 -> 332 t/s (x1.5; the first run of a fresh install pays a one-off Triton compile of the new kernel variants). Decode is unchanged (first-turn generate 51.8-53.3 vs 52.2-53.3 t/s at the same 75.2% draft acceptance). Accuracy against a reference with every P100 expert and projection GEMM in fp32 (6 rows x 1536 tokens, single chunks, so the dense projections take the folded reconstruct; measured with `CUDA_LAUNCH_BLOCKING=1` because the cacheless `flash_attn_nc` path hits an unrelated, pre-existing illegal memory access without it): KL 2.19e-3 before, 2.17e-3 after; perplexity 9.0423 before, 9.0412 after, 9.0505 for the reference. The same check in 512-token chunks (16 rows; the dense projections then take the unfused reconstruct with the Hadamards on the activations, as in a chunked prompt): KL 2.07e-3 before, 1.99e-3 after; perplexity 10.3981 before, 10.4019 after, 10.3931 for the reference. Any change of rounding order lands near KL 2e-3 on this model, including the fused MoE kernel alone, so KL against the previous build measures that floor rather than a loss in accuracy. Greedy outputs diverge from the previous build after a few dozen tokens, as they did for the earlier dot-product decode paths. Not kept: the 32-row MoE tiles on sm_60 (neutral) and lower fused-MoE row limits (32 / 64 rows: slower).

Before these changes, prefill (2048-token chunks) was dominated by cuBLAS fp32-accumulate sgemm (~50% of P100 time), the GatedDeltaNet Triton `h` kernel (~17%) and the weight reconstruct. An opt-in fp16-accumulate cuBLAS path (`COMPUTE_16F`, 1.7x faster GEMM kernels) cut total prefill GPU time by only ~4.5% while raising the mean KL divergence to 3e-4..3e-3 versus the fp32-accumulate result, so it was not kept. Qwen3.8-27B (SC quants) was tried and is not usable on this setup. Draft-token MTP did not beat plain decoding on the hot card in these tests.

**Dense Qwen3.5-architecture 27B (Qwen3.6-27B / Qwen3.8-27B, 4.00bpw).** Every linear layer of these checkpoints is mul1 K = 4 (the MTP layer too); the lm_head is mul1 K = 6 (953 MB). With `-gs 4.0,15.0` the RTX 2060 holds 16 layers and the MTP layer, the P100 48 layers and the lm_head, and the two GPUs run one after the other, so a decode step costs the sum of both GPUs' busy time (P100 68.1 + 2060 28.5 ms against 98.1 ms wall per token before these changes; the gaps are ~1.5 ms, so launch overhead is not the limit). Per-kernel profile of a plain decode step on the previous build (torch profiler around `Generator.iterate`, P100 at 70 C, 1037-1189 MHz):

| GPU | item | ms / token |
| :--- | :--- | ---: |
| P100 | K = 4 GEMVs (48 layers, ~9.1 GB) | 46.0 |
| P100 | lm_head, K = 6, **tile GEMM** (the GEMV covered K <= 5) | 16.6 |
| P100 | attention, GatedDeltaNet recurrence, norms, conv, rest | ~5.5 |
| 2060 | K = 4 **tile GEMM / MGEMM** (the sm_75 heuristic rejected the GEMV at these shapes) | 27.4 |

With MTP (`-mtp -ndt 1 -dds`) the lm_head runs twice per iteration (m = 2 verify, m = 1 draft): 33.2 ms of a 126 ms iteration. Changes, each measured on its own:

- 6 bpw GEMV through staged tile words on Pascal (the 5 bpw path generalised): lm_head 16.6 -> 3.3 ms (m = 1), 4.2 ms (m = 2). Plain decode 98.1 -> 82.6 ms per token, MTP 126 -> 101 ms per iteration.
- Turing (RTX 2060): the GEMV for every eligible mul1 K <= 4 shape and same-input projections unfused (`EXL3_GEMV_SM75`); per shape the GEMV beat the autotuned tile kernel by 1.04-1.70x (K = 2..4, m = 1..8). 2060 busy time 26.8 -> 18.0 ms per token, plain decode 82.6 -> 74.1 ms.
- Cheaper sm_60 mul1 decode, bit-identical (raw outputs of `exl3_gemm` at K = 2..6, 8, mcg / mul1, m = 1..100, and of reconstruct compared against the previous build): the two byte sums are packed by one full-rate XMAD.PSL instead of a half-rate PRMT, and the product reads its 16-bit index straight from a half register, so the 4 bpw windows come from the halves of three funnel shifts instead of six BFEs and two masks (`mul1_product_h16`, `codebook.cuh`). K = 4 GEMV at the model's decode shapes: 905-935 -> 860 us summed (m = 1).
- Wide GEMV config whenever n > k on Pascal (measured per shape, m = 1 and 2): 879 -> 844 us summed (m = 1), 995 -> 974 us (m = 2).

End to end, Qwen3.6-27B 4.00bpw, `-gs 4.0,15.0 -cs 32768 -cq 4`, both GPUs, P100 cooled to 62 C before every run, previous build (A, commit `5ded7bd3`) against this branch (B):

| | A | B | |
| :--- | ---: | ---: | :--- |
| `chat.py` plain, user sampler, 600 tokens, 2 interleaved rounds | 10.21 / 10.24 t/s | 14.21 / 14.19 t/s | x1.39 |
| `chat.py -mtp -ndt 1 -dds`, user sampler, 2 rounds | 14.35 / 14.48 t/s (84.6 / 86.3% accepted) | 21.55 / 21.71 t/s (84.6 / 86.3%) | x1.50 |
| `bench_chat.py` plain (greedy, 2 turns x 500 tokens, second turn at 79 C / 582 MHz) | 9.45 t/s | 13.56 t/s | x1.43 |
| `bench_chat.py -mtp -ndt 1 -dds` (greedy) | 14.74 t/s (91.3%) | 22.26 t/s (92.0%) | x1.51 |
| prefill 512 / 2048 tokens, 2 interleaved rounds | 157 / 166, 156 / 165 t/s | 155 / 165, 155 / 166 t/s | 512: ~-1% |
| prefill 8192 / 16384 / 30000, one run each (sustained, 79 C, 405-1100 MHz from 16384 on) | 158 / 93 / 76 t/s | 158 / 92 / 75 t/s | thermal limit |

Qwen3.8-27B 4.00bpw (same layout, one run each, same conditions): `chat.py` plain 10.28 -> 14.26 t/s, MTP 14.29 -> 21.61 t/s (82.9 / 84.1% accepted), prefill 512 / 2048 tokens 160 / 171 -> 160 / 172 t/s. Qwen3.6-35B-A3B 3.00bpw (no-regression check, one run each): plain 44.45 vs 44.21 t/s, MTP 57.91 vs 58.46 t/s.

Prefill sweep by prompt length (`examples/bench_prefill.py`, `-gs 4.0,15.0 -cs 32768 -cq 4`, real text, 2 runs per length averaged, first model started at 57 C, the 27Bs at 72 C because loading and warm-up heat the card; from 8192 tokens on the 27B runs sit at 79 C and 400-1100 MHz, so they are heat-limited numbers):

| prompt tokens | Qwen3.6-35B-A3B 3.00bpw | Qwen3.6-27B 4.00bpw | Qwen3.8-27B 4.00bpw |
| ---: | ---: | ---: | ---: |
| 512 | 440 t/s | 159 | 161 |
| 2048 | 796 | 169 | 170 |
| 8192 | 754 | 124 | 125 |
| 16384 | 680 | 85 | 83 |
| 30000 | 565 (53 s) | 75 (399 s) | 75 (402 s) |

Independent re-check of the dense decode gain on the merged branch (Qwen3.6-27B, `-mtp -ndt 1 -dds`, greedy, 2 x 400 tokens, A = `5ded7bd3`-based build, interleaved): round 1 (P100 from 61 / 72 C) 15.26 -> 22.36 t/s (x1.47), round 2 (both builds at 79 C, 530-1050 MHz) 10.70 -> 14.00 t/s (x1.31); the gain holds hot, the absolute speed does not.

The ~1% at 512 tokens is consistent over both rounds and probably the unfused 2060 projections (two smaller reconstruct + GEMM calls instead of one); `EXL3_GEMV_SM75=0` restores the fused form. Accuracy, teacher-forced decode (q_len = 1 forwards through the m = 1 GEMVs and the K = 6 lm_head GEMV, 768 positions): KL(A || B) mean 3.8e-5, max 5.2e-4, top-1 agreement 99.87%, perplexity 9.7667 (A) vs 9.7733 (B).

Why decode is far from the 732 GB/s bandwidth bound: the m = 1 GEMV is issue-bound. Measured P100 issue rates (doc/pascal_p100.md): IADD / LOP3 / XMAD / HFMA2 full rate, VABSDIFF4 / BFE / SHF / PRMT half rate; the mul1 decode needs three XMADs for the 32-bit product, a half-rate VABSDIFF4 for the byte sum and two HFMA2 halves per weight, about 7 full-rate issue slots per weight before any loads. The loop now runs about 8.6 slots per weight (it was ~10.8), which at ~1.2 GHz caps the GEMV near 240 GB/s; measured 180-250 GB/s per shape. Raising occupancy does not help (min-blocks launch bounds for 2-3x the resident warps: within 1%), which confirms the issue bound. No PTX form gets ptxas to emit a two-XMAD product for a 16-bit index (it rebuilds the 32-bit multiply). The 2060's GEMV runs about 7.3 instructions per weight, mostly on its INT pipe, and reaches 160-240 GB/s of 336.

Ruled out on this setup (measured): `-ndt 2` (m = 3 verify): 16.1 t/s through the MMA path and 17.6 t/s through the four-row direct products, against 21.3 t/s with `-ndt 1` (greedy profiler loop); the m = 3 direct GEMV costs 1.6x the m = 2 one. The sparse draft head (`-sdh 32768`) does not engage here (the draft keeps the full K = 6 head). The Triton prefill attention (`EXL3_SDPA_PREFILL=0`) hits an illegal memory access on sm_60, so prefill attention stays on SDPA.

**Prefill attention on the P100 (`fa60`).** Profile (torch profiler, P100 kernels only, Qwen3.6-27B 4.00bpw, `-gs 4.0,15.0 -cs 32768 -cq 4`, 2048-token chunks, one run per length from 62 C): attention is 0.31 s of 9.7 s of P100 kernel time at 2048 tokens (3.2%), 3.71 of 41.7 s at 8192 (8.9%), 13.99 of 92.7 s at 16384 (15.1%), 62.3 of 252.8 s at 30000 (24.7%). Over 99% of it is one kernel, PyTorch's memory-efficient SDPA (`fmha_cutlassF_f16_aligned_32x128_gmem_sm50`, SIMT fp32, 3.3-3.5 TFLOPS). Ruled out as costs: the `-cq 4` window dequant (0.12 s at 30000), copies / cat (< 0.02 s), the GQA repeat (expand views, no copy) and the mask (`causal_lower_right`, never materialised). `fa60` (`exllamav3_ext/pascal/fa60.cu`, `EXL3_FA60`, default on for sm_60) is an HFMA2 flash-attention forward for head_dim 256: 7.2-7.3 TFLOPS, 2.15-2.45x SDPA on a 2048-row chunk over 2048-30000 keys (24/4 and 16/2 heads). End to end (`examples`-style generator prefill, interleaved ABBA in one process, A = SDPA, B = fa60, P100 cooled to 62 C before every run):

| prompt tokens | A (SDPA) | B (fa60) | B / A | P100 after the runs |
| ---: | ---: | ---: | ---: | :--- |
| 512 | 167.6 t/s | 168.6 | 1.006 | 65 C |
| 2048 | 183.0 | 185.8 | 1.016 | 67 C |
| 8192 | 172.7 | 179.0 | 1.037 | 71 C |
| 16384 | 157.1 | 168.3 | 1.071 (104.3 -> 97.4 s) | 77-78 C |
| 30000 | 110.2 (one run, ends at 79 C / 556 MHz) | not measured | | heat-limited |

Qwen3.6-35B-A3B 3.00bpw (`bench_prefill.py`, `-gs 4.0,15.0 -cs 32768 -cq 4`, `EXL3_FA60=1` against `0` interleaved, 2 rounds, P100 started each run at 61-67 C and ended at 62-70 C): 2048 tokens 821 / 808 against 799 / 781 t/s (+2.5%), 8192 tokens 823 / 814 against 759 / 755 (+8%), 16384 tokens 773 / 773 against 677 / 676 (+14%). The 35B gains more than the 27B because its attention layers (16 / 2 heads) are a larger share of the prefill time. Load check on the merged branch: 12 prefill runs of the 35B plus 3 loads of the 27B with fa60 on, no fault, 0 Xid events in the kernel log.

Precision: on real activations fa60's max abs error against fp32 is 3-12x SDPA's (1.4e-2..7.4e-2 on outputs up to |45|, SDPA 2e-3..2e-2; an fp32 Q K^T variant halves that again but costs +55%); model-level KL against SDPA over 256 scored positions x 2 prompts is 1.25e-3 (8192) / 0.84e-3 (16384), top-1 agreement 99.2-99.4%, against 0.8e-3..5.4e-3 between the fp16 and the `-cq 4` cache. A fault found on the way: with fa60 on, loading the 27B faulted (Xid 31, illegal address surfacing in `exl3_gemv` / `empty_cache`) 3 of 3 times, 0 of 2 with fa60 off, 0 of 2 with expandable segments off, 0 of 3 with `CUDA_LAUNCH_BLOCKING=1`: `empty_cache` with expandable segments unmaps pages after syncing only the owning stream. `free_mem()` now synchronizes every device first; since then 0 faults in 7 fa60-on loads.

Prefill: 85% of the P100 time of a 2048-token chunk is the HFMA2 GEMM (8.69 of 10.2 s; reconstruct 5%, attention 3%, GatedDeltaNet ~4%). Under that kernel the P100 sits at its 125 W cap at 860-940 MHz, and the GEMM reaches 66-73% of the HFMA2 peak at that clock (its loop is ~78% HFMA2 instructions), so prefill is power-bound and not changed here. Attention (PyTorch's memory-efficient SDPA, a SIMT cutlass kernel for head_dim 256) grows to 8.6% of P100 time at 8192 tokens and 16% at 16384; since addressed by `fa60`, below. Long prompts are dominated by heat: a 400 s 30000-token prefill runs at 405-870 MHz once the card reaches 78-79 C.

```sh
# Optimal command for Qwen 3.6 35B-A3B on Dual-GPU (RTX 2060 + Tesla P100):
python examples/chat.py \
  -m models/Qwen3.6-35B-A3B-exl3-3.00bpw \
  -mode qwen35 \
  -gs 4.0,15.0 \
  -cs 32768 \
  -cq 4 \
  -mtp -ndt 1 -dds \
  -tps
```

## Examples

A number of example scripts are provided to showcase the features of the backend and generator. 
For instance, a versatile CLI chatbot:

<p align="center">
  <img src="doc/chatpy.png" width="640" alt="Llama 3.1 8B Instruct quantization benchmark across bits per weight">
</p>

```sh
python examples/chat.py -m <input_dir> -mode <prompt_mode>

# Wealth of options
python examples/chat.py -h
```

## Architecture support

| Model family                                      | HF architecture                                                               | Multimodal | Notes                 |
|---------------------------------------------------|-------------------------------------------------------------------------------|:----------:|-----------------------|
| **AFM**                                           | `ArceeForCausalLM`                                                            |            |                       |
| **AfMoE**                                         | `AfmoeForCausalLM`                                                            |            |                       |
| **Apertus**                                       | `ApertursForCausalLM`                                                         |            |                       |
| **Command-R** etc.                                | `CohereForCausalLM`                                                           |            |                       |
| **Command-A**, **Command-R+** etc.                | `Cohere2ForCausalLM`                                                          |            |                       |
| **DeciLM**, **Nemotron**                          | `DeciLMForCausalLM`                                                           |            |                       |
| **Deepseek V3**                                   | `DeepseekV3ForCausalLM`                                                       |            |                       |
| **Deepseek V4**                                   | `DeepseekV4ForCausalLM`                                                       |     ✓     |                       |
| **dots.llm1**                                     | `Dots1ForCausalLM`                                                            |            |                       |
| **ERNIE 4.5**                                     | `Ernie4_5_ForCausalLM`<br>`Ernie4_5_MoeForCausalLM`                           |            |                       |
| **EXAONE 4.0**                                    | `Exaone4ForCausalLM`                                                          |            |                       |
| **Gemma 2**                                       | `Gemma2ForCausalLM`                                                           |            |                       |
| **Gemma 3**                                       | `Gemma3ForCausalLM`<br>`Gemma3ForConditionalGeneration`                       |     ✓     |                       |
| **Gemma 4**                                       | `Gemma4ForConditionalGeneration`<br>`Gemma4UnifiedForConditionalGeneration`   |     ✓     | E2B/E4B unsupported   |
| **GLM 4**, **GLM 4.6**, etc.                      | `Glm4ForCausalLM`<br>`Glm4MoeForCausalLM`                                     |            |                       |
| **GLM 4.1V**, **GLM 4.5V**                        | `Glm4vForConditionalGeneration`<br>`Glm4vMoeForConditionalGeneration`         |     ✓     |                       |
| **GLM 4.7 Flash**                                 | `Glm4MoeLiteForCausalLM`                                                      |            |                       |
| **GLM 5.2**                                       | `GlmMoeDsaForCausalLM`                                                        |            |                       |
| **GLM 5.3-Flash**                                 | `Glm5NextForConditionalGeneration`                                            |     ✓     |                       |
| **GPT-OSS**                                       | `GptOssForCausalLM`                                                           |            |                       |
| **HyperCLOVAX**                                   | `HyperCLOVAXForCausalLM`<br>`HCXVisionV2ForCausalLM`                          |     ✓     |                       |
| **Hy3**                                           | `HYV3ForCausalLM`                                                             |            |                       |
| **IQuest-Coder**                                  | `IQuestCoderForCausalLM`                                                      |            |                       |
| **Kimi Linear**                                   | `KimiLinearForCausalLM`                                                       |            |                       |
| **Ling 3.x**                                      | `BailingMoeV3ForCausalLM`                                                     |            |                       |
| **Laguna 2.1**                                    | `LagunaForCausalLM`                                                           |            |                       |
| **LFM 2.5**, **LFM2.5-VL**                        | `Lfm2ForCausalLM`<br>`Lfm2MoeForCausalLM`<br>`Lfm2VlForConditionalGeneration` |     ✓     |                       |
| **Llama 1/2/3**,**3.1-Nemotron** etc.             | `LlamaForCausalLM`                                                            |            |                       |
| **MiMo-RL**                                       | `MiMoForCausalLM`                                                             |            |                       |
| **MiMo-V2.6-Flash**                               | `MiMoV2ForCausalLM`                                                           |     ✓     | no audio              |
| **MiniMax-M2**                                    | `MiniMaxM2ForCausalLM`                                                        |            |                       |
| **Mistral**, **Ministral 3**, **Mistral-4** etc.  | `MistralForCausalLM`<br>`Mistral3ForConditionalGeneration`                    |     ✓     |                       |
| **Mixtral**                                       | `MixtralForCausalLM`                                                          |            |                       |
| **NemotronH, Nemotron-3 Nano/Super**              | `NemotronHForCausalLM`                                                        |            |                       |
| **Nemotron 3.5 Bolt**, **Nemotron 3.5 Super VL**  | `NemotronH_Omni_Reasoning_V3`                                                 |     ✓     | Image and video input |
| **Olmo 3.1**                                      | `Olmo3ForCausalLM`                                                            |            |                       |
| **Olmo-Hybrid**                                   | `OlmoHybridForCausalLM`                                                       |            |                       |
| **Phi3**, **Phi4**                                | `Phi3ForCausalLM`                                                             |            |                       |
| **Qwen 2**, **Qwen 2.5**, **Qwen 2.5 VL**         | `Qwen2ForCausalLM`<br>`Qwen2_5_VLForConditionalGeneration`                    |     ✓     |                       |
| **Qwen 3**                                        | `Qwen3ForCausalLM`<br>`Qwen3MoeForCausalLM`                                   |            |                       |
| **Qwen 3-Next**                                   | `Qwen3NextForCausalLM`                                                        |            |                       |
| **Qwen 3-VL**                                     | `Qwen3VLForConditionalGeneration`                                             |     ✓     |                       |
| **Qwen 3-VL MoE**                                 | `Qwen3VLMoeForConditionalGeneration`                                          |     ✓     |                       |
| **Qwen 3.5**                                      | `Qwen3_5ForConditionalGeneration`                                             |     ✓     |                       |
| **Qwen 3.5 MoE**                                  | `Qwen3_5MoeForConditionalGeneration`                                          |     ✓     |                       |
| **Qwen 3.8-Flash-Next**                           | `Qwen4ExpForConditionalGeneration`                                            |     ✓     |                       |
| **Seed-OSS**                                      | `SeedOssForCausalLM`                                                          |            |                       |
| **SmolLM**                                        | `SmolLM3ForCausalLM`                                                          |            |                       |
| **SolarOpen**                                     | `SolarOpenForCausalLM`                                                        |            |                       |
| **Step 3.5 Flash**                                | `Step3p5ForCausalLM`                                                          |            |                       |
| **Step 3.7 Flash**                                | `Step3p7ForConditionalGeneration`                                             |     ✓     |                       |

Always adding more, stay tuned.

## Conversion

To convert a model to EXL3 format, use:

```sh
# Convert model
python convert.py -i <input_dir> -o <output_dir> -w <working_dir> -b <bitrate>

# Resume an interrupted quant job
python convert.py -w <working_dir> -r

# More options
python convert.py -h
```

The working directory is temporary storage for state checkpoints and for storing quantized tensors 
until the converted model can be compiled. It should have enough free space to store an entire copy 
of the output model.

See the [conversion guide](doc/convert.md) for more information, or the 
[self-calibration guide](doc/optimize.md). 

## EXL3 quantization

EXL3 quantization is a streamlined variant of [**QTIP**](https://github.com/Cornell-RelaxML/qtip) from Cornell RelaxML. It aims to make
SOTA quantization available to users on consumer hardware. The conversion process is designed to be
simple and efficient and requires only an input model (in HF format) and a target bitrate. By
computing Hessians on the fly and thanks to a fused Viterbi kernel, the quantizer can convert a 
model in a single step, taking a couple of minutes for smaller models, up to a few hours for larger
ones (70B+) on a single high-end consumer GPU (see the [conversion guide](doc/convert.md)).

For more information, see the [**QTIP**](https://arxiv.org/abs/2406.11235) and [**QuIP#**](https://arxiv.org/abs/2402.04396) papers, as well as this 
[excellent writeup](https://www.together.ai/blog/even-better-even-faster-quantized-llms-with-qtip) on **QTIP** from together.ai.


## Community

You are always welcome to join the [ExLlama discord server](https://discord.gg/NSFwVuCjRq) ←🎮


### 🤗 Models on Hugging Face

Browse the [EXL3 model collection](https://huggingface.co/collections/turboderp/exl3-models-67f2dfe530f05cb9f596d21a) for quantized models. Also shout out to the following lovely
people:

- [ArtusDev](https://huggingface.co/ArtusDev)
- [MikeRoz](https://huggingface.co/MikeRoz)
- [MetaphoricalCode](https://huggingface.co/MetaphoricalCode)
- [Ready.Art](https://huggingface.co/ReadyArt)
- [isogen](https://huggingface.co/isogen/models)


## Acknowledgements

This project owes its existence to a wonderful community of FOSS developers and some very generous
supporters (🐈❤️!) The following projects in particular deserve a special mention:

- [TabbyAPI](https://github.com/theroyallab/tabbyAPI/)
- [PyTorch](https://github.com/pytorch/pytorch)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [QTIP](https://github.com/Cornell-RelaxML/qtip)
- [Transformers](https://github.com/huggingface/transformers)
- [Marlin](https://github.com/IST-DASLab/marlin)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention) (chunked linear-attention prefill kernels, vendored under `exllamav3/vendor/fla`)

<p align="center">
  <img src="doc/cat.png" width="40" alt="">
</p>


