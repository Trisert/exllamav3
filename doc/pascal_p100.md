# NVIDIA Tesla P100-PCIE-16GB (GP100, cc 6.0): technical fact sheet

> Research notes gathered with a subagent from NVIDIA documentation and `nvidia-smi` on the reference machine. Items marked UNVERIFIED have no primary source behind them; treat them as hypotheses and measure.
>
> One inference not in the sources: the throughput table puts shifts, bitfield extracts and shuffles at half the FP32 FMA rate and 32-bit integer multiplies as several instructions, so the trellis decode (shift/BFE/multiply per weight) is a likely reason the EXL3 GEMV kernels are decode-bound rather than bandwidth-bound on this card. Unmeasured.


Scope: FP16 and kernel-design facts for the P100 (torch device 1, nvidia-smi index 0).
Snapshot date: 2026-10-30. No GPU compute was run.

Source keys:
- [LOCAL-SMI] `nvidia-smi -q -i 0` on this machine (2026-10-30)
- [LOCAL-TORCH] `torch.cuda.get_device_properties(1)` on this machine
- [BLOG] https://developer.nvidia.com/blog/parallelforall/inside-pascal/
- [TUNE] https://docs.nvidia.com/cuda/pascal-tuning-guide/index.html
- [CG9] https://docs.nvidia.com/cuda/archive/9.0/cuda-c-programming-guide/index.html (CUDA 9.0 guide, sec 5.4.1 table)
- [PTX] https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (PTX ISA, current)
- [DS] https://nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-p100/pdf/nvidia-tesla-p100-PCIe-datasheet.pdf (fetch returned only a perf-chart PDF with no spec table; not usable)

## 1. Chip and board

| Item | Value | Source |
|---|---|---|
| Product / arch / CC | Tesla P100-PCIE-16GB, Pascal, cc 6.0 | [LOCAL-SMI], [LOCAL-TORCH] |
| SMs enabled | 56 (GP100 full chip has up to 60) | [LOCAL-TORCH] multi_processor_count=56; [BLOG] 56 SMs; [TUNE] "up to 60 SMs" for GP100 |
| FP32 cores / SM | 64 (56 x 64 = 3584 total) | [BLOG] |
| FP64 units / SM | 32 (2:1 FP32:FP64; 1792 FP64 total) | [BLOG]; [CG9] throughput table: 64-bit FP = 32/clk/SM on cc 6.0 |
| SM layout | 2 processing blocks per SM, each with 32 FP32 cores, a warp scheduler and 2 dispatch units | [BLOG] |
| Warp schedulers / SM | 2 (one per block); each dispatches up to 2 warp instructions/clk | [BLOG]; [TUNE] "two warp schedulers" per SM |
| Base / boost clock (PCIe) | 1126 / 1303 MHz (datasheet-style spec) | [BLOG] PCIe row. NOT re-verified from the official datasheet (fetch failed). |
| Clock reported here | Max graphics/SM 1328 MHz; idle 405 MHz; app-clock default 1189 MHz; P0 state | [LOCAL-SMI] |
| Peak FP32 (derived) | 56 x 64 x 2 x 1.328 GHz = ~9.5 TFLOPS at the reported max clock | derived from above |
| Peak FP16 (derived) | 2x FP32 rate = ~19.0 TFLOPS at 1.328 GHz | derived; 2x ratio from [BLOG], [TUNE] |
| Peak FP32/FP16 as quoted | 9340 GFLOPS FP32 (PCIe, 1303 MHz); FP16 21.2 TFLOPS is the NVLink (SXM2) figure at 1480 MHz | [BLOG] table; FP16 figure cited in secondary-source summary (legitreviews) - UNVERIFIED for PCIe |
| Memory type / bus | HBM2, 4096-bit total (eight 512-bit controllers) | [BLOG] |
| Memory clock reported | 715 MHz (DDR, so ~1.43 Gb/s/pin effective) | [LOCAL-SMI] |
| Memory bandwidth | 732 GB/s peak for 16 GB parts (blog table); the blog body says 720 GB/s. Derived: 4096 bit x 715 MHz x 2 / 8 = 732 GB/s | [BLOG]; derived |
| Memory size | 16 GB (16384 MiB reported; 115 MiB reserved; ~16269 MiB free) | [LOCAL-SMI], [LOCAL-TORCH] total_memory=16269MB |
| L2 cache | 4096 KB (4 MB) | [BLOG]; [LOCAL-TORCH] L2_cache_size=4MB |
| L1 / texture | Unified L1/texture per SM; global loads ARE cached in L1 by default on cc 6.0 (contrast GP104: L2 only) | [TUNE] "GP100 caches global loads in the L1/Texture cache" by default. L1 SIZE: UNVERIFIED (not in fetched text) |
| Data access granularity | 32 B on Pascal, whether or not L1 caches the load | [TUNE] |
| TDP / power limit | Default 250 W; min 125 W; max 250 W; current cap 125 W; instantaneous draw 44 W at snapshot | [LOCAL-SMI] Power readings. Datasheet TDP 250 W (PCIe): [BLOG] |
| nvidia-smi power range | `-pl` accepts 125.00 W to 250.00 W | [LOCAL-SMI] |
| PCIe | Gen3 current and max (device); link width max x16, current x4 as reported at snapshot. Cause NOT determined (could be idle power-saving or slot electrical width; do not assume idle). Host max Gen4. Relevant to host-to-device transfer rates, not kernel speed. | [LOCAL-SMI] PCI section |
| ECC | Enabled (current and pending). Register files, shared memory, L1, L2 and HBM2 are SECDED ECC protected | [LOCAL-SMI] ECC Mode; [BLOG] |
| Cooling | Passive (no fan; nvidia-smi Fan Speed N/A). Passive cooling is the datasheet claim: UNVERIFIED (datasheet fetch failed) | [LOCAL-SMI] Fan Speed: N/A |
| Temps at snapshot | 75 C current; slowdown 82 C; shutdown 85 C | [LOCAL-SMI] |

## 2. Per-SM limits, cc 6.0

| Item | Value | Source |
|---|---|---|
| Max threads / block | 1024 | [CG9] / CUDA guide "up to 1024 threads" (general statement, UNVERIFIED for 6.0 specific table) |
| Warp size | 32 | standard; CUDA guide |
| Max warps / SM | 64 (= 2048 threads / SM) | [TUNE] "maximum number of concurrent warps per SM ... 64" |
| Max blocks / SM | 32 | UNVERIFIED (appendix table not retrieved) |
| Registers / SM | 65,536 x 32-bit (256 KB) | [BLOG]; [TUNE] "64k 32-bit registers" |
| Registers / thread | max 255 | [BLOG] |
| Registers / block | max 65,536 | [BLOG] |
| Shared memory / SM | 64 KB | [BLOG]; [TUNE] |
| Shared memory / block | 48 KB max. NO opt-in above 48 KB on cc 6.0 (Volta adds opt-in). NVIDIA recommends <= 32 KB per block | [TUNE] "Thread-blocks remain limited to 48 KB" |
| Shared memory banks | 32 banks, fixed 4-byte width (8-byte Kepler mode removed) | [TUNE] "fixed four-byte banks" (count of 32 = standard, not quoted) |
| Constant memory | 64 KB total, cached constant path | UNVERIFIED (not retrieved) |

## 3. FP16 specifics

Throughput table (operations per clock per SM), [CG9] sec 5.4.1 "Arithmetic Instructions", cc 6.0 column:

| Operation (cc 6.0) | ops/clk/SM | vs 32-bit FP add/mul/fma (64) |
|---|---|---|
| 16-bit FP add, mul, mul-add (table row; counts each 16-bit op) | 128 | 2x |
| 32-bit FP add, mul, fma | 64 | 1x |
| 64-bit FP add, mul, fma | 32 | 0.5x |
| 32-bit FP rcp/rsqrt/log2/exp2/sin/cos | 16 | 0.25x |
| 32-bit int add/sub | 64 | 1x |
| 32-bit int multiply / multiply-add | "multiple instructions" | no single rate |
| 32-bit int shift | 32 | 0.5x |
| compare, min, max | 32 | 0.5x |
| bit reverse, bitfield extract/insert (BFE/BFI) | 32 | 0.5x |
| bitwise AND/OR/XOR (LOP) | 64 | 1x |
| count leading zeros / find MSB | 16 | 0.25x |
| population count (POPC) | 16 | 0.25x |
| warp shuffle | 32 | 0.5x |
| sum of absolute differences (SAD) | 32 | 0.5x |
| type conversion 8/16-bit int to 32-bit | 16 | 0.25x |
| type conversion, all other | 16 | 0.25x |
| conversions to/from 64-bit | 16 | 0.25x |

Source for all rows: [CG9] sec 5.4.1 table (cc 6.0 column). The table is per SM per clock.
Not in the fetched table: funnel shift (SHF), PRMT, dp4a. Their cc 6.0 rates are UNVERIFIED. Treat as "use 32/clk as a conservative guess, measure it".

Half2 and FP16 instructions (PTX ISA, [PTX] section 9.7.4):
- fma.rn.f16 / fma.rn.f16x2: requires sm_53 or higher (so available on sm_60).
- mul.f16x2: requires sm_53 or higher.
- neg.f16x2 and abs.f16x2: require sm_53 or higher.
- setp.f16x2: requires sm_53 or higher.
- min.f16x2 and max.f16x2: require sm_80 or higher. NOT in PTX for sm_60. Use fma/compare-select or fp32 min/max instead.
- Intrinsic level: CUDA Math API half2 page https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF2__ARITHMETIC.html lists `__hadd2`, `__hmul2`, `__hfma2` (rn/sat/relu variants), `__habs2`, `__hneg2` with no sm-specific note. `__hmax2`/`__hmin2` are not on that page. add.f16x2 PTX and half2<->float2 cvt minimum sm: not in fetched text, UNVERIFIED for sm_60.
- cc 6.0 FP16 rate: [CG9] gives 128 16-bit ops/clk/SM, i.e. 64 half2 instructions/clk/SM. The 2x FP32 claim is in [BLOG] and [TUNE] ("FP16 throughput up to 2x that of FP32").

Contrast on GP104 (cc 6.1, NOT this card): 16-bit FP = 2 ops/clk/SM (1/64 of FP32). This is why the P100 has the FP16 advantage and consumer Pascal does not. Source: [CG9] table; [TUNE] "GP104 ... 1/64th that of FP32".

FP16 with FP32 accumulate: there is NO single-instruction FP16-multiply / FP32-accumulate path on cc 6.0 that the sources support. The throughput table lists only 16-bit-result ops. Practical route: convert half2 to float2 (cvt rate is 16/clk/SM, so this is expensive) and do FP32 FMA (64/clk/SM). Verify the PTX mixed-precision entries before relying on any "f32 accumulate" form: UNVERIFIED.

dp4a / dp2a: [PTX] 9.7.1.24 and 9.7.1.25 "Requires sm_61 or higher". So NOT on cc 6.0 (P100). P100 cannot use dp4a. Throughput: [CG9]/[TUNE] say dp4a runs at FP32 rate on GP104 (sm_61).

Warp shuffle: 32 per clk per SM (= 0.5x FP32 instruction rate) [CG9].

Shared memory load throughput: UNVERIFIED as a number. Expected: 32 banks x 4 B per clk = 128 B/clk/SM, but not confirmed from a fetched source.

## 4. NOT available on cc 6.0 (vs Volta/Turing/Ampere)

| Feature | cc 6.0 status | Source |
|---|---|---|
| Tensor cores / mma.sync (warp MMA) | NOT available. mma.sync is later than sm_60; exact PTX minimum: UNVERIFIED (general knowledge sm_70+) | [PTX] ToC lists mma; sm entry not fetched |
| ldmatrix | NOT available (general knowledge sm_75+; UNVERIFIED) | [PTX] ToC; sm entry not fetched |
| cp.async (async global->shared copy) | NOT available (general knowledge sm_80+; UNVERIFIED) | [PTX] ToC; sm entry not fetched |
| Independent thread scheduling | NOT available (Volta feature). cc 6.0 uses the lockstep warp model; __syncwarp/ *_sync shuffles still exist | general knowledge; UNVERIFIED with a primary cite |
| Async barriers (mbarrier) | NOT available (Ampere) | UNVERIFIED |
| dp4a / dp2a | NOT on sm_60 (requires sm_61) | [PTX] 9.7.1.24, 9.7.1.25 |
| fp16 min/max (f16x2) | NOT on sm_60 (requires sm_80) | [PTX] 9.7.4 |
| bf16 | NOT on sm_60 (bf16 fma requires sm_80+ / mul requires sm_90) | [PTX] 9.7.4 |
| FP16 at high rate on GeForce Pascal (GP10x) | NOT: GP104 16-bit FP at 1/64 rate | [CG9]; [TUNE] |
| Opt-in shared memory above 48 KB | NOT available | [TUNE] |
| Volta-style unified L1/shared-memory pool | NOT: shared memory is a separate 64 KB/SM; global loads are L1-cached by default on GP100 | [TUNE] |
| 8-byte bank mode (Kepler) | NOT available (fixed 4 B on Pascal) | [TUNE] |
| 1024-thread blocks | available (cc 6.0 allows 1024) | CUDA guide, UNVERIFIED for table |

Available on cc 6.0 (confirmed in sources above): fma/mul/neg/abs/setp .f16x2 (sm_53+), 32-bit shifts/LOP/BFE/BFI (table), popc/clz, warp shuffle (table), FP64 at 1/2 FP32 rate. shfl.sync's sm minimum is UNVERIFIED.

## 5. Latency numbers

No trustworthy primary source was retrieved for cycle latencies on GP100.
- FMA dependent latency: UNVERIFIED
- HFMA2 dependent latency: UNVERIFIED
- Shared memory latency: UNVERIFIED
- Global memory latency (DRAM): UNVERIFIED
- L2 hit latency: UNVERIFIED
Recommendation: measure with a dependent-chain microbenchmark, or cite a Pascal dissection paper once located.

## 6. Implications for kernel design

- Use `__half2` FMA (`__hfma2`, `__hmul2`) for FP16 math: this is the only path to the 2x rate in the throughput table. Scalar half ops give no gain over FP32.
- K-chunked FP16 accumulation: keep FP16 partial sums over a bounded K chunk and flush to an FP32 accumulator per chunk. The chunk length is a tuning parameter (not set by any source here).
- For FP32 accumulation, use FP32 FMA (64/clk/SM) with conversions on the input side. Conversions run at 16/clk/SM, a quarter of the FP32 rate, so count them in the cost model.
- Do not use `min.f16x2`/`max.f16x2` in PTX (sm_80+). Use FP32 min/max or compare-select.
- Do not use dp4a on this card (sm_61+). Int8 dot products need FP32 or half2 FMA after conversion.
- Register budget: 65,536 regs/SM. At 64 regs/thread that is 1024 threads (50% of the 2048-thread SM limit); at 32 regs/thread it is 2048 threads (full occupancy).
- Shared memory: 64 KB/SM and 48 KB/block with no opt-in. Two blocks of 32 KB fill the SM's shared memory; NVIDIA recommends at most 32 KB per block.
- Latency is undocumented here (UNVERIFIED). Provide enough independent accumulators per thread to hide it, and tune by measurement rather than a fixed count.
- Warp shuffle runs at 32/clk/SM, half the FP32 instruction rate. Shuffle-heavy emulation of MMA-style reductions costs throughput; prefer FMA paths.
- Memory-bound vs compute-bound: FP32 peak is ~9.5 TFLOPS and HBM2 peak is 732 GB/s (16 GB parts). Measure the achievable figures on this card; the 125 W cap (half of the 250 W default) may throttle sustained FMA work, so measure at the cap. ECC is enabled; its bandwidth cost is UNVERIFIED.

## UNVERIFIED items that matter

1. Cycle latencies (FMA, HFMA2, shared, global, L2): no primary source found.
2. cc 6.0 max blocks/SM (32), constant memory size (64 KB), L1 size: not retrieved from the appendix table.
3. Funnel shift (SHF), PRMT, dp4a and mma/ldmatrix/cp.async minimum sm: not retrieved (dp4a sm_61 IS confirmed; mma/ldmatrix/cp.async sm minima are UNVERIFIED).
4. PCIe base/boost (1126/1303 MHz) and passive-cooling claim: the official datasheet fetch returned only a perf chart; cited from the blog table only.
5. FP16 peak TFLOPS of 21.2 is the SXM2 figure; PCIe FP16 peak is derived, not quoted.
6. add.f16x2 and cvt half2<->float2 availability on sm_60: not retrieved.
