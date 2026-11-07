"""Decode GEMV microbench on the P100: 5 bpw mul1 shapes of Qwen3.8-Flash-Next, m = 1 and 4.
Reports us per call (CUDA events, 200 calls after warmup) and achieved GB/s over the trellis bytes."""
import sys, os
src = "/home/nicola/exllamav3/.claude/worktrees/draft-penalty"
sys.path.insert(0, src)
sys.path.insert(0, src + "/tests")
import torch
from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import generator, rand_linear

dev = torch.device("cuda:0")
K = float(os.environ.get("BG_K", "5"))
SHAPES = [(2560, 10240), (2560, 6144), (6144, 2560), (2560, 4096), (4096, 2560), (2560, 8192), (2560, 2048), (2048, 2560)]
if len(sys.argv) > 1:
    SHAPES = [tuple(int(v) for v in s.split("x")) for s in sys.argv[1:]]
with torch.inference_mode():
    gen = generator(1)
    for (k, n) in SHAPES:
        w = rand_linear(k, n, K, gen, dev)
        for m in (1, 4):
            x = (torch.randn(m, k, generator=gen) * 0.3).half().to(dev)
            y = torch.empty(m, n, dtype=torch.half, device=dev)
            xh = torch.empty_like(x)
            f = lambda: ext.exl3_gemm(x, w["trellis"], y, w["suh"], xh, w["svh"], 0, False, True, 0)
            for _ in range(20): f()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(200): f()
            e1.record(); torch.cuda.synchronize()
            us = e0.elapsed_time(e1) / 200 * 1000
            gb = k * n * K / 8 / 1e9 / (us * 1e-6)
            print(f"GEMV K={K} k={k} n={n} m={m}: {us:7.1f} us  {gb:6.1f} GB/s  groups32={n // 32}", flush=True)
