"""EXL3 decode GEMV/GEMM rate microbench for the LFM3 shapes (m=1). Run with CUDA_VISIBLE_DEVICES=<n>.
Reports us per call (CUDA events, 300 calls after warmup, weights cycled over a pool larger than L2) and GB/s over trellis bytes."""
import sys, os
src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src)
sys.path.insert(0, src + "/tests")
import torch
from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import generator, rand_linear

dev = torch.device("cuda:0")
print("device", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
SHAPES = [(2048, 7680), (7680, 2048), (2048, 2048), (2048, 8192), (2048, 65536)]
KS = [float(x) for x in os.environ.get("BG_K", "4,6,8").split(",")]
with torch.inference_mode():
    gen = generator(1)
    for K in KS:
        for (k, n) in SHAPES:
            nbytes = k * n * K / 8
            pool = max(2, int(48e6 / nbytes) + 1)  # defeat L2
            ws = [rand_linear(k, n, K, gen, dev) for _ in range(pool)]
            x = (torch.randn(1, k, generator=gen) * 0.3).half().to(dev)
            y = torch.empty(1, n, dtype=torch.half, device=dev)
            xh = torch.empty_like(x)
            def f(w):
                ext.exl3_gemm(x, w["trellis"], y, w["suh"], xh, w["svh"], 0, False, True, 0)
            for i in range(20): f(ws[i % pool])
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for i in range(300): f(ws[i % pool])
            e1.record(); torch.cuda.synchronize()
            us = e0.elapsed_time(e1) / 300 * 1000
            print(f"K={K:g} {k}x{n}: {us:8.1f} us  {nbytes / 1e9 / (us * 1e-6):6.1f} GB/s  ({nbytes/1e6:.1f} MB)", flush=True)
            del ws
