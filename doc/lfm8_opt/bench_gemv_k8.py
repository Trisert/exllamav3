"""m = 1 GEMV microbench for the LFM3-1.7B 8 bpw mul1 shapes: us per call (CUDA events, 200 calls after warmup) and GB/s
over the trellis bytes, under the GEMV mode given in EXL3_GEMV (1 heuristic, 3 force narrow, 4 force wide; read per
call) and EXL3_GEMV_BALANCE (read once: run one process per value). usage: bench_gemv_k8.py [kxn ...]"""
import sys, os
src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, src)
sys.path.insert(0, src + "/tests")
import torch
from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import generator, rand_linear

dev = torch.device("cuda:0")
K = int(os.environ.get("BG_K", "8"))
SHAPES = [(2048, 7680), (7680, 2048), (2048, 6144), (2048, 2048), (2048, 512)]
if len(sys.argv) > 1:
    SHAPES = [tuple(int(v) for v in s.split("x")) for s in sys.argv[1:]]
with torch.inference_mode():
    gen = generator(1)
    for (k, n) in SHAPES:
        w = rand_linear(k, n, K, gen, dev)
        x = (torch.randn(1, k, generator = gen) * 0.3).half().to(dev)
        y = torch.empty(1, n, dtype = torch.half, device = dev)
        xh = torch.empty_like(x)
        row = []
        for mode in ("1", "3", "4"):
            os.environ["EXL3_GEMV"] = mode
            f = lambda: ext.exl3_gemm(x, w["trellis"], y, w["suh"], xh, w["svh"], 0, False, True, 0)
            for _ in range(20): f()
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
            e0.record()
            for _ in range(200): f()
            e1.record(); torch.cuda.synchronize()
            us = e0.elapsed_time(e1) / 200 * 1000
            row.append(f"{['heur', 'narrow', 'wide'][['1', '3', '4'].index(mode)]} {us:6.1f} us {k * n * K / 8 / 1e3 / us:5.0f} GB/s")
        print(f"K={K} {k:5d}x{n:<5d} bal={os.environ.get('EXL3_GEMV_BALANCE', '1')}: " + " | ".join(row), flush = True)
