"""
EXL3 4 bpw GEMV (quant/exl3_gemv.cu), narrow vs wide config on the Ornith projection shapes at m = 1, 2 (and 4).
EXL3_GEMV is read per call: 1 = heuristic, 3 = force narrow, 4 = force wide. Usage (P100):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python bench_exl3_gemv.py
"""
import os, sys
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "../../.."))        # tests/ for testlib
src = os.environ.get("EXL_SRC")
if src: sys.path.insert(0, src)
import numpy as np, torch
from exllamav3.ext import exllamav3_ext as ext
from testlib import trellis as tref
from testlib.exl3 import generator, rand_scale

dev = torch.device("cuda:0")
K = 4
shapes = [(4096, 2048), (2048, 4096), (2048, 8192), (2048, 12288), (2048, 512), (2048, 1024)]   # (k, n)


def bench(fn, n = 300):
    for _ in range(20): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n * 1000


print(torch.cuda.get_device_name(0), "SMs", torch.cuda.get_device_properties(0).multi_processor_count)
for k, n in shapes:
    g = generator(1)
    tr = tref.random_packed((k // 16) * (n // 16), K, np.random.default_rng(1)).reshape(k // 16, n // 16, -1); tr = torch.from_numpy(tr.astype(np.int16)).to(dev)
    suh, svh = rand_scale(k, g).to(dev), rand_scale(n, g).to(dev)
    for m in (1, 2, 4):
        A = torch.empty((m, k), device = dev, dtype = torch.half).normal_()
        C = torch.empty((m, n), device = dev, dtype = torch.half)
        A_had = torch.empty_like(A)
        row = f"k={k:5d} n={n:5d} m={m}"
        for name, mode in (("heur", "1"), ("narrow", "3"), ("wide", "4")):
            os.environ["EXL3_GEMV"] = mode
            us = bench(lambda: ext.exl3_gemm(A, tr, C, suh, A_had, svh, -1, False, True, 0))
            row += f" | {name} {us:6.1f} us {k*n/us/1e3:5.0f} Gw/s"
        print(row, flush = True)
