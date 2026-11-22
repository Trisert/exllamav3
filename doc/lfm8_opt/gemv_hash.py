"""Bit-exactness probe: m = 1 exl3_gemm outputs for fixed random K-bit mul1 weights (LFM3 shapes), printed as one hash
per shape. Run once per build (EXL_SRC) and diff. usage: BG_K=8 python gemv_hash.py"""
import sys, os, hashlib
src = os.environ.get("EXL_SRC") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, src)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tests"))
import torch
from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import generator, rand_linear

dev = torch.device("cuda:0")
K = int(os.environ.get("BG_K", "8"))
with torch.inference_mode():
    gen = generator(3)
    for (k, n) in [(2048, 7680), (7680, 2048), (2048, 6144), (2048, 2048), (2048, 512)]:
        w = rand_linear(k, n, K, gen, dev)
        x = (torch.randn(1, k, generator = gen) * 0.3).half().to(dev)
        y = torch.empty(1, n, dtype = torch.half, device = dev)
        xh = torch.empty_like(x)
        ext.exl3_gemm(x, w["trellis"], y, w["suh"], xh, w["svh"], 0, False, True, 0)
        torch.cuda.synchronize()
        print(f"K={K} {k}x{n} {hashlib.sha1(y.cpu().numpy().tobytes()).hexdigest()[:16]}")
