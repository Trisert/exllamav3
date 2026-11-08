"""
Standalone bench for hgemv_f16.cu (compiles only that file): time, GB/s and error against an fp32 reference on the
Ornith shapes, m = 1, 2, 4. Usage: python bench.py <tag> [path to hgemv_f16.cu]

Run it on an idle GPU (nothing else on the P100); CUDA_VISIBLE_DEVICES selects the card.
"""
import sys, os, torch
from torch.utils.cpp_extension import load

here = os.path.dirname(os.path.abspath(__file__))
ext_dir = os.path.abspath(os.path.join(here, "../../../../exllamav3/exllamav3_ext"))
tag = sys.argv[1] if len(sys.argv) > 1 else "x"
src = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ext_dir, "hgemv_f16.cu")
m = load(name = "hgemv_bench_" + tag, sources = [os.path.join(here, "binding.cpp"), src],
         extra_include_paths = [ext_dir], extra_cuda_cflags = ["-O3"], extra_cflags = ["-O3"], verbose = False)

dev = torch.device("cuda:0")
shapes = [(2048, 8192), (2048, 4096), (4096, 2048), (2048, 512), (512, 2048), (2048, 248320)]


def bench(fn, n = 200):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n * 1000


torch.manual_seed(0)
print(torch.cuda.get_device_name(0))
# SPLITS=1 sweeps ksplit (0 = automatic) at M = 2; the K split changes only the fp32 summation order
sweep = os.environ.get("SPLITS") == "1"
for K, N in shapes:
    w = torch.empty((K, N), device = dev, dtype = torch.half).normal_(std = 0.02)
    for M in ((2,) if sweep else (1, 2, 4)):
        x = torch.empty((M, K), device = dev, dtype = torch.half).normal_()
        y = torch.empty((M, N), device = dev, dtype = torch.half)
        ref = x.float() @ w.float()
        for ks in ((0, 1, 2, 3, 4, 6, 8, 12, 16) if sweep else (0,)):
            if ks * M * N > (1 << 20): continue
            m.hgemv_f16(x, w, y, ks)
            err = (y.float() - ref).abs().max().item() / ref.abs().max().item()
            us = bench(lambda: m.hgemv_f16(x, w, y, ks))
            print(f"K={K:5d} N={N:6d} M={M} ksplit={ks:2d}  {us:8.1f} us  {K*N*2/us/1e3:6.0f} GB/s  relerr {err:.2e}", flush = True)
