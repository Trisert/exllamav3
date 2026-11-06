"""
Sustained (thermally throttled) wall-clock throughput: back-to-back launches of one variant for --secs seconds, then
the next, ABAB. Pair with the 2 s nvidia-smi log for clock / power per block.
python hot.py --variants v0:64,ext:64 --blocks 4 --secs 20 [--M 2048 --K 5120 --N 17408]
"""
import argparse, os, time, datetime
import torch

p = argparse.ArgumentParser()
p.add_argument("--variants", default = "v0:64,ext:64")
p.add_argument("--blocks", type = int, default = 4)
p.add_argument("--secs", type = float, default = 20.0)
p.add_argument("--M", type = int, default = 2048)
p.add_argument("--K", type = int, default = 5120)
p.add_argument("--N", type = int, default = 17408)
args = p.parse_args()

dev = next(torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count()) if torch.cuda.get_device_capability(i) == (6, 0))
torch.cuda.set_device(dev)
from exllamav3.ext import exllamav3_ext as ext
from torch.utils.cpp_extension import load
here = os.path.dirname(os.path.abspath(__file__))
vm = load(name = "hfma2_vars", sources = [os.path.join(here, "vars.cu")],
          extra_cuda_cflags = ["-O3", "-gencode=arch=compute_60,code=sm_60", "-Xptxas=-v", "-ccbin=/usr/bin/g++-14",
                               "--use_fast_math", "-Xptxas=-dlcm=cg", "-std=c++20"], extra_include_paths = [here])
a = torch.randn(args.M, args.K, device = dev, dtype = torch.half)
b = torch.randn(args.K, args.N, device = dev, dtype = torch.half) / args.K ** 0.5
c = torch.empty(args.M, args.N, device = dev, dtype = torch.half)
flop = 2.0 * args.M * args.N * args.K
def fn(var):
    name, bm = var.split(":")
    if name == "ext": return lambda: ext.hgemm_hfma2(a, b, c, int(bm))
    return lambda: vm.run(int(name[1:]), a, b, c, int(bm))
variants = args.variants.split(",")
for blk in range(args.blocks):
    var = variants[blk % len(variants)]
    f = fn(var); f(); torch.cuda.synchronize()
    t0 = time.time(); n = 0
    start = datetime.datetime.now().strftime("%H:%M:%S")
    while time.time() - t0 < args.secs:
        for _ in range(4): f()
        torch.cuda.synchronize(); n += 4
    dt = time.time() - t0
    print(f"block {blk} {var:8s} {start} -> {datetime.datetime.now().strftime('%H:%M:%S')}: {n} launches, "
          f"{flop * n / dt / 1e12:.2f} TFLOP/s wall", flush = True)
