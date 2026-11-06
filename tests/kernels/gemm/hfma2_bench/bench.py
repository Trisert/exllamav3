"""
hfma2 prefill GEMM microbenchmark on the P100 (scratch variant module + the built extension), ABAB interleaved.
Variants: ext:BM (exllamav3_ext.hgemm_hfma2), vN:BM (vars.cu, JIT-built), cublas (ext.hgemm, fp32 compute).
Efficiency = GEMM rate / rate of the register-only HFMA2 calib kernel timed right before and after each launch, so
it is independent of the (thermally throttled) clock; ~96.5% of the nominal 1328 MHz peak is what calib reaches.
Run: TORCH_CUDA_ARCH_LIST=6.0 CC=gcc-14 CXX=g++-14 PYTHONPATH=<tree> python bench.py --variants v0:64,ext:64 --M 2048
Each timing: warmup, `iters` launches between CUDA events, then nvidia-smi temp / SM clock.
"""
import argparse, os, subprocess, sys, time, statistics
import torch

p = argparse.ArgumentParser()
p.add_argument("--variants", default = "ext:64,ext:128")
p.add_argument("--M", default = "2048,4096")
p.add_argument("--shapes", default = "all")
p.add_argument("--reps", type = int, default = 3)
p.add_argument("--iters", type = int, default = 6)
p.add_argument("--warmup", type = int, default = 2)
p.add_argument("--out", default = "model", help = "model / half / float")
p.add_argument("--check", action = "store_true")
p.add_argument("--vars", action = "store_true", help = "build/load the scratch variant module")
p.add_argument("--defs", default = "")
p.add_argument("--calib_iters", type = int, default = 3000)
p.add_argument("--custom", default = "", help = "batch:M:K:N[:float],... instead of the 27B shapes / --M")
args = p.parse_args()

SHAPES = {
    # K, N, output dtype of the 27B module (gate/up interm fp16, residual-side and GDN qkv/z outputs fp32)
    "gate_up": (5120, 17408, "half"), "down": (17408, 5120, "float"), "gdn_qkv": (5120, 10240, "float"),
    "gdn_z": (5120, 6144, "float"), "o_out": (6144, 5120, "float"), "attn_q": (5120, 12288, "half"),
    "attn_kv": (5120, 1024, "half"),
}
shapes = SHAPES if args.shapes == "all" else {k: SHAPES[k] for k in args.shapes.split(",")}
cases = []   # (name, batch, M, K, N, dtype)
if args.custom:
    for e in args.custom.split(","):
        f = e.split(":")
        cases.append((f"{f[0]}x{f[1]}x{f[2]}x{f[3]}", int(f[0]), int(f[1]), int(f[2]), int(f[3]), f[4] if len(f) > 4 else "half"))
else:
    for M in [int(x) for x in args.M.split(",")]:
        for sname, (K, N, sdt) in shapes.items(): cases.append((sname, 1, M, K, N, sdt))

dev = None
for i in range(torch.cuda.device_count()):
    if torch.cuda.get_device_capability(i) == (6, 0): dev = torch.device(f"cuda:{i}")
assert dev is not None, "no sm_60 device"
torch.cuda.set_device(dev)
print("device", dev, torch.cuda.get_device_name(dev))
# nvidia-smi index of the P100 by PCI bus id
bus = torch.cuda.get_device_properties(dev).pci_bus_id if hasattr(torch.cuda.get_device_properties(dev), "pci_bus_id") else None
smi_idx = os.environ.get("SMI_IDX", "0")

ext = None
if any(v.startswith("ext") or v == "cublas" for v in args.variants.split(",")):
    from exllamav3.ext import exllamav3_ext as ext
vm = None
if True:
    from torch.utils.cpp_extension import load
    here = os.path.dirname(os.path.abspath(__file__))
    defs = [f"-D{d}" for d in args.defs.split(",") if d]
    vm = load(name = "hfma2_vars" + "".join(d.replace("-D", "_") for d in defs), sources = [os.path.join(here, "vars.cu")],
              extra_cuda_cflags = ["-O3", "-gencode=arch=compute_60,code=sm_60", "-Xptxas=-v", "-ccbin=/usr/bin/g++-14", "--use_fast_math", "-Xptxas=-dlcm=cg", "-std=c++20"] + defs,
              extra_include_paths = [here], verbose = False)

def smi():
    r = subprocess.run(["nvidia-smi", "-i", smi_idx, "--query-gpu=temperature.gpu,clocks.sm,power.draw",
                        "--format=csv,noheader,nounits"], capture_output = True, text = True).stdout.strip()
    t, c, pw = [x.strip() for x in r.split(",")]
    return int(t), int(c), float(pw)

def fn_for(var, a, b, c):
    if var == "cublas": return lambda: ext.hgemm(a, b, c)
    name, bm = var.split(":")
    bm = int(bm)
    if name == "ext": return lambda: ext.hgemm_hfma2(a, b, c, bm)
    v = int(name[1:])
    return lambda: vm.run(v, a, b, c, bm)

variants = args.variants.split(",")
PEAK = 56 * 64 * 4 * 1.328e-3   # TFLOP/s HFMA2 peak at 1328 MHz
sink = torch.zeros(256, device = dev, dtype = torch.half)
res = {}
for _ in range(4):
    e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
    e0.record(); fl = vm.calib(args.calib_iters, sink); e1.record(); e1.synchronize()
    r = fl / e0.elapsed_time(e1) / 1e9
    t, clk, pw = smi()
    print(f"calib {e0.elapsed_time(e1):.2f} ms {r:.2f} TFLOP/s = {r / (56 * 256) * 1e6:.0f} MHz-equivalent; smi {clk} MHz {t} C")
torch.manual_seed(0)
for sname, bat, M, K, N, sdt in cases:
    if True:
        odt = torch.half if (args.out if args.out != "model" else sdt) == "half" else torch.float
        lead = (bat,) if bat > 1 else ()
        a = torch.empty(lead + (M, K), device = dev, dtype = torch.half).normal_(std = 0.5)
        b = torch.empty(lead + (K, N), device = dev, dtype = torch.half).normal_(std = 0.5 / K ** 0.5 * 4)
        c = torch.empty(lead + (M, N), device = dev, dtype = odt)
        flop = 2.0 * bat * M * N * K
        if args.check:
            ref = torch.matmul(a.float(), b.float())
            outs = {}
            for var in variants:
                c.fill_(float("nan"))
                fn_for(var, a, b, c)()
                torch.cuda.synchronize()
                outs[var] = c.clone()
                rel = float(((c.float() - ref).square().mean() / ref.square().mean()).sqrt())
                print(f"  check {sname} M={M} {var}: rel RMS vs fp32 {rel:.3e}  nan {bool(c.isnan().any())}")
            v0 = variants[0]
            for var in variants[1:]:
                bits = torch.int16 if odt == torch.half else torch.int32
                eq = torch.equal(outs[var].view(bits), outs[v0].view(bits))
                d = float((outs[var].float() - outs[v0].float()).abs().max())
                print(f"  check {sname} M={M} {var} vs {v0}: bit-identical {eq}, max abs diff {d:.3e}")
            del ref
        evs = [torch.cuda.Event(enable_timing = True) for _ in range(4)]
        def calib_rate(e0, e1):
            e0.record(); fl = vm.calib(args.calib_iters, sink); e1.record()
            return fl, e0, e1
        for var in variants:
            for _ in range(args.warmup): fn_for(var, a, b, c)()
        # Per launch: calib, gemm, calib; efficiency = gemm rate / mean calib rate (HFMA2 issue peak at that clock)
        for rep in range(args.reps):
            for var in variants:
                f = fn_for(var, a, b, c)
                effs, tfs = [], []
                for it in range(args.iters):
                    fl0, _, _ = calib_rate(evs[0], evs[1])
                    g0 = torch.cuda.Event(enable_timing = True); g1 = torch.cuda.Event(enable_timing = True)
                    g0.record(); f(); g1.record()
                    fl1, _, _ = calib_rate(evs[2], evs[3])
                    evs[3].synchronize()
                    c0 = fl0 / evs[0].elapsed_time(evs[1]) / 1e9
                    c1 = fl1 / evs[2].elapsed_time(evs[3]) / 1e9
                    tf = flop / g0.elapsed_time(g1) / 1e9
                    effs.append(tf / (0.5 * (c0 + c1))); tfs.append(tf)
                t, clk, pw = smi()
                eff = statistics.median(effs); tf = statistics.median(tfs)
                res.setdefault((M, sname, var), []).append((eff, clk, t, tf))
                print(f"M={M:5d} {sname:8s} K={K:5d} N={N:5d} {str(odt)[6:]:7s} {var:8s} rep {rep}: eff {eff * 100:5.1f}% "
                      f"(-> {eff * PEAK:5.2f} TFLOP/s @1328) raw {tf:6.2f} TFLOP/s  {t} C {clk} MHz {pw:.0f} W", flush = True)
        del a, b, c

print(f"\nsummary: efficiency vs calib HFMA2 peak (median), TFLOP/s at 1328 MHz equivalent, ratio = median per-rep ratio vs first")
for sname, bat, M, K, N, sdt in cases:
    line = f"M={M:5d} {sname:8s}"
    r0 = res[(M, sname, variants[0])]
    for var in variants:
        r = res[(M, sname, var)]
        ratio = statistics.median(x[0] / y[0] for x, y in zip(r, r0))
        e = statistics.median(x[0] for x in r)
        line += f" | {var} {e * 100:5.1f}% {e * PEAK:5.2f}T x{ratio:.3f}"
    print(line)
