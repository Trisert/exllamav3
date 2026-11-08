"""Microbenchmark of the decode MoE kernel pair (shared + routed, exl3_moe_coop_pair) at the Ornith serving
shape, without a model: builds synthetic 4 bpw mul1 experts and times the launch. Also dumps the EXL3_MOE_TRACE
per-item records and runs analyze.py on them when --trace is passed.

  .venv/bin/python tests/kernels/moe/coop_trace/bench_dual.py [--bsz 2] [--E 256] [--topk 8] [--iters 300] [--trace]

The shape defaults are Ornith-1.5-35B-A3B: Hi 2048, I 512, Ho 2048, 256 experts, top-8, bsz 2 (m = 2, the
-ndt 1 verification pass). The scratch is sized for the fused (persistent) path; passing a base-length counter
scratch to the same entry point runs the two-launch path instead, for comparison.
"""
import os, sys, time, argparse, json
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
OUT = os.environ.get("COOP_TRACE_DIR", os.path.join(ROOT, "tmp", "coop_trace"))
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import torch
from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import rand_scale, rand_trellis
from testlib.moe import ptr_table


def _proj(k, n, K, E, gen, device):
    t = [rand_trellis(k, n, K, gen, device) for _ in range(E)]
    su = [rand_scale(k, gen, device) for _ in range(E)]
    sv = [rand_scale(n, gen, device) for _ in range(E)]
    return (ptr_table(t, device), ptr_table(su, device), ptr_table(sv, device)), (t, su, sv)


def _scratch(smax, rows, Hi, I, Ho, device, fused):
    base = smax * (I // 128) + rows * (Ho // 128) + 2 * smax + 3
    return [
        torch.empty((smax, Hi), dtype = torch.half, device = device),
        torch.empty((smax, Hi), dtype = torch.half, device = device),
        torch.empty((smax, 1, I), dtype = torch.half, device = device),
        torch.empty((smax, 1, I), dtype = torch.half, device = device),
        torch.empty((smax, 1, I), dtype = torch.half, device = device),
        torch.empty((smax, 1, Ho), dtype = torch.float, device = device),
        torch.zeros((base + (smax + 2 if fused else 0),), dtype = torch.int, device = device),
    ]


def build(device, seed, K, cb, bsz, E, topk, Hi, I, Ho, Ksh, overlap = 0.5):
    gen = torch.Generator().manual_seed(seed)
    keep = []

    def tabs(E_, K_):
        out = []
        for k, n in ((Hi, I), (Hi, I), (I, Ho)):
            t, kp = _proj(k, n, K_, E_, gen, device)
            out += list(t); keep.append(kp)
        return out

    rt = tabs(E, K)
    st = tabs(1, Ksh)
    x = (torch.randn((bsz, Hi), generator = gen) * 0.05).half().to(device)
    # Consecutive verification tokens pick correlated experts: row 0 gets a random top-k, each following row keeps
    # round(overlap * topk) of them and draws the rest (overlap 0 = independent picks)
    rows = [torch.randperm(E, generator = gen)[:topk]]
    for _ in range(1, bsz):
        keep = int(round(overlap * topk))
        rest = torch.randperm(E, generator = gen)
        rest = rest[~torch.isin(rest, rows[-1][:keep])][:topk - keep]
        rows.append(torch.cat((rows[-1][:keep], rest)))
    sel = torch.stack(rows).to(device)
    rw = (torch.rand((bsz, topk), generator = gen) * 0.3).half().to(device)
    sh_w = (torch.randn((Hi,), generator = gen) * 0.05).half().to(device)
    sc_r = _scratch(8 * topk, bsz, Hi, I, Ho, device, True)
    sc_s = _scratch(8, 8, Hi, I, Ho, device, True)
    b_r = _scratch(8 * topk, bsz, Hi, I, Ho, device, False)
    b_s = _scratch(8, bsz, Hi, I, Ho, device, False)
    out = torch.full((8, Ho), float("nan"), dtype = torch.float, device = device)
    sh_buf = torch.zeros((8, Hi), dtype = torch.float, device = device)
    return dict(x = x, sel = sel, rw = rw, sh_w = sh_w, rt = rt, st = st, sc_r = sc_r, sc_s = sc_s,
                b_r = b_r, b_s = b_s, out = out, sh_buf = sh_buf, keep = keep)


def call(t, Hi, bsz, K, Ksh, cb, fused = True):
    sc_r = t["sc_r"] if fused else t["b_r"]
    sc_s = t["sc_s"] if fused else t["b_s"]
    ext.exl3_moe_coop_pair(t["x"], t["sel"], t["rw"], Hi, t["rt"], K, sc_r,
                           t["st"], Ksh, sc_s, cb == 1, cb == 2, t["out"][:bsz], t["sh_buf"], t["sh_w"])


def timeit(t, Hi, bsz, K, Ksh, cb, iters, fused = True):
    for _ in range(20): call(t, Hi, bsz, K, Ksh, cb, fused)
    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing = True); ev1 = torch.cuda.Event(enable_timing = True)
    ev0.record()
    for _ in range(iters): call(t, Hi, bsz, K, Ksh, cb, fused)
    ev1.record(); torch.cuda.synchronize()
    us = ev0.elapsed_time(ev1) * 1000 / iters
    # per-call spread
    times = []
    for _ in range(30):
        a = torch.cuda.Event(enable_timing = True); b = torch.cuda.Event(enable_timing = True)
        a.record(); call(t, Hi, bsz, K, Ksh, cb, fused); b.record(); torch.cuda.synchronize()
        times.append(a.elapsed_time(b) * 1000)
    times.sort()
    return us, times[len(times) // 2], times[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type = int, default = 1)
    ap.add_argument("--bsz", type = int, default = 2)
    ap.add_argument("--E", type = int, default = 256)
    ap.add_argument("--topk", type = int, default = 8)
    ap.add_argument("--Hi", type = int, default = 2048)
    ap.add_argument("--I", type = int, default = 512)
    ap.add_argument("--Ho", type = int, default = 2048)
    ap.add_argument("--K", type = float, default = 4.0)
    ap.add_argument("--Ksh", type = float, default = 4.0)
    ap.add_argument("--cb", type = int, default = 2)
    ap.add_argument("--iters", type = int, default = 300)
    ap.add_argument("--seed", type = int, default = 1234)
    ap.add_argument("--overlap", type = float, default = 0.5,
                    help = "fraction of the previous row's experts a verification row reuses (0 = independent)")
    ap.add_argument("--two_launch", action = "store_true", help = "also time the non-fused two-launch path")
    ap.add_argument("--compact_ab", action = "store_true",
                    help = "A/B the compact dual queue (EXL3_MOE_COOP_COMPACT 1 vs 0), ABAB in one process")
    ap.add_argument("--trace", action = "store_true")
    ap.add_argument("--json", type = str, default = None)
    args = ap.parse_args()
    dev = torch.device("cuda", args.device)
    torch.cuda.set_device(dev)
    if args.trace:
        os.environ["EXL3_MOE_TRACE"] = "1"
    t = build(dev, args.seed, args.K, args.cb, args.bsz, args.E, args.topk, args.Hi, args.I, args.Ho, args.Ksh,
              args.overlap)
    res = {}
    if args.compact_ab:
        # the env is read per launch_pair call, so both queues can be timed in one process, ABAB
        for rnd in range(2):
            for v in ("1", "0"):
                os.environ["EXL3_MOE_COOP_COMPACT"] = v
                us, p50, mn = timeit(t, args.Hi, args.bsz, args.K, args.Ksh, args.cb, args.iters, True)
                print(f"BENCH compact={v} rnd {rnd}: {us:8.2f} us/call (median {p50:8.2f}, min {mn:8.2f})", flush = True)
        os.environ.pop("EXL3_MOE_COOP_COMPACT", None)
    modes = []
    if args.two_launch:
        modes = [True, False]
    elif not args.compact_ab:
        modes = [True]
    for fused in modes:
        us, p50, mn = timeit(t, args.Hi, args.bsz, args.K, args.Ksh, args.cb, args.iters, fused)
        name = "fused" if fused else "two-launch"
        res[name] = us
        print(f"BENCH {name:10s} bsz {args.bsz} E {args.E} topk {args.topk} overlap {args.overlap} "
              f"K {args.K}/{args.Ksh}: {us:8.2f} us/call (median {p50:8.2f}, min {mn:8.2f})", flush = True)
    if args.trace:
        for d, nm in ((args.device, "P100" if args.device == 1 else "RTX2060"),):
            rec = ext.exl3_moe_coop_trace_dump(d).numpy()
            import numpy as np
            np.save(os.path.join(OUT, f"coop_trace_{nm}.npy"), rec)
            print(f"TRACE {nm}: {rec.shape[0]} records -> {OUT}/coop_trace_{nm}.npy", flush = True)
    if args.json:
        json.dump(res, open(args.json, "w"), indent = 1)
