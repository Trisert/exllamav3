"""Idle gaps between consecutive GPU kernels in a bench_decode -profile trace, grouped by (previous kernel -> next kernel).
  .venv/bin/python util/trace_gaps.py trace.json -steps 64 [-min_us 5]
Profiler overhead inflates gaps (CPU side), so treat as an upper bound."""
import sys, json, argparse, collections
ap = argparse.ArgumentParser()
ap.add_argument("trace"); ap.add_argument("-steps", type = int, default = 64); ap.add_argument("-min_us", type = float, default = 5.0)
a = ap.parse_args()
ev = json.load(open(a.trace))["traceEvents"]
k = sorted((e["ts"], e["dur"], e["name"]) for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"))
sh = lambda n: n.split("(")[0][:60]
g = collections.defaultdict(lambda: [0.0, 0])
tot = 0.0
hist = collections.Counter()
for (t0, d0, n0), (t1, d1, n1) in zip(k, k[1:]):
    gap = t1 - (t0 + d0)
    if gap > 0:
        tot += gap
        hist["<5us" if gap < 5 else "5-20us" if gap < 20 else "20-100us" if gap < 100 else ">=100us"] += gap
    if gap >= a.min_us:
        x = g[(sh(n0), sh(n1))]; x[0] += gap; x[1] += 1
print(f"total gap {tot / 1000 / a.steps:.3f} ms/token; by size class (ms/token): " + ", ".join(f"{c} {v / 1000 / a.steps:.3f}" for c, v in sorted(hist.items())))
for (p, n), (t, c) in sorted(g.items(), key = lambda kv: -kv[1][0])[:12]:
    print(f"  {t / 1000 / a.steps:6.3f} ms/tok {c / a.steps:5.1f}x  [{p}] -> [{n}]")
