"""Analyze a chrome trace written by util/bench_decode.py -profile (a decode window of N steps).

  .venv/bin/python util/analyze_decode_trace.py trace.json -steps 64 -wall_ms 8.0 [-weight_mb 933] [-rate_gbs 150] [-top 15]

Prints: kernel launches per token, busy time per token (union of kernel intervals), idle gap vs the unprofiled
wall ms per token (-wall_ms), a bucket split A/B/C and the top kernels. Wall time is never taken from the trace.
A = weight_bytes / rate (floor the EXL3 GEMV/GEMM kernels could reach at the microbenchmarked large-GEMV rate),
B = EXL3 GEMV/GEMM time above A, C = every other kernel (+ memcpy/memset).
"""
import sys, json, argparse, re, collections

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("-steps", type = int, default = 64)
ap.add_argument("-wall_ms", type = float, default = 0.0, help = "unprofiled wall ms per token")
ap.add_argument("-weight_mb", type = float, default = 0.0, help = "weight bytes read per token, MB")
ap.add_argument("-rate_gbs", type = float, default = 0.0, help = "large-shape GEMV rate, GB/s")
ap.add_argument("-top", type = int, default = 15)
ap.add_argument("-json", type = str, default = None)
a = ap.parse_args()

ev = json.load(open(a.trace))["traceEvents"]
ops = {}
kern = []
for e in ev:
    if e.get("ph") != "X":
        continue
    c = e.get("cat")
    ar = e.get("args", {})
    if c == "cpu_op":
        ops[ar.get("External id")] = e["name"]
    elif c in ("kernel", "gpu_memcpy", "gpu_memset"):
        kern.append((e["ts"], e["dur"], e["name"], ar.get("External id"), c, ar.get("stream")))

kern.sort()
N = a.steps
streams = collections.Counter(k[5] for k in kern)

# busy = union of kernel intervals
busy = 0.0
cur_s, cur_e = None, None
for ts, d, *_ in kern:
    if cur_e is None or ts > cur_e:
        if cur_e is not None:
            busy += cur_e - cur_s
        cur_s, cur_e = ts, ts + d
    else:
        cur_e = max(cur_e, ts + d)
if cur_e is not None:
    busy += cur_e - cur_s
span = (max(k[0] + k[1] for k in kern) - kern[0][0]) / 1000 if kern else 0
sumdur = sum(k[1] for k in kern)

EXL3_RE = re.compile(r"exl3|gemv|gemm_kernel|trellis|hgemm", re.I)

def short(n):
    return n if len(n) <= 90 else n[:87] + "..."

by = collections.defaultdict(lambda: [0.0, 0])
byop = collections.defaultdict(lambda: [0.0, 0])
exl3 = 0.0
for ts, d, name, ext, c, st in kern:
    by[name][0] += d; by[name][1] += 1
    byop[ops.get(ext, "(no cpu_op: direct ext/graph launch)")][0] += d
    byop[ops.get(ext, "(no cpu_op: direct ext/graph launch)")][1] += 1
    if EXL3_RE.search(name):
        exl3 += d

per = lambda us: us / 1000 / N
print(f"kernels+memcpy/set events: {len(kern)}  = {len(kern) / N:.1f} per token; streams {dict(streams)}")
print(f"sum of kernel durations {per(sumdur):.3f} ms/token; busy (interval union) {per(busy):.3f} ms/token; window span {span / N:.3f} ms/token (profiler-inflated)")
if a.wall_ms:
    print(f"unprofiled wall {a.wall_ms:.3f} ms/token -> idle gap = wall - busy = {a.wall_ms - per(busy):.3f} ms/token ({100 * (a.wall_ms - per(busy)) / a.wall_ms:.0f}%)")
res = dict(launches_per_tok = len(kern) / N, busy_ms = per(busy), sum_ms = per(sumdur), exl3_ms = per(exl3))
if a.weight_mb and a.rate_gbs:
    A = a.weight_mb / 1000 / a.rate_gbs * 1000
    B = per(exl3) - A
    C = per(sumdur) - per(exl3)
    print(f"A (weights {a.weight_mb:.0f} MB / {a.rate_gbs:.0f} GB/s) = {A:.3f} ms | EXL3 kernels total {per(exl3):.3f} -> B = {B:.3f} | C (other kernels) = {C:.3f}"
          + (f" | gaps = {a.wall_ms - per(busy):.3f}" if a.wall_ms else ""))
    res.update(A = A, B = B, C = C)
print(f"\nTop {a.top} kernels by total time (ms per token, calls per token):")
for name, (t, n) in sorted(by.items(), key = lambda kv: -kv[1][0])[:a.top]:
    print(f"  {per(t):7.3f} ms  {n / N:6.1f}x  {short(name)}")
print("\nBy innermost CPU op (ms per token, kernels per token):")
for name, (t, n) in sorted(byop.items(), key = lambda kv: -kv[1][0])[:22]:
    print(f"  {per(t):7.3f} ms  {n / N:6.1f}x  {short(name)}")
if a.json:
    res["top"] = [(short(n), per(t), c / N) for n, (t, c) in sorted(by.items(), key = lambda kv: -kv[1][0])[:40]]
    res["byop"] = [(n, per(t), c / N) for n, (t, c) in sorted(byop.items(), key = lambda kv: -kv[1][0])[:40]]
    json.dump(res, open(a.json, "w"), indent = 1)
