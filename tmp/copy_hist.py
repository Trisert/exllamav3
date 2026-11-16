"""Byte-size histogram of gpu_memcpy activities in a chrome trace, per direction/device.
usage: python tmp/copy_hist.py <trace.json>
"""
import json, sys, collections
def main(path):
    with open(path) as f:
        tr = json.load(f)
    h = collections.defaultdict(lambda: [0, 0, 0.0])   # (dev, name, size bucket) -> n, bytes, us
    for e in tr["traceEvents"]:
        if e.get("cat") != "gpu_memcpy" or e.get("ph") != "X":
            continue
        a = e.get("args", {})
        b = a.get("bytes", 0) or 0
        bucket = int(round(b / 65536.0))    # 64 KiB buckets
        k = (a.get("device"), e.get("name", "?"), bucket)
        h[k][0] += 1; h[k][1] += b; h[k][2] += e.get("dur", 0.0)
    print(f"TRACE {path}")
    print(f"  {'calls':>6} {'total GB':>9} {'size/call':>11} {'gpu s':>8}  device / kind")
    agg = collections.defaultdict(lambda: [0, 0, 0.0])
    for (dev, name, bucket), (n, b, us) in sorted(h.items(), key = lambda kv: (kv[0][0] or 0, -kv[1][1])):
        print(f"  {n:6d} {b / 1e9:9.3f} {(b / n) / 1024:9.1f} kB {us / 1e6:8.3f}  dev{dev} {name} "
              f"(~{bucket * 64} kB)")
        k = (dev, name)
        agg[k][0] += n; agg[k][1] += b; agg[k][2] += us
    print("  AGGREGATE:")
    for (dev, name), (n, b, us) in sorted(agg.items(), key = lambda kv: -kv[1][1]):
        print(f"  {n:6d} {b / 1e9:9.3f} {(b / n) / 1024:9.1f} kB {us / 1e6:8.3f}  dev{dev} {name}")
main(sys.argv[1])
