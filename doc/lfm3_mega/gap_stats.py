"""Per-token kernel time vs wall from an nsys cuda_gpu_trace CSV (doc/lfm3_mega/nsys_decode.sh).

  python doc/lfm3_mega/gap_stats.py <tag>_cuda_gpu_trace.csv -wall_ms 6.93 [-k 243] [-tokens 128]

Takes the last <tokens> * <k> kernels (steady-state decode graph replays), reports the kernel time per token (hardware
timestamps, little inflated by tracing), the traced gaps between consecutive kernels and the unprofiled wall minus the
kernel time, i.e. what one persistent kernel could at most recover from launch boundaries alone.
"""
import argparse, csv, collections, statistics

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("-wall_ms", type = float, required = True)
ap.add_argument("-k", type = int, default = 243, help = "kernels per token")
ap.add_argument("-tokens", type = int, default = 128)
ap.add_argument("-top", type = int, default = 22)
a = ap.parse_args()

rows = []
with open(a.csv) as f:
    for r in csv.DictReader(f):
        name = r.get("Name", "")
        if "memcpy" in name.lower() or "memset" in name.lower():
            continue
        rows.append((int(r["Start (ns)"]), int(r["Duration (ns)"]), name))
rows.sort()
n = a.tokens * a.k
rows = rows[-n:]
span = (rows[-1][0] + rows[-1][1] - rows[0][0]) / 1e6 / a.tokens
ksum = sum(r[1] for r in rows) / 1e6 / a.tokens
gaps = [rows[i + 1][0] - (rows[i][0] + rows[i][1]) for i in range(len(rows) - 1)]
print(f"kernels {len(rows)} ({a.tokens} tokens x {a.k}); traced span {span:.3f} ms/token, kernel sum {ksum:.3f} ms/token")
print(f"traced gaps: median {statistics.median(gaps) / 1e3:.2f} us, mean {statistics.mean(gaps) / 1e3:.2f} us, "
      f"p90 {sorted(gaps)[int(len(gaps) * 0.9)] / 1e3:.2f} us")
print(f"unprofiled wall {a.wall_ms:.3f} ms/token -> wall - kernel sum = {a.wall_ms - ksum:.3f} ms/token "
      f"({(a.wall_ms - ksum) / a.wall_ms * 100:.1f} %), {(a.wall_ms - ksum) / a.k * 1e3:.2f} us per boundary")

def short(s):
    s = s.replace("void ", "")
    return s if len(s) <= 70 else s[:67] + "..."

by = collections.defaultdict(lambda: [0, 0, []])
for ts, d, name in rows:
    b = by[short(name)]
    b[0] += d
    b[1] += 1
    b[2].append(d)
print(f"\n{'ms/tok':>7} {'calls':>6} {'us/call':>8} {'min':>7} {'max':>7}  kernel")
for k, (t, c, ds) in sorted(by.items(), key = lambda x: -x[1][0])[:a.top]:
    print(f"{t / 1e6 / a.tokens:7.3f} {c / a.tokens:6.1f} {t / c / 1e3:8.1f} {min(ds) / 1e3:7.1f} {max(ds) / 1e3:7.1f}  {k}")
