import re, statistics, sys
# 27B per 4 layers: MLP x4, GDN (qkv, z, out) x3, attention (q, k+v, o) x1; o_out = GDN out (3) + attn o (1)
W = {"gate_up": 2 * 5120 * 17408 * 4, "down": 17408 * 5120 * 4, "gdn_qkv": 5120 * 10240 * 3, "gdn_z": 5120 * 6144 * 3,
     "o_out": 6144 * 5120 * 4, "attn_q": 5120 * 12288, "attn_kv": 2 * 5120 * 1024}
base = sys.argv[1]
for f in sys.argv[2:]:
    d = {}; temps = []; clks = []
    for l in open(f):
        m = re.search(r"M=\s*(\d+) (\S+) .* (\S+:\d+)\s+rep \d: eff\s+([\d.]+)% .* raw\s+([\d.]+) TFLOP/s\s+(\d+) C (\d+) MHz", l)
        if m:
            d.setdefault((m.group(2), m.group(3)), []).append(float(m.group(4)))
            temps.append(int(m.group(6))); clks.append(int(m.group(7)))
    vs = sorted({k[1] for k in d})
    for new in vs:
        if new == base: continue
        to = sum(W[s] / statistics.median(d[(s, base)]) for s in W if (s, base) in d)
        tn = sum(W[s] / statistics.median(d[(s, new)]) for s in W if (s, new) in d)
        print(f, new, f"time-weighted speedup vs {base}: {to / tn:.3f}")
    print(f"  temp {min(temps)}-{max(temps)} C, clock {min(clks)}-{max(clks)} MHz (median {statistics.median(clks)})")
