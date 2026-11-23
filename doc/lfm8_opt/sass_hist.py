"""Opcode histogram of a SASS address range. usage: sass_hist.py <cuobjdump -sass file> <start hex> <end hex>"""
import re, sys, collections

path, lo, hi = sys.argv[1], int(sys.argv[2], 16), int(sys.argv[3], 16)
pat = re.compile(r"/\*([0-9a-f]{4,})\*/\s+[{}]?\s*(@!?P[0-9T]\s+)?([A-Z][A-Z0-9_]*)(\.[A-Z0-9_.]+)?")
c = collections.Counter()
for line in open(path):
    m = pat.search(line)
    if not m:
        continue
    addr = int(m.group(1), 16)
    if lo <= addr < hi:
        c[m.group(3) + (m.group(4) or "")] += 1
total = sum(c.values())
print(f"total {total} instructions in [{lo:#x}, {hi:#x})")
for k, v in c.most_common(40):
    print(f"{v:5d}  {k}")
