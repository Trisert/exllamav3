"""Opcode histogram of the hottest loop (backward branch body with the most VABSDIFF4) of one kernel in a
cuobjdump -sass dump. usage: hot_loop.py <sass file> <substring of the demangled-or-mangled function line>"""
import re, sys, collections, subprocess

path, key = sys.argv[1], sys.argv[2]
text = open(path).read().split("\n")
# demangle function header lines
funcs, cur = {}, None
for line in text:
    if "Function :" in line:
        name = line.split("Function :")[1].strip()
        dem = subprocess.run(["c++filt", name], capture_output = True, text = True).stdout.strip()
        cur = dem
        funcs[cur] = []
    elif cur is not None:
        funcs[cur].append(line)
sel = [f for f in funcs if key in f]
assert len(sel) == 1, sel
body = funcs[sel[0]]
pat = re.compile(r"/\*([0-9a-f]{4,})\*/\s+[{}]?\s*(@!?P[0-9T]\s+)?([A-Z][A-Z0-9_]*)(\.[A-Z0-9_.]+)?([^;]*);")
ins = []
for line in body:
    m = pat.search(line)
    if m:
        ins.append((int(m.group(1), 16), m.group(3), (m.group(4) or ""), m.group(5)))
loops = []
for addr, op, mod, args in ins:
    if op == "BRA":
        t = re.search(r"0x([0-9a-f]+)", args)
        if t and int(t.group(1), 16) < addr:
            lo, hi = int(t.group(1), 16), addr + 8
            n = sum(1 for a, o, *_ in ins if lo <= a < hi and o == "VABSDIFF4")
            loops.append((n, lo, hi))
loops.sort(reverse = True)
print(sel[0][:120])
for n, lo, hi in loops[:4]:
    print(f"loop [{lo:#x}, {hi:#x}): {n} VABSDIFF4")
n, lo, hi = loops[0]
c = collections.Counter(o + m for a, o, m, _ in ins if lo <= a < hi)
tot = sum(c.values())
print(f"hot loop: {tot} instructions, {n} VABSDIFF4 (= weights per lane per iteration if one per weight)")
half = {"VABSDIFF4", "SHF", "PRMT", "BFE", "I2F", "F2I", "POPC", "FLO", "BREV"}
slots = sum(v * (2 if o.split(".")[0] in half else 1) for o, v in c.items())
print(f"issue slots (half-rate ops x2): {slots}, per weight {slots / max(n, 1):.2f}")
for k, v in c.most_common(30):
    print(f"{v:5d}  {k}")
