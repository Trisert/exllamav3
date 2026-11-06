import re, collections, sys

# usage: loopmix.py file.sass [name-filter]: op mix of the whole function and of its largest backward-branch loop
txt = open(sys.argv[1]).read()
flt = sys.argv[2] if len(sys.argv) > 2 else ""
pat = re.compile(r'/\*([0-9a-f]{4})\*/\s+(@!?P\d\s+)?([A-Z0-9_]+)(\.[A-Z0-9_.]+)?([^;]*);')
for fn in re.split(r'\n\s*Function : ', "\n" + txt)[1:]:
    name = fn.split('\n')[0]
    if flt not in name: continue
    ins = [(int(m.group(1), 16), m.group(3), m.group(5)) for m in pat.finditer(fn)]
    best = None
    for addr, op, rest in ins:
        if op == "BRA":
            t = re.search(r'0x([0-9a-f]+)', rest)
            if t and int(t.group(1), 16) < addr:
                span = (int(t.group(1), 16), addr)
                if best is None or span[1] - span[0] > best[1] - best[0]: best = span
    def mix(sel):
        c = collections.Counter(op for a, op, _ in sel)
        tot = sum(c.values())
        return tot, c.get("HFMA2", 0) / max(tot, 1), ' '.join(f'{k}:{v}' for k, v in c.most_common(16))
    t, h, s = mix(ins)
    print(f"{name[:80]}\n  all : {t} instr, HFMA2 {100 * h:.1f}%  {s}")
    if best:
        t, h, s = mix([x for x in ins if best[0] <= x[0] <= best[1]])
        print(f"  loop: {t} instr, HFMA2 {100 * h:.1f}%  {s}")
