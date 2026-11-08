"""Culprit analysis: which host ops block in cudaMemcpyAsync / cudaStreamSynchronize.

usage: python tmp/trace_blocked.py <trace.json>
"""
import json, sys, collections

def main(path):
    with open(path) as f:
        tr = json.load(f)
    ev = tr["traceEvents"]
    # cpu_op external id -> name
    opname = {}
    for e in ev:
        if e.get("cat") == "cpu_op" and e.get("ph") == "X":
            ext = e.get("args", {}).get("External id")
            if ext is not None:
                opname[ext] = e.get("name", "?")
    # runtime calls: external id, correlation, dur, name
    rt = []
    for e in ev:
        if e.get("cat") == "cuda_runtime" and e.get("ph") == "X":
            a = e.get("args", {})
            rt.append((e.get("name", "?"), e.get("dur", 0.0), a.get("External id"), a.get("correlation")))
    # memcpy activities by correlation
    memby = {}
    for e in ev:
        if e.get("cat") == "gpu_memcpy" and e.get("ph") == "X":
            a = e.get("args", {})
            corr = a.get("correlation")
            if corr is not None:
                memby[corr] = (e.get("name", "?"), a.get("bytes", 0) or 0, e.get("dur", 0.0), a.get("device"))
    # group blocked calls by (runtime name, cpu op name, memcpy kind)
    grp = collections.defaultdict(lambda: [0, 0.0, 0.0, 0])
    for (rname, dur, ext, corr) in rt:
        if dur < 500:
            continue
        op = opname.get(ext, "?")
        mk = memby.get(corr)
        kind = mk[0] if mk else "no-memcpy"
        g = grp[(rname, op, kind)]
        g[0] += 1; g[1] += dur
        if mk: g[2] += mk[1] or 0; g[3] += mk[2] or 0
    print(f"TRACE {path}")
    print(f"  {'CPU s':>8} {'calls':>6} {'GB':>7} {'gpu_s':>7}  runtime / cpu-op / memcpy-kind")
    for (rname, op, kind), (n, us, b, gus) in sorted(grp.items(), key = lambda kv: -kv[1][1])[:25]:
        print(f"  {us / 1e6:8.3f} {n:6d} {b / 1e9:7.2f} {gus / 1e6:7.3f}  {rname} / {op[:60]} / {kind}")
    # biggest single blocked calls
    big = sorted(((dur, ext, corr) for (rname, dur, ext, corr) in rt if dur > 500), reverse = True)[:12]
    print("  largest blocked calls:")
    for dur, ext, corr in big:
        mk = memby.get(corr)
        print(f"    {dur / 1e3:9.2f} ms  op={opname.get(ext, '?')[:55]}  memcpy={mk[0] if mk else '?'} "
              f"bytes={(mk[1] if mk else 0) / 1e3:.1f} kB")
    # kernel instantiations comparison
    kall = collections.defaultdict(lambda: [0, 0.0])
    for e in ev:
        if e.get("cat") == "kernel" and e.get("ph") == "X":
            kall[(e.get("args", {}).get("device"), e.get("name", "?")[:60])][0] += 1
            kall[(e.get("args", {}).get("device"), e.get("name", "?")[:60])][1] += e.get("dur", 0.0)
    print("  exl3_moe / hfma2 / gemm kernels:")
    for (dev, name), (c, t) in sorted(kall.items(), key = lambda kv: (kv[0][0] or 0, -kv[1][1])):
        if any(k in name for k in ("exl3_moe", "hfma2", "gemm", "sgemm")):
            print(f"    dev{dev} {t / 1e6:7.3f} s {c:5d} calls {1000 * t / c:8.2f} us/call  {name}")

if __name__ == "__main__":
    main(sys.argv[1])
