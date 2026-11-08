"""Mechanism analysis of a torch/kineto chrome trace for the Flash-Next prefill A/B.

Reports per device: kernel busy, memcpy busy, activity span, idle gaps; and correlates the
cudaMemcpyAsync / cudaStreamSynchronize runtime calls with their GPU activities, so the
CPU-blocked time per copy kind (H2D / D2H / D2D) and per byte size is visible.

usage: python tmp/trace_analyze2.py <trace.json>
"""
import json, sys, collections

DEV = {0: "cuda:0 2060", 1: "cuda:1 P100"}

def devname(d):
    return DEV.get(d, f"dev{d}")

def main(path):
    with open(path) as f:
        tr = json.load(f)
    ev = tr["traceEvents"]
    print(f"TRACE {path}: {len(ev)} events")
    # memcpy activities by event name (args.bytes, args.device)
    mem = collections.defaultdict(lambda: [0, 0.0, 0])          # (dev, name) -> n, us, bytes
    mem_by_corr = {}
    kern = collections.defaultdict(lambda: [0, 0.0])
    span = {}
    gaps = collections.defaultdict(list)
    t0, t1 = 1e18, 0.0
    for e in ev:
        ph, cat = e.get("ph"), e.get("cat", "")
        if ph != "X" or cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        dev = e.get("args", {}).get("device")
        dur = e.get("dur", 0.0)
        ts = e.get("ts", 0.0)
        if cat == "kernel":
            kern[(dev, e.get("name", "?"))][0] += 1
            kern[(dev, e.get("name", "?"))][1] += dur
        else:
            b = e.get("args", {}).get("bytes", 0) or 0
            mem[(dev, e.get("name", "?"))][0] += 1
            mem[(dev, e.get("name", "?"))][1] += dur
            mem[(dev, e.get("name", "?"))][2] += b
            corr = e.get("args", {}).get("correlation")
            if corr is not None:
                mem_by_corr[corr] = (cat, e.get("name", "?"), dev, dur, b)
        s = span.setdefault(dev, [1e18, 0.0, []])
        s[0] = min(s[0], ts); s[1] = max(s[1], ts + dur)
        s[2].append((ts, ts + dur))
        t0 = min(t0, ts); t1 = max(t1, ts + dur)
    print(f"  window: {(t1 - t0) / 1e6:.3f} s")
    for dev, (a, b, iv) in sorted(span.items(), key = lambda kv: kv[1][0]):
        iv.sort()
        busy = sum(y - x for x, y in iv)
        idle = 0.0; gap_list = []
        cur = iv[0][1]
        for x, y in iv[1:]:
            if x - cur > 1000:      # gap > 1 ms
                idle += x - cur
                gap_list.append((cur - t0, x - cur))
            cur = max(cur, y)
        kbusy = sum(t for (d, n), (c, t) in kern.items() if d == dev)
        mbusy = sum(t for (d, n), (c, t, b) in mem.items() if d == dev)
        print(f"  {devname(dev):12s}: span {(b - a) / 1e6:6.3f} s  kernels {kbusy / 1e6:6.3f} s  "
              f"copies {mbusy / 1e6:6.3f} s  gaps>1ms {idle / 1e6:6.3f} s in {len(gap_list)} gaps; "
              f"largest: " + ", ".join(f"{d / 1e6:.2f}s@{t / 1e6:.1f}s" for t, d in
                                       sorted(gap_list, key = lambda g: -g[1])[:4]))
    print("\n  MEMCPY ACTIVITIES")
    for (dev, name), (n, us, b) in sorted(mem.items(), key = lambda kv: -kv[1][1]):
        gbs = (b / 1e9) / (us * 1e-6) if us else 0
        print(f"    {us / 1e6:7.3f} s {n:6d} calls {1000 * us / n:8.1f} us/call  {b / 1e9:7.2f} GB  "
              f"{gbs:5.2f} GB/s  {devname(dev)}  {name}")
    # runtime API events with CPU durations
    rt = collections.defaultdict(lambda: [0, 0.0, 0.0, 0])     # name -> n, us, blocked_us, big
    blocked = collections.defaultdict(lambda: [0, 0.0, 0, 0.0, 0])  # name -> n, us, cnt>0.5ms, us, bytes
    for e in ev:
        if e.get("ph") != "X" or e.get("cat") != "cuda_runtime":
            continue
        name = e.get("name", "?")
        dur = e.get("dur", 0.0)
        rt[name][0] += 1
        rt[name][1] += dur
        if dur > 500:
            rt[name][2] += dur
            rt[name][3] += 1
        corr = e.get("args", {}).get("correlation", e.get("args", {}).get("External id"))
        kind = mem_by_corr.get(corr)
        if kind is not None and dur > 500:
            key = kind[1]
            blocked[key][0] += 1
            blocked[key][1] += dur
            blocked[key][4] += kind[4] or 0
    print("\n  RUNTIME API (CPU time; >0.5 ms = host-blocked)")
    for name, (n, us, bus, nbig) in sorted(rt.items(), key = lambda kv: -kv[1][2]):
        print(f"    {us / 1e6:7.3f} s CPU {n:6d} calls, {bus / 1e6:7.3f} s in {nbig:5d} calls > 0.5 ms"
              f"  {name}")
    print("\n  BLOCKED >0.5 ms CORRELATED TO MEMCPY KIND")
    for name, (n, us, _, _, b) in sorted(blocked.items(), key = lambda kv: -kv[1][1]):
        print(f"    {us / 1e6:7.3f} s CPU in {n:5d} blocked calls, {b / 1e9:6.2f} GB  {name}")
    print("\n  TOP KERNELS per device")
    for dev in sorted(span):
        top = [(n, c, t) for (d, n), (c, t) in kern.items() if d == dev]
        top.sort(key = lambda x: -x[2])
        tot = sum(t for _, _, t in top)
        print(f"    {devname(dev)}: kernel time {tot / 1e6:.3f} s")
        for n, c, t in top[:12]:
            print(f"      {t / 1e6:7.3f} s {c:5d} calls {1000 * t / c:8.3f} ms/call  {n[:95]}")

if __name__ == "__main__":
    main(sys.argv[1])
