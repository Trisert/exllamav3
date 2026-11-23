#!/bin/bash
# Long-context gate on both cards: needle at ~49k tokens (64k 8-bit cache), prefill speed to 64k
M=/home/nicola/exllamav3/models/LFM3-1.7B-exl3-8.0bpw
P=/home/nicola/exllamav3/.venv/bin/python
O=/home/nicola/exllamav3/tmp/lfm8_opt
L=/home/nicola/exllamav3/tmp
card() {   # cvd lane
    flock -w 3000 $2 env CUDA_VISIBLE_DEVICES=$1 NEEDLE_CACHE=65536 NEEDLE_FRACS=0.75 $P doc/lfm8_opt/needle.py $M \
        > $O/needle64k_$1.log 2>&1
    flock -w 3000 $2 env CUDA_VISIBLE_DEVICES=$1 $P doc/lfm8_opt/prefill_perf.py -m $M -cs 65536 -cq 8 -max_length 65536 -sg \
        > $O/prefill64k_$1.log 2>&1
}
card 1 $L/gpu.lane.p100 &
card 0 $L/gpu.lane
wait
$P - <<'EOF'
import re
for cvd, name in (("1", "P100"), ("0", "RTX 2060")):
    t = open(f"/home/nicola/exllamav3/tmp/lfm8_opt/needle64k_{cvd}.log", errors = "ignore").read()
    print("==", name)
    for l in t.splitlines():
        if l.startswith("ctx") or l.startswith("NEEDLE") or "Error" in l: print("  " + l)
    t = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", open(f"/home/nicola/exllamav3/tmp/lfm8_opt/prefill64k_{cvd}.log", errors = "ignore").read())
    print("  prefill: " + ", ".join(f"{L}: {float(v):.0f}" for L, v in re.findall(r"Length\s+(\d+):\s+([\d.]+) tokens/s", t)))
    if "Error" in t: print("  prefill error:", [l for l in t.splitlines() if "Error" in l][:3])
EOF
