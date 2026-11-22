#!/bin/bash
# Prefill A/B: control env vs candidate env, ctl / cand / ctl per card (both cards in parallel), perf script to 16k.
# usage: CTL_ENV="EXL3_SDPA_SWA=0" CAND_ENV="" prefill_ab.sh <tag>
TAG=$1
M=/home/nicola/exllamav3/models/LFM3-1.7B-exl3-8.0bpw
P=/home/nicola/exllamav3/.venv/bin/python
O=/home/nicola/exllamav3/tmp/lfm8_opt
L=/home/nicola/exllamav3/tmp
one() {   # cvd lane arm envs
    local cvd=$1 lane=$2 arm=$3; shift 3
    flock -w 3000 $lane env CUDA_VISIBLE_DEVICES=$cvd "$@" $P doc/lfm8_opt/prefill_perf.py -m $M -cs 16384 -cq 8 \
        -max_length 16384 -sg > $O/pf_${TAG}_${cvd}_${arm}.log 2>&1
}
card() {
    local cvd=$1 lane=$2
    one $cvd $lane ctl_a $CTL_ENV X=1
    one $cvd $lane cand $CAND_ENV X=1
    one $cvd $lane ctl_b $CTL_ENV X=1
}
card 1 $L/gpu.lane.p100 &
card 0 $L/gpu.lane
wait
$P - "$TAG" <<'EOF'
import re, sys, glob
tag = sys.argv[1]
for cvd, name in (("1", "P100"), ("0", "RTX 2060")):
    rows = {}
    for arm in ("ctl_a", "cand", "ctl_b"):
        t = open(f"/home/nicola/exllamav3/tmp/lfm8_opt/pf_{tag}_{cvd}_{arm}.log", errors = "ignore").read()
        for L, v in re.findall(r"Length\s+(\d+):\s+([\d.]+) tokens/s", t):
            rows.setdefault(int(L), {})[arm] = float(v)
    print(f"== {name}  (tokens/s: ctl_a | cand | ctl_b)")
    for L in sorted(rows):
        r = rows[L]
        print(f"  {L:6d}: {r.get('ctl_a', 0):8.0f} | {r.get('cand', 0):8.0f} | {r.get('ctl_b', 0):8.0f}")
EOF
