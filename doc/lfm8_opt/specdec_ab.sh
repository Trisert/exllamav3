#!/bin/bash
# m = 1..4 forward cost (perf script -sd: spec-decode seqlens 1..4), EXL3_UNFUSE_K8=0 vs default, ctl / cand / ctl
M=/home/nicola/exllamav3/models/LFM3-1.7B-exl3-8.0bpw
P=/home/nicola/exllamav3/.venv/bin/python
O=/home/nicola/exllamav3/tmp/lfm8_opt
L=/home/nicola/exllamav3/tmp
run() {   # cvd lane tag env...
    local cvd=$1 lane=$2 tag=$3; shift 3
    flock -w 3000 $lane env CUDA_VISIBLE_DEVICES=$cvd "$@" $P doc/lfm8_opt/prefill_perf.py -m $M -cs 16384 -cq 8 \
        -max_length 2048 -spf -sd > $O/sd_${cvd}_${tag}.log 2>&1
}
card() {
    run $1 $2 ctl_a EXL3_UNFUSE_K8=0
    run $1 $2 cand X=1
    run $1 $2 ctl_b EXL3_UNFUSE_K8=0
}
card 1 $L/gpu.lane.p100 &
card 0 $L/gpu.lane
wait
$P - <<'EOF'
import re
for cvd, name in (("1", "P100"), ("0", "RTX 2060")):
    print("==", name)
    for arm in ("ctl_a", "cand", "ctl_b"):
        t = open(f"/home/nicola/exllamav3/tmp/lfm8_opt/sd_{cvd}_{arm}.log", errors = "ignore").read()
        t = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", t)
        lines = [l.strip() for l in re.split(r"[\r\n]", t) if l.strip() and "━" not in l and not l.startswith("--")]
        print(f"  {arm}: " + " | ".join(lines[-6:]))
EOF
