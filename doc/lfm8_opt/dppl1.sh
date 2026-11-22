#!/bin/bash
# Decode-mode ppl, round 1 arms (control = EXL3_GEMV_PASCAL_K8=0 EXL3_UNFUSE_K8=0)
M=/home/nicola/exllamav3/models/LFM3-1.7B-exl3-8.0bpw
P=/home/nicola/exllamav3/.venv/bin/python
O=/home/nicola/exllamav3/tmp/lfm8_opt
L=/home/nicola/exllamav3/tmp
flock -w 600 $L/gpu.lane.p100 env CUDA_VISIBLE_DEVICES=1 $P doc/lfm8_opt/decode_ppl.py $M $O/dppl_p100_e23.json 2>&1 | grep DECODE_PPL &
flock -w 600 $L/gpu.lane env EXL3_UNFUSE_K8=0 CUDA_VISIBLE_DEVICES=0 $P doc/lfm8_opt/decode_ppl.py $M $O/dppl_2060_ctl.json 2>&1 | grep DECODE_PPL
flock -w 600 $L/gpu.lane env CUDA_VISIBLE_DEVICES=0 $P doc/lfm8_opt/decode_ppl.py $M $O/dppl_2060_e1.json 2>&1 | grep DECODE_PPL
wait
$P - <<EOF
import json
d = {k: json.load(open("$O/dppl_" + k + ".json"))["top1"] for k in ["p100_ctl", "p100_e23", "2060_ctl", "2060_e1"]}
for a, b in [("p100_ctl", "p100_e23"), ("2060_ctl", "2060_e1"), ("p100_ctl", "2060_ctl")]:
    print(a, b, "top1 agree", sum(x == y for x, y in zip(d[a], d[b])), "/", len(d[a]))
EOF
