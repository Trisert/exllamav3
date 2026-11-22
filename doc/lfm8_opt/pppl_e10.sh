#!/bin/bash
# Prefill-mode ppl, E10 arms (control EXL3_SDPA_SWA=0) on both cards, then top-1 agreement
M=/home/nicola/exllamav3/models/LFM3-1.7B-exl3-8.0bpw
P=/home/nicola/exllamav3/.venv/bin/python
O=/home/nicola/exllamav3/tmp/lfm8_opt
L=/home/nicola/exllamav3/tmp
(flock -w 900 $L/gpu.lane.p100 env CUDA_VISIBLE_DEVICES=1 EXL3_SDPA_SWA=0 $P doc/lfm8_opt/prefill_ppl.py $M $O/pppl_p100_ctl.json 2>&1 | grep PPL
 flock -w 900 $L/gpu.lane.p100 env CUDA_VISIBLE_DEVICES=1 $P doc/lfm8_opt/prefill_ppl.py $M $O/pppl_p100_e10.json 2>&1 | grep PPL) &
flock -w 900 $L/gpu.lane env CUDA_VISIBLE_DEVICES=0 EXL3_SDPA_SWA=0 $P doc/lfm8_opt/prefill_ppl.py $M $O/pppl_2060_ctl.json 2>&1 | grep PPL
flock -w 900 $L/gpu.lane env CUDA_VISIBLE_DEVICES=0 $P doc/lfm8_opt/prefill_ppl.py $M $O/pppl_2060_e10.json 2>&1 | grep PPL
wait
$P - <<EOF
import json
d = {k: json.load(open("$O/pppl_" + k + ".json"))["top1"] for k in ["p100_ctl", "p100_e10", "2060_ctl", "2060_e10"]}
for a, b in [("p100_ctl", "p100_e10"), ("2060_ctl", "2060_e10"), ("p100_ctl", "2060_ctl")]:
    print(a, b, "top1 agree", sum(x == y for x, y in zip(d[a], d[b])), "/", len(d[a]))
EOF
