#!/bin/bash
# E11 quality on the P100: prefill ppl ctl (EXL3_FA60_HD128=0) vs fa60 at head_dim 128, top-1 agreement, needle
M=/home/nicola/exllamav3/models/LFM3-1.7B-exl3-8.0bpw
P=/home/nicola/exllamav3/.venv/bin/python
O=/home/nicola/exllamav3/tmp/lfm8_opt
L=/home/nicola/exllamav3/tmp/gpu.lane.p100
flock -w 900 $L env CUDA_VISIBLE_DEVICES=1 EXL3_FA60_HD128=0 $P doc/lfm8_opt/prefill_ppl.py $M $O/pppl_p100_e11ctl.json 4096 2>&1 | grep PPL
flock -w 900 $L env CUDA_VISIBLE_DEVICES=1 $P doc/lfm8_opt/prefill_ppl.py $M $O/pppl_p100_e11.json 4096 2>&1 | grep PPL
$P -c "
import json
a = json.load(open('$O/pppl_p100_e11ctl.json'))['top1']; b = json.load(open('$O/pppl_p100_e11.json'))['top1']
print('top1 agree', sum(x == y for x, y in zip(a, b)), '/', len(a))"
flock -w 900 $L env CUDA_VISIBLE_DEVICES=1 $P doc/lfm8_opt/needle.py $M 2>&1 | grep -E "NEEDLE|FAIL"
