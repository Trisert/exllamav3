#!/bin/bash
# E15 (CUDA-native decode split kernel, P100): ctl EXL3_ATTN_NATIVE=0 (Triton split) vs default, cand / ctl / cand at
# depths 512 / 8192 / 32768; then (FULL=1) oracle + decode ppl for the candidate
R=doc/lfm8_opt/run.sh
O=/home/nicola/exllamav3/tmp/lfm8_opt
M=/home/nicola/exllamav3/models/LFM3-1.7B-exl3-8.0bpw
P=/home/nicola/exllamav3/.venv/bin/python
S="-depths 512 8192 -cs 16384 -samplers greedy -n 256 -reps 3"
L="-depths 32768 -cs 65536 -samplers greedy -n 256 -reps 3"
for arm in cand_a ctl cand_b; do
    e="EXL3_ATTN_NATIVE=1"; [ $arm = ctl ] && e="EXL3_ATTN_NATIVE=0"
    env $e X=1 $R 1 e15_${arm}_s $S | sed "s/^/$arm: /" | grep RESULT
    env $e X=1 $R 1 e15_${arm}_l $L | sed "s/^/$arm: /" | grep RESULT
done
[ "$FULL" = 1 ] || exit 0
$R 1 e15_oracle -oracle $O/oracle_p100_e15.json | grep ORACLE
$P doc/lfm8_opt/cmp_oracle.py $O/oracle_p100_e13.json $O/oracle_p100_e15.json
flock -w 900 /home/nicola/exllamav3/tmp/gpu.lane.p100 env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD $P doc/lfm8_opt/decode_ppl.py $M $O/dppl_p100_e15.json 2>&1 | grep DECODE_PPL
$P -c "
import json
a = json.load(open('$O/dppl_p100_e13.json')); b = json.load(open('$O/dppl_p100_e15.json'))
print('decode ppl', a.get('ppl'), '->', b.get('ppl'), '| top1 agree vs E13', sum(x == y for x, y in zip(a['top1'], b['top1'])), '/', len(a['top1']))"
