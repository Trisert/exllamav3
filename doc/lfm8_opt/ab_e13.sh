#!/bin/bash
# E13 (multi-matrix GEMV for the gate + up and Q / K / V bundles on Pascal): ctl EXL3_MGEMV=0 (separate GEMVs, the
# previous build's behaviour) vs default on the P100, depths 512 / 8192 / 32768, then oracle + decode ppl
R=doc/lfm8_opt/run.sh
O=/home/nicola/exllamav3/tmp/lfm8_opt
M=/home/nicola/exllamav3/models/LFM3-1.7B-exl3-8.0bpw
P=/home/nicola/exllamav3/.venv/bin/python
S="-depths 512 8192 -cs 16384 -samplers greedy -n 256 -reps 3"
L="-depths 32768 -cs 65536 -samplers greedy -n 256 -reps 3"
for arm in ctl_a cand ctl_b; do
    e=""; [ $arm != cand ] && e="EXL3_MGEMV=0"
    env $e X=1 $R 1 e13_${arm}_s $S | sed "s/^/$arm: /" | grep RESULT
    env $e X=1 $R 1 e13_${arm}_l $L | sed "s/^/$arm: /" | grep RESULT
done
$R 1 e13_oracle -oracle $O/oracle_p100_e13.json | grep ORACLE
$P doc/lfm8_opt/cmp_oracle.py $O/oracle_p100_e8.json $O/oracle_p100_e13.json
flock -w 900 /home/nicola/exllamav3/tmp/gpu.lane.p100 env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD $P doc/lfm8_opt/decode_ppl.py $M $O/dppl_p100_e13.json 2>&1 | grep DECODE_PPL
$P -c "
import json
a = json.load(open('$O/dppl_p100_e23.json')); b = json.load(open('$O/dppl_p100_e13.json'))
print('decode ppl', a.get('ppl'), '->', b.get('ppl'), '| top1 agree vs E23 build', sum(x == y for x, y in zip(a['top1'], b['top1'])), '/', len(a['top1']))"
