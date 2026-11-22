#!/bin/bash
# E12 (sm_75, head_dim <= 128: 64-key decode attention tiles): ctl EXL3_ATTN_DECODE_BN=16 vs default on the RTX 2060,
# depths 512 / 8192 / 32768, then oracle + decode ppl for the candidate
R=doc/lfm8_opt/run.sh
O=/home/nicola/exllamav3/tmp/lfm8_opt
M=/home/nicola/exllamav3/models/LFM3-1.7B-exl3-8.0bpw
P=/home/nicola/exllamav3/.venv/bin/python
S="-depths 512 8192 -cs 16384 -samplers greedy -n 256 -reps 3"
L="-depths 32768 -cs 65536 -samplers greedy -n 256 -reps 3"
for arm in ctl_a cand ctl_b; do
    e=""; [ $arm != cand ] && e="EXL3_ATTN_DECODE_BN=16"
    env $e X=1 $R 0 e12_${arm}_s $S | sed "s/^/$arm: /" | grep RESULT
    env $e X=1 $R 0 e12_${arm}_l $L | sed "s/^/$arm: /" | grep RESULT
done
$R 0 e12_oracle -oracle $O/oracle_2060_e12.json | grep ORACLE
$P doc/lfm8_opt/cmp_oracle.py $O/oracle_2060_e6.json $O/oracle_2060_e12.json
flock -w 900 /home/nicola/exllamav3/tmp/gpu.lane env CUDA_VISIBLE_DEVICES=0 $P doc/lfm8_opt/decode_ppl.py $M $O/dppl_2060_e12.json 2>&1 | grep DECODE_PPL
$P -c "
import json
a = json.load(open('$O/dppl_2060_e1.json'))['top1']; b = json.load(open('$O/dppl_2060_e12.json'))['top1']
print('decode top1 agree vs E1 build', sum(x == y for x, y in zip(a, b)), '/', len(a))"
