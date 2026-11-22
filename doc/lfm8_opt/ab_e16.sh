#!/bin/bash
# E16 (HFMA2 inner loops of the native decode split kernel, P100): ctl EXL3_ATTN_NATIVE_FP32=1 vs default,
# cand / ctl / cand at depths 512 / 8192 / 32768; then (FULL=1) oracle + needle for the candidate
R=doc/lfm8_opt/run.sh
O=/home/nicola/exllamav3/tmp/lfm8_opt
P=/home/nicola/exllamav3/.venv/bin/python
S="-depths 512 8192 -cs 16384 -samplers greedy -n 256 -reps 3"
L="-depths 32768 -cs 65536 -samplers greedy -n 256 -reps 3"
for arm in cand_a ctl cand_b; do
    e="EXL3_ATTN_NATIVE_FP32=0"; [ $arm = ctl ] && e="EXL3_ATTN_NATIVE_FP32=1"
    env $e X=1 $R 1 e16_${arm}_s $S | sed "s/^/$arm: /" | grep RESULT
    env $e X=1 $R 1 e16_${arm}_l $L | sed "s/^/$arm: /" | grep RESULT
done
[ "$FULL" = 1 ] || exit 0
$R 1 e16_oracle -oracle $O/oracle_p100_e16.json | grep ORACLE
$P doc/lfm8_opt/cmp_oracle.py $O/oracle_p100_e15.json $O/oracle_p100_e16.json
doc/lfm8_opt/needle_p100.sh e16
