#!/bin/bash
# E13 side effect on the RTX 2060: LFM3's headwise attention gate no longer excludes the sliced Q / K / V bundle (tile
# MGEMM on sm_75). ctl EXL3_QKV_HEADWISE=0 vs default, depths 512 / 8192 / 32768
R=doc/lfm8_opt/run.sh
S="-depths 512 8192 -cs 16384 -samplers greedy -n 256 -reps 3"
L="-depths 32768 -cs 65536 -samplers greedy -n 256 -reps 3"
for arm in ctl_a cand ctl_b; do
    e=""; [ $arm != cand ] && e="EXL3_QKV_HEADWISE=0"
    env $e X=1 $R 0 e13t_${arm}_s $S | sed "s/^/$arm: /" | grep RESULT
    env $e X=1 $R 0 e13t_${arm}_l $L | sed "s/^/$arm: /" | grep RESULT
done
