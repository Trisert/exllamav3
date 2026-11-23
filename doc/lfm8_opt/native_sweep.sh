#!/bin/bash
# Native decode split kernel on the P100: Triton baseline (EXL3_ATTN_NATIVE=0) vs native at split targets 2 / 4 / 8 x SMs
R=doc/lfm8_opt/run.sh
for arm in "EXL3_ATTN_NATIVE=0" "EXL3_ATTN_NATIVE_SPLITS=2" "EXL3_ATTN_NATIVE_SPLITS=4" "EXL3_ATTN_NATIVE_SPLITS=8"; do
    tag=ns_$(echo $arm | tr '=' '_')
    env $arm X=1 $R 1 ${tag}_l -depths 32768 -cs 65536 -samplers greedy -n 256 -reps 3 | sed "s/^/$arm: /" | grep RESULT
    env $arm X=1 $R 1 ${tag}_s -depths 512 8192 -cs 16384 -samplers greedy -n 256 -reps 3 | sed "s/^/$arm: /" | grep RESULT
done
