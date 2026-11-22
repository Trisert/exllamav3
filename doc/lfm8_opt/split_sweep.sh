#!/bin/bash
# Decode attention split / tile sweep at depth 32768 (64k 8-bit cache).
# usage: split_sweep.sh <cvd> [cfg ...]   cfg = M<mult>[_BN<block_n>], default list below
cvd=$1; shift
R=doc/lfm8_opt/run.sh
B="-cs 65536 -depths 32768 -samplers greedy -n 256 -reps 3"
cfgs=("$@"); [ ${#cfgs[@]} -eq 0 ] && cfgs=(M2 M8 M32 M8_BN64 M2)
for cfg in "${cfgs[@]}"; do
    m=${cfg%%_*}; m=${m#M}
    bn=""; [[ $cfg == *_BN* ]] && bn=${cfg##*_BN}
    env EXL3_ATTN_SPLIT_MULT=$m ${bn:+EXL3_ATTN_DECODE_BN=$bn} $R $cvd sw_${cvd}_$cfg $B | sed "s/^/$cfg: /" | grep RESULT
done
