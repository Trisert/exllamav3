#!/bin/bash
# Generic decode A/B on one card: ctl / cand / ctl at depths 512 / 8192 / 32768.
# usage: doc/lfm8_opt/ab_dec.sh <cvd> <tag> "<ctl env>" "<cand env>"   e.g. ab_dec.sh 1 e20 "EXL3_ATTN_NATIVE_WARPS=4" "EXL3_ATTN_NATIVE_WARPS=8"
R=doc/lfm8_opt/run.sh
CVD=$1; TAG=$2; CTL=$3; CAND=$4
S="-depths 512 8192 -cs 16384 -samplers greedy -n 256 -reps 3"
L="-depths 32768 -cs 65536 -samplers greedy -n 256 -reps 3"
for arm in ctl_a cand ctl_b; do
    e="$CTL"; [ $arm = cand ] && e="$CAND"
    env $e X=1 $R $CVD ${TAG}_${arm}_s $S | grep RESULT | sed -E "s/.*depth/$arm: depth/; s/ greedy: median / /; s/ t\/s.*/ t\/s/"
    env $e X=1 $R $CVD ${TAG}_${arm}_l $L | grep RESULT | sed -E "s/.*depth/$arm: depth/; s/ greedy: median / /; s/ t\/s.*/ t\/s/"
done
