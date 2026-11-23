#!/bin/bash
# Decode A/B between two extension builds on one card: ctl / cand / ctl at depths 512 / 8192 / 32768, by swapping the
# .so in the worktree (ctl = tmp/lfm3_mega/ext_ctl.so), plus a greedy oracle per arm; restores ctl at the end.
# usage: doc/lfm3_mega/ab_so_dec.sh <cvd> <tag> <cand .so>
WT=/home/nicola/exllamav3/.claude/worktrees/lfm8-opt
SO=$WT/exllamav3_ext.cpython-313-x86_64-linux-gnu.so
CTL=/home/nicola/exllamav3/tmp/lfm3_mega/ext_ctl.so
O=/home/nicola/exllamav3/tmp/lfm3_mega
CVD=$1; TAG=$2; CAND=$3
cd "$WT" || exit 1
R=doc/lfm8_opt/run.sh
S="-depths 512 8192 -cs 16384 -samplers greedy -n 256 -reps 3"
L="-depths 32768 -cs 65536 -samplers greedy -n 256 -reps 3"
for arm in ctl_a cand ctl_b; do
    if [ $arm = cand ]; then cp "$CAND" "$SO"; else cp "$CTL" "$SO"; fi
    $R $CVD ${TAG}_${arm}_s $S | grep RESULT | sed -E "s/.*depth/$arm: depth/; s/ greedy: median / /; s/ t\/s.*/ t\/s/"
    $R $CVD ${TAG}_${arm}_l $L | grep RESULT | sed -E "s/.*depth/$arm: depth/; s/ greedy: median / /; s/ t\/s.*/ t\/s/"
    if [ $arm != ctl_b ]; then $R $CVD ${TAG}_${arm}_or -oracle $O/oracle_${TAG}_${arm}.json > /dev/null; fi
done
cp "$CTL" "$SO"
echo "oracle ctl vs cand: $(/home/nicola/exllamav3/.venv/bin/python doc/lfm8_opt/cmp_oracle.py $O/oracle_${TAG}_ctl_a.json $O/oracle_${TAG}_cand.json)"
