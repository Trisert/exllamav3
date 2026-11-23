#!/bin/bash
# Kernel A/B: control tree (EXL_SRC=$CTL) vs this worktree, alternating, on one card. usage: kab.sh <cvd> <K> [shape...]
cvd=$1; K=$2; shift 2
CTL=${CTL:-/home/nicola/.claude/jobs/680193f5/tmp/ctl_e6}
WT=/home/nicola/exllamav3/.claude/worktrees/lfm8-opt
lane=/home/nicola/exllamav3/tmp/gpu.lane; [ "$cvd" = 1 ] && lane=/home/nicola/exllamav3/tmp/gpu.lane.p100
for arm in ctl cand ctl cand; do
    src=$WT; [ $arm = ctl ] && src=$CTL
    echo "-- $arm"
    flock -w 900 $lane env CUDA_VISIBLE_DEVICES=$cvd BG_K=$K EXL_SRC=$src /home/nicola/exllamav3/.venv/bin/python $WT/doc/lfm8_opt/bench_gemv_k8.py "$@" 2>&1 | grep "^K=" | sed 's/ | narrow.*//'
done
