#!/bin/bash
# usage: doc/lfm3_mega/ab_so_micro.sh <cand .so> [shapes...]   P100 GEMV microbench (bench_gemv_k8.py), ctl / cand / ctl by
# swapping the extension .so in the worktree (ctl = tmp/lfm3_mega/ext_ctl.so); restores ctl at the end
WT=/home/nicola/exllamav3/.claude/worktrees/lfm8-opt
SO=$WT/exllamav3_ext.cpython-313-x86_64-linux-gnu.so
CTL=/home/nicola/exllamav3/tmp/lfm3_mega/ext_ctl.so
CAND=$1; shift
cd "$WT" || exit 1
exec 9> /home/nicola/exllamav3/tmp/gpu.lane.p100
flock -w 3000 9 || exit 1
for arm in ctl cand ctl; do
    if [ $arm = ctl ]; then cp "$CTL" "$SO"; else cp "$CAND" "$SO"; fi
    echo "== $arm"
    CUDA_VISIBLE_DEVICES=1 /home/nicola/exllamav3/.venv/bin/python doc/lfm8_opt/bench_gemv_k8.py "$@" 2>&1 | tail -8
done
cp "$CTL" "$SO"
