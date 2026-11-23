#!/bin/bash
# usage: doc/lfm8_opt/gemv_p100.sh <EXL3_GEMV_BALANCE values, comma-separated> [kxn ...]
# Cold K=8 GEMV microbench on the P100 lane, one process per balance mode
WT=/home/nicola/exllamav3/.claude/worktrees/lfm8-opt
PY=/home/nicola/exllamav3/.venv/bin/python
MODES=$1; shift
cd "$WT" || exit 1
exec 9> /home/nicola/exllamav3/tmp/gpu.lane.p100
flock -w 3000 9 || exit 1
for b in ${MODES//,/ }; do
    CUDA_VISIBLE_DEVICES=1 EXL3_GEMV_BALANCE=$b "$PY" doc/lfm8_opt/bench_gemv_k8.py "$@" 2>&1 | grep "^K="
done
