#!/bin/bash
# usage: doc/lfm3_mega/nsys_decode.sh <cvd: 0 = RTX 2060, 1 = P100> <tag> [depth]
# nsys trace of a greedy decode run with per-node CUDA graph timing (lower overhead than CUPTI via torch.profiler);
# serialized on the card's lane, one card at a time. Output: tmp/lfm3_mega/<tag>.nsys-rep + <tag>_cuda_gpu_trace.csv
CVD=$1; TAG=$2; D=${3:-512}
WT=/home/nicola/exllamav3/.claude/worktrees/lfm8-opt
OUT=/home/nicola/exllamav3/tmp/lfm3_mega
M=/home/nicola/exllamav3/models/LFM3-1.7B-exl3-8.0bpw
PY=/home/nicola/exllamav3/.venv/bin/python
NSYS=/opt/cuda/nsight_systems/target-linux-x64/nsys
mkdir -p "$OUT"
LANE=/home/nicola/exllamav3/tmp/gpu.lane
if [ "$CVD" = 1 ]; then LANE=/home/nicola/exllamav3/tmp/gpu.lane.p100; fi
cd "$WT" || exit 1
exec 9> "$LANE"
flock -w 3000 9 || exit 1
CUDA_VISIBLE_DEVICES=$CVD "$NSYS" profile -f true -t cuda --cuda-graph-trace=node -o "$OUT/$TAG" \
    "$PY" util/bench_decode.py -m "$M" -cs 16384 -cq 8 -depths "$D" -samplers greedy -n 256 -reps 1 \
    > "$OUT/$TAG.log" 2>&1
grep -E "RESULT" "$OUT/$TAG.log" | tail -3
"$NSYS" stats -f csv -r cuda_gpu_trace -o "$OUT/${TAG}" "$OUT/$TAG.nsys-rep" > /dev/null 2>&1
ls "$OUT" | grep "$TAG"
