#!/bin/bash
# usage: doc/lfm8_opt/run.sh <cvd: 0 = RTX 2060, 1 = P100> <tag> [bench_decode args...]   (env passes through)
# Serialized per card (gpu.lane / gpu.lane.p100), waits for <= 55 C before starting, log in tmp/lfm8_opt/<tag>.log
CVD=$1; TAG=$2; shift 2
WT=/home/nicola/exllamav3/.claude/worktrees/lfm8-opt
OUT=/home/nicola/exllamav3/tmp/lfm8_opt
M=${LFM_MODEL:-/home/nicola/exllamav3/models/LFM3-1.7B-exl3-8.0bpw}
PY=/home/nicola/exllamav3/.venv/bin/python
SMI=$((1 - CVD))
LANE=/home/nicola/exllamav3/tmp/gpu.lane
if [ "$CVD" = 1 ]; then LANE=/home/nicola/exllamav3/tmp/gpu.lane.p100; fi
cd "$WT" || exit 1
exec 9> "$LANE"
flock -w 3000 9 || exit 1
for i in $(seq 1 120); do
    t=$(nvidia-smi -i "$SMI" --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    [ "$t" -le 55 ] && break
    sleep 5
done
CUDA_VISIBLE_DEVICES=$CVD "$PY" util/bench_decode.py -m "$M" -cs 16384 -cq 8 "$@" > "$OUT/$TAG.log" 2>&1
grep -E "RESULT|ORACLE|oracle|Error" "$OUT/$TAG.log" | tail -12
