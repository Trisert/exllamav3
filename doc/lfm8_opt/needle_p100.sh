#!/bin/bash
# Needle at ~49k tokens (64k 8-bit cache) on the P100 only (one card at a time: UPS)
WT=/home/nicola/exllamav3/.claude/worktrees/lfm8-opt
M=/home/nicola/exllamav3/models/LFM3-1.7B-exl3-8.0bpw
O=/home/nicola/exllamav3/tmp/lfm8_opt
cd "$WT" || exit 1
exec 9> /home/nicola/exllamav3/tmp/gpu.lane.p100
flock -w 3000 9 || exit 1
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$WT NEEDLE_CACHE=65536 NEEDLE_FRACS=${NEEDLE_FRACS:-0.75} \
    /home/nicola/exllamav3/.venv/bin/python doc/lfm8_opt/needle.py $M > $O/needle64k_${1:-p100}.log 2>&1
grep -E "^ctx|^NEEDLE|Error" $O/needle64k_${1:-p100}.log
