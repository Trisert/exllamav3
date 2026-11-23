#!/bin/bash
# Native decode split on a non-current GPU: the P100 alone (CUDA_VISIBLE_DEVICES=1, cuda:0) vs as cuda:1 with the RTX
# 2060 current and idle; greedy ids must match. P100 lane only (one card computing at a time)
WT=/home/nicola/exllamav3/.claude/worktrees/lfm8-opt
M=/home/nicola/exllamav3/models/LFM3-1.7B-exl3-8.0bpw
PY=/home/nicola/exllamav3/.venv/bin/python
cd "$WT" || exit 1
exec 9> /home/nicola/exllamav3/tmp/gpu.lane.p100
flock -w 3000 9 || exit 1
CUDA_VISIBLE_DEVICES=1 $PY doc/lfm8_opt/secondary_gpu.py $M 0 2>&1 | grep -E "^IDS|^current|Error" | sed 's/^/alone:     /'
CUDA_VISIBLE_DEVICES=0,1 $PY doc/lfm8_opt/secondary_gpu.py $M 1 2>&1 | grep -E "^IDS|^current|Error" | sed 's/^/secondary: /'
