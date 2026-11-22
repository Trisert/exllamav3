#!/bin/bash
# One-shot chat.py run of a coding prompt with LFM3's recommended sampling and the think tag (RTX 2060 lane)
cd /home/nicola/exllamav3/.claude/worktrees/lfm8-opt
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/nicola/exllamav3/.claude/worktrees/lfm8-opt
exec flock /home/nicola/exllamav3/tmp/gpu.lane /home/nicola/exllamav3/.venv/bin/python examples/chat.py \
    -m /home/nicola/exllamav3/models/LFM3-1.7B-exl3-8.0bpw -cs 16384 -cq 8 -mode chatml -think \
    -temp 0.1 -topk 50 -repp 1.1 -minp 0 -maxr 8000 -tps -basic -prompt "${1:-Implement a quicksort in go}"
