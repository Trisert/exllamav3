#!/bin/bash
# usage: doc/lfm8_opt/pytest_lane.sh <cvd: 0 = RTX 2060, 1 = P100> <pytest args...>   (serialized on the card's lane)
CVD=$1; shift
WT=/home/nicola/exllamav3/.claude/worktrees/lfm8-opt
LANE=/home/nicola/exllamav3/tmp/gpu.lane
if [ "$CVD" = 1 ]; then LANE=/home/nicola/exllamav3/tmp/gpu.lane.p100; fi
cd "$WT" || exit 1
exec 9> "$LANE"
flock -w 3000 9 || exit 1
PYTHONPATH=$WT CUDA_VISIBLE_DEVICES=$CVD /home/nicola/exllamav3/.venv/bin/python -m pytest -q -x "$@" 2>&1 | tail -15
