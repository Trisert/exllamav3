#!/bin/bash
# K=8 GEMV config sweep on both cards, balanced split on (heuristic) and off
P=/home/nicola/exllamav3/.venv/bin/python
L=/home/nicola/exllamav3/tmp
for cvd in 1 0; do
    lane=$L/gpu.lane; [ $cvd = 1 ] && lane=$L/gpu.lane.p100
    for bal in 1 0; do
        echo "== cvd $cvd ($( [ $cvd = 1 ] && echo P100 || echo 2060 ))"
        flock -w 900 $lane env CUDA_VISIBLE_DEVICES=$cvd EXL3_GEMV_BALANCE=$bal $P doc/lfm8_opt/bench_gemv_k8.py "$@" 2>&1 | grep "K="
    done
done
