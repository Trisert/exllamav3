#!/bin/bash
# Alternate EXL3_GEMV_BALANCE 0/1 on one card for the given shapes. usage: bal_alt.sh <cvd> <shape...>
cvd=$1; shift
lane=/home/nicola/exllamav3/tmp/gpu.lane; [ "$cvd" = 1 ] && lane=/home/nicola/exllamav3/tmp/gpu.lane.p100
for bal in 0 1 0 1; do
    flock -w 900 $lane env CUDA_VISIBLE_DEVICES=$cvd EXL3_GEMV_BALANCE=$bal ${EXTRA_ENV} /home/nicola/exllamav3/.venv/bin/python doc/lfm8_opt/bench_gemv_k8.py "$@" 2>&1 | grep "^K="
done
