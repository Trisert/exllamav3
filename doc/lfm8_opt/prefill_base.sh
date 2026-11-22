#!/bin/bash
# Prefill baseline on both cards (perf script, generation skipped), 8-bit KV cache, up to 16k context
M=/home/nicola/exllamav3/models/LFM3-1.7B-exl3-8.0bpw
P=/home/nicola/exllamav3/.venv/bin/python
O=/home/nicola/exllamav3/tmp/lfm8_opt
L=/home/nicola/exllamav3/tmp
flock -w 1800 $L/gpu.lane.p100 env CUDA_VISIBLE_DEVICES=1 $P doc/lfm8_opt/prefill_perf.py -m $M -cs 16384 -cq 8 -max_length 16384 -sg "$@" > $O/prefill_p100.log 2>&1 &
flock -w 1800 $L/gpu.lane env CUDA_VISIBLE_DEVICES=0 $P doc/lfm8_opt/prefill_perf.py -m $M -cs 16384 -cq 8 -max_length 16384 -sg "$@" > $O/prefill_2060.log 2>&1
wait
for f in $O/prefill_p100.log $O/prefill_2060.log; do echo "== $f"; grep -v -E "━|Loading|Warm" $f | tail -25; done
