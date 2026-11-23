#!/bin/bash
# usage: doc/lfm8_opt/gemv_probe.sh "k n cfg bal" ...   (P100 lane; binary built from gemv_probe.cu)
BIN=${PROBE_BIN:-/home/nicola/.claude/jobs/680193f5/tmp/gemv_probe}
exec 9> /home/nicola/exllamav3/tmp/gpu.lane.p100
flock -w 3000 9 || exit 1
for a in "$@"; do
    CUDA_VISIBLE_DEVICES=1 "$BIN" $a
done
