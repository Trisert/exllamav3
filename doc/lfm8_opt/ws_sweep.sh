#!/bin/bash
# P100 decode attention launch-shape sweep (warps,stages x key tile) at depth 512 and 32768, 64k 8-bit cache
R=doc/lfm8_opt/run.sh
B="-cs 65536 -depths 512 32768 -samplers greedy -n 256 -reps 3"
for cfg in "16:4,2" "16:2,1" "16:2,2" "16:8,2" "16:4,1" "64:2,1" "64:8,1" "16:4,2"; do
    bn=${cfg%%:*}; ws=${cfg##*:}
    EXL3_ATTN_DECODE_BN=$bn EXL3_ATTN_DECODE_WS=$ws $R 1 ws_${bn}_${ws/,/_} $B | grep RESULT \
        | sed -E "s/.*depth +([0-9]+) greedy: median ([0-9.]+) t\/s.*/bn $bn ws $ws depth \1: \2 t\/s/"
done
