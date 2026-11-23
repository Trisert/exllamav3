#!/bin/bash
# Decode attention tile 16 (default) vs 64 at short and mid depth, ctl / cand / ctl, both cards in parallel
R=doc/lfm8_opt/run.sh
B="-cs 16384 -depths 512 8192 -samplers greedy -n 256 -reps 3"
card() {
    local cvd=$1
    $R $cvd bn_${cvd}_ctl_a $B | sed "s/^/ctl_a: /" | grep RESULT
    EXL3_ATTN_DECODE_BN=64 $R $cvd bn_${cvd}_cand $B | sed "s/^/bn64: /" | grep RESULT
    $R $cvd bn_${cvd}_ctl_b $B | sed "s/^/ctl_b: /" | grep RESULT
}
card 1 > /home/nicola/exllamav3/tmp/lfm8_opt/ab_bn_p100.txt 2>&1 &
card 0 > /home/nicola/exllamav3/tmp/lfm8_opt/ab_bn_2060.txt 2>&1
wait
cat /home/nicola/exllamav3/tmp/lfm8_opt/ab_bn_p100.txt /home/nicola/exllamav3/tmp/lfm8_opt/ab_bn_2060.txt | sed 's/graphs=1 LFM3-1.7B-exl3-8.0bpw //;s/RESULT //'
