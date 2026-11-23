#!/bin/bash
# LFM3-1.7B 4.0bpw vs 8.0bpw on the current build: decode at 512 / 8192 / 32768 and decode-mode ppl, the P100 then
# the RTX 2060 (strictly one card at a time)
WT=/home/nicola/exllamav3/.claude/worktrees/lfm8-opt
P=/home/nicola/exllamav3/.venv/bin/python
O=/home/nicola/exllamav3/tmp/lfm3_mega
M4=/home/nicola/exllamav3/models/LFM3-1.7B-exl3-4.0bpw
cd "$WT" || exit 1
S="-depths 512 8192 -cs 16384 -samplers greedy -n 256 -reps 3"
L="-depths 32768 -cs 65536 -samplers greedy -n 256 -reps 3"
for c in 1 0; do
    lane=/home/nicola/exllamav3/tmp/gpu.lane; [ $c = 1 ] && lane=/home/nicola/exllamav3/tmp/gpu.lane.p100
    echo "== cvd $c"
    LFM_MODEL=$M4 doc/lfm8_opt/run.sh $c b4_${c}_s $S | grep RESULT | sed -E 's/.*depth/4.0bpw depth/; s/ greedy: median / /; s/ t\/s.*/ t\/s/'
    LFM_MODEL=$M4 doc/lfm8_opt/run.sh $c b4_${c}_l $L | grep RESULT | sed -E 's/.*depth/4.0bpw depth/; s/ greedy: median / /; s/ t\/s.*/ t\/s/'
    echo "4.0bpw $(flock -w 3000 $lane env CUDA_VISIBLE_DEVICES=$c $P doc/lfm8_opt/decode_ppl.py $M4 $O/dppl_4bpw_$c.json 2>&1 | grep DECODE_PPL)"
done
