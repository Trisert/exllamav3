#!/bin/bash
# E16b quality on the P100 (one card): greedy oracle HF=0 vs HF=1 at depth 512 and decode ppl (decode path, so the
# native split kernel runs), then a repeat of the speed A/B
WT=/home/nicola/exllamav3/.claude/worktrees/lfm8-opt
M=/home/nicola/exllamav3/models/LFM3-1.7B-exl3-8.0bpw
P=/home/nicola/exllamav3/.venv/bin/python
O=/home/nicola/exllamav3/tmp/lfm3_mega
cd "$WT" || exit 1
EXL3_ATTN_NATIVE_HF=0 doc/lfm8_opt/run.sh 1 e16b_or0 -oracle $O/oracle_e16b_0.json > /dev/null
EXL3_ATTN_NATIVE_HF=1 doc/lfm8_opt/run.sh 1 e16b_or1 -oracle $O/oracle_e16b_1.json > /dev/null
echo "oracle HF0 vs HF1: $($P doc/lfm8_opt/cmp_oracle.py $O/oracle_e16b_0.json $O/oracle_e16b_1.json)"
for hf in 0 1; do
    echo "HF=$hf $(flock -w 3000 /home/nicola/exllamav3/tmp/gpu.lane.p100 env EXL3_ATTN_NATIVE_HF=$hf CUDA_VISIBLE_DEVICES=1 \
        $P doc/lfm8_opt/decode_ppl.py $M $O/dppl_e16b_$hf.json 2>&1 | grep DECODE_PPL)"
done
doc/lfm8_opt/ab_dec.sh 1 e16b_r "EXL3_ATTN_NATIVE_HF=0" "EXL3_ATTN_NATIVE_HF=1"
