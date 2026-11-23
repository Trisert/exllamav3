#!/bin/bash
# P100 decode attention kernel time per token for key tile 16 vs 64 at depth 512 and 32768 (profiled windows)
for bn in 16 64; do
    for d in 512 32768; do
        EXL3_ATTN_DECODE_BN=$bn doc/lfm8_opt/run.sh 1 prof_bn${bn}_d$d -cs 65536 -depths $d -samplers greedy \
            -profile /home/nicola/exllamav3/tmp/lfm8_opt/trace_bn${bn}_d$d.json -pn 64 > /dev/null 2>&1
        echo "BN$bn depth $d: $(/home/nicola/exllamav3/.venv/bin/python util/analyze_decode_trace.py \
            /home/nicola/exllamav3/tmp/lfm8_opt/trace_bn${bn}_d$d.json 2>&1 | grep -E 'sum of kernel|paged_attn_decode_(split|combine)' \
            | sed -E 's/ +/ /g' | cut -c1-90 | tr '\n' '|')"
    done
done
