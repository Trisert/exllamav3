#!/bin/bash
# Gated down projection (EXL3_GATED_DOWN): greedy oracle off vs on, the P100 then the RTX 2060 (one card at a time);
# bit-identical expected (max |dlogit| 0)
cd /home/nicola/exllamav3/.claude/worktrees/lfm8-opt || exit 1
O=/home/nicola/exllamav3/tmp/lfm8_opt
for c in 1 0; do
    EXL3_GATED_DOWN=0 doc/lfm8_opt/run.sh $c gd_off_$c -oracle $O/oracle_gd0_$c.json > /dev/null
    doc/lfm8_opt/run.sh $c gd_on_$c -oracle $O/oracle_gd1_$c.json | grep -E "Error"
    echo "cvd $c: $(/home/nicola/exllamav3/.venv/bin/python doc/lfm8_opt/cmp_oracle.py $O/oracle_gd0_$c.json $O/oracle_gd1_$c.json)"
done
