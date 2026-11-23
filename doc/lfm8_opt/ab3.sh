#!/bin/bash
# A/B round 3: E6 (hgemv split-K scratch inside graph captures; control EXL3_HGEMV_GRAPH=0), both cards
R=doc/lfm8_opt/run.sh
O=/home/nicola/exllamav3/tmp/lfm8_opt
B="-depths 512 8192 -samplers greedy -n 256 -reps 3"
(
EXL3_HGEMV_GRAPH=0 $R 1 p100_r3_ctl_a $B -json $O/p100_r3_ctl_a.json
$R 1 p100_r3_e6 $B -json $O/p100_r3_e6.json
EXL3_HGEMV_GRAPH=0 $R 1 p100_r3_ctl_b $B -json $O/p100_r3_ctl_b.json
$R 1 p100_r3_e6b $B -json $O/p100_r3_e6b.json
$R 1 p100_r3_e6_oracle -oracle $O/oracle_p100_e6.json
) > $O/ab3_p100.txt 2>&1 &
(
EXL3_HGEMV_GRAPH=0 $R 0 2060_r3_ctl_a $B -json $O/2060_r3_ctl_a.json
$R 0 2060_r3_e6 $B -json $O/2060_r3_e6.json
EXL3_HGEMV_GRAPH=0 $R 0 2060_r3_ctl_b $B -json $O/2060_r3_ctl_b.json
$R 0 2060_r3_e6b $B -json $O/2060_r3_e6b.json
$R 0 2060_r3_e6_oracle -oracle $O/oracle_2060_e6.json
) > $O/ab3_2060.txt 2>&1 &
wait
cat $O/ab3_p100.txt $O/ab3_2060.txt
