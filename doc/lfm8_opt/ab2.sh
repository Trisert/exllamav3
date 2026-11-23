#!/bin/bash
# A/B round 2: P100 E4 (wide for n == k at K >= 7), 2060 E5 (balanced split off). Same .so for all arms.
R=doc/lfm8_opt/run.sh
O=/home/nicola/exllamav3/tmp/lfm8_opt
B="-depths 512 8192 -samplers greedy -n 256 -reps 3"
(
EXL3_GEMV_PASCAL_K8_SQ=0 $R 1 p100_r2_ctl_a $B -json $O/p100_r2_ctl_a.json
$R 1 p100_r2_e4 $B -json $O/p100_r2_e4.json
EXL3_GEMV_PASCAL_K8_SQ=0 $R 1 p100_r2_ctl_b $B -json $O/p100_r2_ctl_b.json
$R 1 p100_r2_e4_oracle -oracle $O/oracle_p100_e4.json
) > $O/ab2_p100.txt 2>&1 &
(
$R 0 2060_r2_ctl_a $B -json $O/2060_r2_ctl_a.json
EXL3_GEMV_BALANCE=0 $R 0 2060_r2_e5 $B -json $O/2060_r2_e5.json
$R 0 2060_r2_ctl_b $B -json $O/2060_r2_ctl_b.json
EXL3_GEMV_BALANCE=0 $R 0 2060_r2_e5b $B -json $O/2060_r2_e5b.json
) > $O/ab2_2060.txt 2>&1 &
wait
cat $O/ab2_p100.txt $O/ab2_2060.txt
