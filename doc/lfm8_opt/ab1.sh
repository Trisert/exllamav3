#!/bin/bash
# A/B round 1: P100 E2 (K=7/8 GEMV on Pascal) / E3 (unfuse), 2060 E1 (unfuse gate/up at K=8). Same .so for all arms.
R=doc/lfm8_opt/run.sh
O=/home/nicola/exllamav3/tmp/lfm8_opt
B="-depths 512 8192 -samplers greedy -n 256 -reps 3"
(
EXL3_GEMV_PASCAL_K8=0 EXL3_UNFUSE_K8=0 $R 1 p100_r1_ctl_a $B -json $O/p100_r1_ctl_a.json
EXL3_UNFUSE_K8=0 $R 1 p100_r1_e2 $B -json $O/p100_r1_e2.json
$R 1 p100_r1_e23 $B -json $O/p100_r1_e23.json
EXL3_GEMV_PASCAL_K8=0 EXL3_UNFUSE_K8=0 $R 1 p100_r1_ctl_b $B -json $O/p100_r1_ctl_b.json
$R 1 p100_r1_e23_oracle -oracle $O/oracle_p100_e23.json
) > $O/ab1_p100.txt 2>&1 &
(
EXL3_UNFUSE_K8=0 $R 0 2060_r1_ctl_a $B -samplers greedy user -json $O/2060_r1_ctl_a.json
$R 0 2060_r1_e1 $B -samplers greedy user -json $O/2060_r1_e1.json
EXL3_UNFUSE_K8=0 $R 0 2060_r1_ctl_b $B -samplers greedy user -json $O/2060_r1_ctl_b.json
$R 0 2060_r1_e1_oracle -oracle $O/oracle_2060_e1.json
) > $O/ab1_2060.txt 2>&1 &
wait
cat $O/ab1_p100.txt $O/ab1_2060.txt
