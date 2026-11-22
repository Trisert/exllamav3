#!/bin/bash
# A/B round 4: E8 (sm_60 8 bpw mul1 staged decode) vs the E6 build (control tree via EXL_SRC), P100
R=doc/lfm8_opt/run.sh
O=/home/nicola/exllamav3/tmp/lfm8_opt
C=/home/nicola/.claude/jobs/680193f5/tmp/ctl_e6
B="-depths 512 8192 -samplers greedy -n 256 -reps 3"
EXL_SRC=$C $R 1 p100_r4_ctl_a $B -json $O/p100_r4_ctl_a.json
$R 1 p100_r4_e8 $B -json $O/p100_r4_e8.json
EXL_SRC=$C $R 1 p100_r4_ctl_b $B -json $O/p100_r4_ctl_b.json
$R 1 p100_r4_e8b $B -json $O/p100_r4_e8b.json
$R 1 p100_r4_e8_oracle -oracle $O/oracle_p100_e8.json
