#!/bin/bash
# E17 (post-norm + next pre-norm in one launch, rms_norm_post_pre): ctl EXL3_NORM_POST_PRE=0 vs default,
# cand / ctl / cand at depths 512 / 8192, the P100 then the RTX 2060 (one card at a time); bit-identical oracle check
R=doc/lfm8_opt/run.sh
O=/home/nicola/exllamav3/tmp/lfm8_opt
P=/home/nicola/exllamav3/.venv/bin/python
S="-depths 512 8192 -cs 16384 -samplers greedy -n 256 -reps 3"
for cvd in 1 0; do
    for arm in cand_a ctl cand_b; do
        e="EXL3_NORM_POST_PRE=1"; [ $arm = ctl ] && e="EXL3_NORM_POST_PRE=0"
        env $e X=1 $R $cvd e17_${cvd}_${arm} $S | sed "s/^/$arm: /" | grep RESULT
    done
done
$R 1 e17_oracle -oracle $O/oracle_p100_e17.json | grep ORACLE
$P doc/lfm8_opt/cmp_oracle.py $O/oracle_p100_e15.json $O/oracle_p100_e17.json
