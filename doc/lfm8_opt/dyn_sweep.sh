#!/bin/bash
# Balanced GEMV dynamic tail sweep (P100, standalone probe): EXL3_GEMV_DYN_NUM (sixteenths) x EXL3_GEMV_DYN_CL (slices)
WT=/home/nicola/exllamav3/.claude/worktrees/lfm8-opt
TMP=/home/nicola/.claude/jobs/680193f5/tmp
TI="-I/home/nicola/exllamav3/.venv/lib/python3.13/site-packages/torch/include -I/home/nicola/exllamav3/.venv/lib/python3.13/site-packages/torch/include/torch/csrc/api/include"
cd "$WT" || exit 1
for cfg in ${@:-"0 2" "2 2" "4 2" "4 1" "4 4" "6 2" "8 2"}; do
    set -- $cfg
    bin=$TMP/probe_dyn_$1_$2
    [ -x "$bin" ] || nvcc -arch=sm_60 -O3 -std=c++17 -Wno-deprecated-gpu-targets -DEXL3_GEMV_BAL_DBG=128 -DEXL3_GEMV_DYN_NUM=$1 -DEXL3_GEMV_DYN_CL=$2 \
        -Iexllamav3/exllamav3_ext $TI -o "$bin" doc/lfm8_opt/gemv_probe.cu 2>&1 | grep error
    echo "DYN_NUM=$1 CL=$2"
    PROBE_BIN=$bin doc/lfm8_opt/gemv_probe.sh "2048 7680 1 1" "7680 2048 1 1" "2048 15360 1 1" "2048 6144 1 1" | \
        sed -E 's/ \| phase.*//; s/^/  /'
done
