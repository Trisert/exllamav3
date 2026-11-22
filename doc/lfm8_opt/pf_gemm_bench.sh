#!/bin/bash
# LFM3 prefill GEMM shapes (M = 4096 tokens) through tests/kernels/gemm/hfma2_bench/bench.py on the P100 lane
# usage: doc/lfm8_opt/pf_gemm_bench.sh [variants=ext:64,cublas] [extra bench args...]
WT=/home/nicola/exllamav3/.claude/worktrees/lfm8-opt
V=${1:-ext:64,cublas}; shift
cd "$WT" || exit 1
exec 9> /home/nicola/exllamav3/tmp/gpu.lane.p100
flock -w 3000 9 || exit 1
SHAPES="1:4096:2048:7680,1:4096:7680:2048:float,1:4096:2048:6144,1:4096:2048:2048:float,1:4096:2048:2048,1:4096:2048:512"
CUDA_VISIBLE_DEVICES=1 SMI_IDX=0 TORCH_CUDA_ARCH_LIST=6.0 CC=gcc-14 CXX=g++-14 PYTHONPATH=$WT \
    /home/nicola/exllamav3/.venv/bin/python tests/kernels/gemm/hfma2_bench/bench.py --variants "$V" --custom "$SHAPES" "$@" 2>&1 | \
    grep -vE "Warmup|━"
