#!/bin/bash
# Correctness + within-load A/B loads for the async-copy change (prefill-gpu).
# usage: run_gate.sh <tag> <async_dma 0|1> <toggle-rounds> [lengths] [repeat] [abba pattern]
# Run from the worktree (model path hashes to the worktree expert-stats key), static expert
# placement, same backed-up stats file restored before every load so both loads see identical
# placement.
set -u
WT=/home/nicola/exllamav3/.claude/worktrees/prefill-gpu
PY=/home/nicola/exllamav3/.venv/bin/python
STATS=$HOME/.cache/exllamav3/expert_stats/8d811058b3a9550f.json
BACKUP=$WT/tmp/expert_stats_worktree.json
TAG=$1; ASYNC=$2; ROUNDS=$3; LENS=${4:-1024}; REP=${5:-1}; ABBA=${6:-}
mkdir -p $WT/tmp/logs
if [ -f "$BACKUP" ]; then cp "$BACKUP" "$STATS"; fi
cd $WT
echo "== gate $TAG $(date -Is) ASYNC_DMA=$ASYNC SWAP=0 BC_ATTN=0 rounds=$ROUNDS lens=$LENS rep=$REP abba=$ABBA" | tee -a $WT/tmp/logs/ab.log
nvidia-smi -i 0 --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader
EXTRA=""
if [ -n "$ABBA" ]; then EXTRA="-abba_async $ABBA"; fi
EXL3_MOE_ASYNC_DMA=$ASYNC EXL3_MOE_CPU_SWAP=0 EXL3_BC_ATTN=0 $PY util/bench_flash.py -m models/Qwen3.8-Flash-Next-exl3-1.66bpw -gs 4.0,15.0 -cs 24576 -cq 4 -mcs 256 -run prefill -lengths $LENS -repeat $REP -check_text 0 -check_text_toggle "$ROUNDS" -copy_sites $EXTRA -tag "$TAG" 2>&1 | tee $WT/tmp/logs/$TAG.log
nvidia-smi -i 0 --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv,noheader
if [ -f "$BACKUP" ]; then cp "$BACKUP" "$STATS"; fi
