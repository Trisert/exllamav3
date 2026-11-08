#!/bin/bash
# Ornith decode-MoE microbench session 2: calibration (expert overlap), ksplit, and the per-item trace.
cd /home/nicola/exllamav3/.claude/worktrees/decode-draft || exit 1
P=/home/nicola/exllamav3/.venv/bin/python
B=tests/kernels/moe/coop_trace/bench_dual.py
A=tests/kernels/moe/coop_trace/analyze.py
nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm --format=csv,noheader
for ov in 0.0 0.5 0.75 1.0; do
  echo "### overlap $ov"
  $P $B --iters 200 --overlap $ov 2>&1 | grep -E "BENCH|Error" | sed 's/^/  /'
done
echo "### baseline overlap 0.5 + TRACE"
$P $B --iters 60 --overlap 0.5 --trace 2>&1 | grep -E "BENCH|TRACE|Error" | sed 's/^/  /'
$P $A P100 2>&1 | sed 's/^/  /'
echo "### KSPLIT=1 overlap 0.5"
EXL3_MOE_COOP_KSPLIT=1 $P $B --iters 200 --overlap 0.5 2>&1 | grep -E "BENCH|Error" | sed 's/^/  /'
echo "### KSPLIT=1 overlap 0.5 + TRACE"
EXL3_MOE_COOP_KSPLIT=1 EXL3_MOE_TRACE=1 $P $B --iters 60 --overlap 0.5 --trace 2>&1 | grep -E "BENCH|TRACE|Error" | sed 's/^/  /'
$P $A P100 2>&1 | sed 's/^/  /'
nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm --format=csv,noheader
