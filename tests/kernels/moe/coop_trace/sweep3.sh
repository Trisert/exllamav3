#!/bin/bash
cd /home/nicola/exllamav3/.claude/worktrees/decode-draft || exit 1
P=/home/nicola/exllamav3/.venv/bin/python
B=tests/kernels/moe/coop_trace/bench_dual.py
A=tests/kernels/moe/coop_trace/analyze.py
nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm --format=csv,noheader
for ov in 0.35 0.5; do
  echo "### overlap $ov trace"
  $P $B --iters 100 --overlap $ov --trace 2>&1 | grep -E "BENCH|TRACE|Error" | sed 's/^/  /'
  $P $A P100 2>&1 | grep -vE "RuntimeWarning|ret =|S\[n" | sed 's/^/  /'
done
echo "### overlap 0.5 KSPLIT=1 trace"
EXL3_MOE_COOP_KSPLIT=1 $P $B --iters 100 --overlap 0.5 --trace 2>&1 | grep -E "BENCH|TRACE|Error" | sed 's/^/  /'
$P $A P100 2>&1 | grep -vE "RuntimeWarning|ret =|S\[n" | sed 's/^/  /'
nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm --format=csv,noheader
