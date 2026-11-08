#!/bin/bash
# Ornith decode-MoE microbench sweep (no model): serving shape bsz=2, 256 experts, top-8, Hi 2048, I 512, Ho 2048.
# usage: sweep_bench.sh   (run under util/gpu_lane.sh)
cd /home/nicola/exllamav3/.claude/worktrees/decode-draft || exit 1
P=/home/nicola/exllamav3/.venv/bin/python
B=tests/kernels/moe/coop_trace/bench_dual.py
run() {
  local tag=$1; shift
  echo "### $tag"
  env "$@" $P $B --iters 300 2>&1 | grep -E "BENCH|Error|error" | sed 's/^/  /'
}
nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm --format=csv,noheader
run "baseline (auto fused/wide/ksplit)"
run "KSPLIT=1" EXL3_MOE_COOP_KSPLIT=1
run "KSPLIT=2" EXL3_MOE_COOP_KSPLIT=2
run "KSPLIT=4" EXL3_MOE_COOP_KSPLIT=4
run "WIDE=0" EXL3_MOE_COOP_WIDE=0
run "FUSED=0 (two launches)" EXL3_MOE_COOP_FUSED=0
echo "### trace baseline"
EXL3_MOE_TRACE=1 $P $B --iters 100 --trace 2>&1 | grep -E "BENCH|TRACE|Error" | sed 's/^/  /'
$P tests/kernels/moe/coop_trace/analyze.py P100 2>&1 | sed 's/^/  /'
nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm --format=csv,noheader
