#!/bin/bash
# E19 (64-element fp16 partials in the hfma2 prefill GEMM, P100 only; one card at a time): ctl EXL3_HGEMM_HFMA2_K64=0 (32-element
# partials) vs EXL3_HGEMM_HFMA2_K64=1, ctl / cand / ctl, perf script to 16k; then prefill ppl ctl vs cand (top-1 agreement) and needle
M=/home/nicola/exllamav3/models/LFM3-1.7B-exl3-8.0bpw
P=/home/nicola/exllamav3/.venv/bin/python
O=/home/nicola/exllamav3/tmp/lfm8_opt
WT=/home/nicola/exllamav3/.claude/worktrees/lfm8-opt
cd "$WT" || exit 1
exec 9> /home/nicola/exllamav3/tmp/gpu.lane.p100
flock -w 3000 9 || exit 1
for arm in ctl_a cand ctl_b; do
    e="EXL3_HGEMM_HFMA2_K64=0"; [ $arm = cand ] && e="EXL3_HGEMM_HFMA2_K64=1"
    env CUDA_VISIBLE_DEVICES=1 $e X=1 $P doc/lfm8_opt/prefill_perf.py -m $M -cs 16384 -cq 8 -max_length 16384 -sg \
        > $O/pf_e19_$arm.log 2>&1
done
$P - <<'EOF'
import re
rows = {}
for arm in ("ctl_a", "cand", "ctl_b"):
    t = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", open(f"/home/nicola/exllamav3/tmp/lfm8_opt/pf_e19_{arm}.log", errors = "ignore").read())
    for L, v in re.findall(r"Length\s+(\d+):\s+([\d.]+) tokens/s", t):
        rows.setdefault(int(L), {})[arm] = float(v)
    if "Error" in t: print(arm, "ERROR", [l for l in t.splitlines() if "Error" in l][:2])
for L in sorted(rows):
    r = rows[L]
    print(f"P100 prefill {L:6d}: " + " / ".join(f"{r.get(a, 0):7.0f}" for a in ("ctl_a", "cand", "ctl_b")))
EOF
CUDA_VISIBLE_DEVICES=1 $P doc/lfm8_opt/prefill_ppl.py $M $O/pppl_p100_e19ctl.json 4096 2>&1 | grep PPL
CUDA_VISIBLE_DEVICES=1 EXL3_HGEMM_HFMA2_K64=1 $P doc/lfm8_opt/prefill_ppl.py $M $O/pppl_p100_e19.json 4096 2>&1 | grep PPL
$P -c "
import json
a = json.load(open('$O/pppl_p100_e19ctl.json'))['top1']; b = json.load(open('$O/pppl_p100_e19.json'))['top1']
print('top1 agree', sum(x == y for x, y in zip(a, b)), '/', len(a))"
CUDA_VISIBLE_DEVICES=1 EXL3_HGEMM_HFMA2_K64=1 PYTHONPATH=$WT $P doc/lfm8_opt/needle.py $M 2>&1 | grep -E "NEEDLE|FAIL"
