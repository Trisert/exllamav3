#!/bin/bash
# Final-build qualification of the lfm8-opt branch on LFM3-1.7B 8.0bpw, the P100 then the RTX 2060 (one card at a
# time: UPS). Per card: kernel test suites touched by rounds 7-8, decode 512 / 8192 / 32768, prefill to 16k, greedy
# oracle, decode ppl, needle at 49k (64k 8-bit cache), a chat.py one-shot. Summary in tmp/lfm8_opt/qual_final.txt
WT=/home/nicola/exllamav3/.claude/worktrees/lfm8-opt
M=/home/nicola/exllamav3/models/LFM3-1.7B-exl3-8.0bpw
P=/home/nicola/exllamav3/.venv/bin/python
O=/home/nicola/exllamav3/tmp/lfm8_opt
S=$O/qual_final.txt
cd "$WT" || exit 1
: > $S
log() { echo "$@" | tee -a $S; }
strip() { sed 's/\x1b\[[0-9;?]*[A-Za-z]//g'; }
TESTS="tests/kernels/gemm/test_exl3_mgemv.py tests/kernels/gemm/test_exl3_gemv.py tests/kernels/gemm/test_hgemm_hfma2.py
       tests/kernels/gemm/test_exl3_mgemm_sliced.py tests/kernels/attention/test_decode_native.py
       tests/kernels/attention/test_fa60.py tests/kernels/attention/test_prefill_sdpa_window.py
       tests/kernels/attention/test_decode_live_split.py tests/kernels/norm/test_rms_norm_post_pre.py
       tests/kernels/norm/test_rms_norm.py"
for cvd in 1 0; do
    name=$([ $cvd = 1 ] && echo P100 || echo "RTX 2060")
    lane=$([ $cvd = 1 ] && echo /home/nicola/exllamav3/tmp/gpu.lane.p100 || echo /home/nicola/exllamav3/tmp/gpu.lane)
    log "===== $name"
    log "tests: $(doc/lfm8_opt/pytest_lane.sh $cvd $TESTS --maxfail=1000 2>&1 | strip | grep -E 'passed|failed' | tail -1)"
    for d in "512 8192 -cs 16384" "32768 -cs 65536"; do
        set -- $d
        X=1 doc/lfm8_opt/run.sh $cvd qf_${cvd}_$1 -depths ${@:1:$(($#-2))} ${@: -2} -samplers greedy -n 256 -reps 3 | \
            grep RESULT | sed -E 's/.*depth/  decode depth/' | tee -a $S
    done
    flock -w 3000 $lane env CUDA_VISIBLE_DEVICES=$cvd X=1 $P doc/lfm8_opt/prefill_perf.py -m $M -cs 16384 -cq 8 \
        -max_length 16384 -sg 2>&1 | strip | grep -oE "Length +[0-9]+: +[0-9.]+ tokens/s" | sed 's/^/  prefill /' | tee -a $S
    doc/lfm8_opt/run.sh $cvd qf_${cvd}_oracle -oracle $O/oracle_qf_$cvd.json > /dev/null
    ref=$([ $cvd = 1 ] && echo $O/oracle_p100_e17.json || echo $O/oracle_2060_e12.json)
    log "  oracle vs $(basename $ref): $($P doc/lfm8_opt/cmp_oracle.py $ref $O/oracle_qf_$cvd.json)"
    log "  $(flock -w 3000 $lane env CUDA_VISIBLE_DEVICES=$cvd PYTHONPATH=$WT $P doc/lfm8_opt/decode_ppl.py $M $O/dppl_qf_$cvd.json 2>&1 | grep DECODE_PPL)"
    flock -w 3000 $lane env CUDA_VISIBLE_DEVICES=$cvd PYTHONPATH=$WT NEEDLE_CACHE=65536 NEEDLE_FRACS=0.75 $P \
        doc/lfm8_opt/needle.py $M 2>&1 | grep -E "^ctx|^NEEDLE|Error" | sed 's/^/  /' | tee -a $S
    flock -w 3000 $lane env CUDA_VISIBLE_DEVICES=$cvd PYTHONPATH=$WT $P examples/chat.py -m $M -cs 16384 -cq 8 \
        -mode lfm3 -think -temp 0.1 -topk 50 -repp 1.1 -minp 0 -maxr 8000 -tps -basic \
        -prompt "Implement a quicksort in go" > $O/qf_chat_$cvd.log 2>&1
    log "  chat: $(strip < $O/qf_chat_$cvd.log | grep -cE 'func quickSort|func quicksort|func partition') quicksort/partition funcs, $(strip < $O/qf_chat_$cvd.log | grep -oE '[0-9.]+ tokens/s|[0-9.]+ t/s' | tail -1)"
done
log "done"
