#!/bin/bash
# Alternate two builds of exllamav3 on the same chat-like workload (examples/bench_chat.py).
#   usage: bench_chat_ab.sh <dirA> <dirB> [rounds] [extra bench_chat.py args...]
# <dirA>/<dirB> are checkouts with the extension built in place (python setup.py build_ext --inplace).
# Order is A B A B ..., so slow thermal drift hits both builds alike. Defaults match a typical P100 + RTX 2060 setup.
A=$1; B=$2; ROUNDS=${3:-3}; shift 3
MODEL=${MODEL:-models/Qwen3.6-35B-A3B-exl3-3.00bpw}
PY=${PY:-python}
BENCH=${BENCH:-$(dirname "$(readlink -f "$0")")/bench_chat.py}
ARGS=${@:--gs 4.0,15.0 -cs 32768 -cq 4 -mtp -ndt 1 -dds}
for r in $(seq 1 $ROUNDS); do
    for D in "$A" "$B"; do
        echo "== round $r  build $D"
        RAW=$(mktemp)
        (cd "$D" && EXL_SRC="$D" $PY "$BENCH" -m "$MODEL" $ARGS -tag "$(basename $D)" > "$RAW" 2>&1)
        grep -E "^TURN|^SUMMARY" "$RAW" || { echo "   (no result, last lines of the output:)"; tail -5 "$RAW"; }
        rm -f "$RAW"
    done
done
