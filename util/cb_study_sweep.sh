#!/bin/bash
# Sweep of pair-trellis LUT codebooks (see cb_study_vq.py). Output: one summary line per config.
PY=/home/nicola/exllamav3/.venv/bin/python
D=$(dirname "$0")
for cfg in "2 12 12 top" "2 12 12 low" "2 14 12 top" "2 14 12 low" "2 16 12 top" "2 14 10 top" "2 16 10 top" "2 16 9 top" "4 12 12 top" "4 16 12 top" "4 16 10 top"; do
  set -- $cfg
  $PY $D/cb_study_vq.py $1 $2 $3 $4 18 192 512 | tail -1
done
