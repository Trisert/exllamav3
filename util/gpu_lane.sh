#!/bin/bash
# Serialize all GPU work on this host: one E2E/kernel job at a time.
# usage: util/gpu_lane.sh <seconds-timeout> <command...>
if [ "$#" -lt 2 ]; then echo "usage: $0 <timeout-sec> <cmd...>"; exit 2; fi
TMO=$1; shift
exec flock -w "$TMO" /home/nicola/exllamav3/tmp/gpu.lane "$@"
