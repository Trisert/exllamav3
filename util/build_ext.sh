#!/bin/bash
# Incremental build of the exllamav3 CUDA extension in the current checkout.
# usage: util/build_ext.sh [max_jobs]   (default 6; each nvcc job needs ~1-2 GB RAM)
set -e
cd "$(dirname "$0")/.."
JOBS="${1:-6}"
MAX_JOBS="$JOBS" /home/nicola/exllamav3/.venv/bin/python setup.py build_ext --inplace
ls -la exllamav3_ext*.so
