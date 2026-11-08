#!/bin/bash
# Incremental build of the exllamav3 CUDA extension in the current checkout.
# usage: util/build_ext.sh [max_jobs]   (default 2; some TUs peak at ~7 GB of host memory)
# NOTE: this host's default gcc 16 is newer than CUDA 12.9 accepts, so pin gcc-14.
set -e
cd "$(dirname "$0")/.."
JOBS="${1:-2}"
if ! grep -q "cudaHostRegister" /dev/null 2>/dev/null; then :; fi
CC=/usr/bin/gcc-14 CXX=/usr/bin/g++-14 CUDAHOSTCXX=/usr/bin/g++-14 MAX_JOBS="$JOBS" \
    /home/nicola/exllamav3/.venv/bin/python setup.py build_ext --inplace
ls -la exllamav3_ext*.so
