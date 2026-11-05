#!/usr/bin/env bash
# usage: wsl_py3.sh SCRIPT.py [args...]   system python (no Hydra), pinned to 2 cores
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec taskset -c 8,9 python3 "$@"
