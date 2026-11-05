#!/usr/bin/env bash
# usage: wsl_py.sh SCRIPT.py [args...]   run a python script in the Hydra venv, pinned to 2 cores, 1 BLAS thread
source ~/hydra_env.sh 2>/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec taskset -c 10,11 python "$@"
