#!/usr/bin/env bash
# usage: wsl_hydra_py.sh SCRIPT.py [args...]   Hydra venv python (spark_dsg), pinned to 2 cores, 1 BLAS thread
source ~/hydra_env.sh 2>/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec taskset -c 8,9 python "$@"
