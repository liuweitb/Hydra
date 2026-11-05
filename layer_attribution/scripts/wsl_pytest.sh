#!/usr/bin/env bash
# Run pytest in the Hydra env: wsl_pytest.sh ARGS...  (e.g. tests/snapshot tests/features -q)
# pytest is not in ~/venvs/hydra; it sits in ~/pytest_lib (pip install --target ~/pytest_lib pytest), so the shared venv stays untouched.
source ~/hydra_env.sh 2>/dev/null
export PYTHONPATH=$HOME/pytest_lib:/mnt/d/Hydra/layer_attribution/src
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /mnt/d/Hydra/layer_attribution
exec nice -n 10 python -m pytest -p no:cacheprovider "$@"
