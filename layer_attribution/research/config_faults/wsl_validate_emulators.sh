#!/usr/bin/env bash
# usage: wsl_validate_emulators.sh   run the default-config emulator validation on all finished dev-data runs (skips done ones)
H=/mnt/d/Hydra/layer_attribution/research/config_faults
OUT=/mnt/d/hydra_work/results/config_faults/emulator_default_validation.json
names=$(bash "$H/wsl_list_runs.sh" | grep -v '^uh2_clean_b$')
source ~/hydra_env.sh 2>/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/mnt/d/Hydra/layer_attribution/src:$PYTHONPATH
exec nice -n 15 taskset -c 8,9 python "$H/validate_emulators_default.py" /home/hydra/runs/uh2 "$OUT" $names
