#!/usr/bin/env bash
# objects emulator at the exact parameter values of the severity ladders (objects_min_size, objects_tol_merge)
H=/mnt/d/Hydra/layer_attribution/research/config_faults
O=/mnt/d/hydra_work/results/config_faults
R=/home/hydra/runs/uh2
MS="0.25:40"; for t in 76 105 145 200 276 381 525 617 725 1000; do MS="$MS,0.25:$t"; done
TM=""; for v in 0.4353 0.5743 0.7579 1.0 1.3195 1.7411 2.2974 2.639 3.0314 4.0; do TM="$TM,$v:40"; done
CASES="$MS$TM"
PY="bash $H/wsl_hydra_py_src.sh $H/emulate_objects.py"
$PY $R/uh2_clean $O/emulate_objects_ladders_full_clean.json --cases $CASES
$PY $R/tmp_cfg_ref $O/emulate_objects_ladders_trunc_ref.json --cases $CASES
