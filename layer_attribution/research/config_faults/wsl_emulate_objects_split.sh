#!/usr/bin/env bash
# objects emulator at the objects_tol_split values (truncated reference and full clean)
H=/mnt/d/Hydra/layer_attribution/research/config_faults
O=/mnt/d/hydra_work/results/config_faults
R=/home/hydra/runs/uh2
CASES="0.05:40,0.0559:40,0.0627:40,0.0703:40,0.0788:40,0.0884:40,0.0992:40,0.1118:40,0.125:40,0.1403:40,0.1575:40,0.1768:40"
PY="bash $H/wsl_hydra_py_src.sh $H/emulate_objects.py"
$PY $R/tmp_cfg_ref $O/emulate_objects_split_trunc_ref.json --cases $CASES
$PY $R/uh2_clean $O/emulate_objects_split_full_clean.json --cases $CASES
