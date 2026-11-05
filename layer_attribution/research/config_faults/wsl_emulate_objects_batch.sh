#!/usr/bin/env bash
# offline objects emulator: tolerance sweep (min size 40) + min-size sweep (tol 0.25) on the truncated ref and the full clean run
H=/mnt/d/Hydra/layer_attribution/research/config_faults
O=/mnt/d/hydra_work/results/config_faults
R=/home/hydra/runs/uh2
CASES="0.05:40,0.08:40,0.1:40,0.12:40,0.15:40,0.2:40,0.25:40,0.4:40,0.6124:40,1.0:40,1.5:40,2.0:40,3.0:40,4.0:40,5.0:40,0.25:60,0.25:80,0.25:100,0.25:150,0.25:200,0.25:300,0.25:400,0.25:600,0.25:1000"
PY="bash $H/wsl_hydra_py_src.sh $H/emulate_objects.py"
$PY $R/tmp_cfg_ref $O/emulate_objects_trunc_ref.json --cases $CASES
$PY $R/uh2_clean $O/emulate_objects_full_clean.json --cases $CASES
