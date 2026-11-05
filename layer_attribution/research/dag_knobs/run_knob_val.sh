#!/usr/bin/env bash
# 5 truncated Hydra runs (120 steps) to validate knob effects; 2 cores, lowest priority.
# outputs ~/runs/uh2/tmp_dk_kv_*; delete with cleanup_apply_test.sh (removes tmp_dk_*)
HERE=/mnt/d/Hydra/layer_attribution/research/dag_knobs
DATA=/mnt/d/hydra_work/data/uhumans2_office_mp3d
for v in kv_inval kv_base kv_gvdmin kv_meshres kv_minedge; do
  nice -n 19 taskset -c 10,11 bash /mnt/d/Hydra/layer_attribution/scripts/run_job.sh tmp_dk_$v "$DATA" "$HERE/tmp_cfg/$v" 120
  echo "finished $v rc=$?"
done
