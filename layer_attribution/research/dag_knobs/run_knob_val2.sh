#!/usr/bin/env bash
# 9 truncated Hydra runs (250 steps) to validate knob effects; 2 cores, lowest priority.
# outputs ~/runs/uh2/tmp_dk_kv2_*; delete with cleanup_apply_test.sh (removes tmp_dk_*)
HERE=/mnt/d/Hydra/layer_attribution/research/dag_knobs
DATA=/mnt/d/hydra_work/data/uhumans2_office_mp3d
for v in kv2_base kv2_meshres3 kv2_meshw20 kv2_meshw200 kv2_maxclu kv2_mergedist kv2_mindiff kv2_dsratio kv2_inert; do
  nice -n 19 taskset -c 10,11 bash /mnt/d/Hydra/layer_attribution/scripts/run_job.sh tmp_dk_$v "$DATA" "$HERE/tmp_cfg/$v" 250
  echo "finished $v rc=$?"
done
