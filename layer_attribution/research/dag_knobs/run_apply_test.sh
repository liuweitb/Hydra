#!/usr/bin/env bash
# 3 truncated Hydra runs (40 steps) to test apply-vs-ignore of config keys; 2 cores, lowest priority.
# outputs ~/runs/uh2/tmp_dk_{base,mis,set}; delete with cleanup_apply_test.sh
HERE=/mnt/d/Hydra/layer_attribution/research/dag_knobs
DATA=/mnt/d/hydra_work/data/uhumans2_office_mp3d
for v in base mis set; do
  nice -n 19 taskset -c 10,11 bash /mnt/d/Hydra/layer_attribution/scripts/run_job.sh tmp_dk_$v "$DATA" "$HERE/tmp_cfg/$v" 40
done
