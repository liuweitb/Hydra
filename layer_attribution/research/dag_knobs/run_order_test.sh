#!/usr/bin/env bash
# 2 truncated Hydra runs (60 steps): functor-order test. 2 cores, lowest priority.
# outputs ~/runs/uh2/tmp_dk_{ordbase,ordtest}; delete with cleanup_apply_test.sh
HERE=/mnt/d/Hydra/layer_attribution/research/dag_knobs
DATA=/mnt/d/hydra_work/data/uhumans2_office_mp3d
for v in ordbase ordtest; do
  nice -n 19 taskset -c 10,11 bash /mnt/d/Hydra/layer_attribution/scripts/run_job.sh tmp_dk_$v "$DATA" "$HERE/tmp_cfg/$v" 60
done
