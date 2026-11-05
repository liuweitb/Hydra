#!/usr/bin/env bash
# 2 more truncated runs (40 steps): misspelled virtual-config type names
HERE=/mnt/d/Hydra/layer_attribution/research/dag_knobs
DATA=/mnt/d/hydra_work/data/uhumans2_office_mp3d
for v in typea typeb; do
  nice -n 19 taskset -c 10,11 bash /mnt/d/Hydra/layer_attribution/scripts/run_job.sh tmp_dk_$v "$DATA" "$HERE/tmp_cfg/$v" 40
done
