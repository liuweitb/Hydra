#!/usr/bin/env bash
source ~/hydra_env.sh 2>/dev/null
cd /mnt/d/Hydra/layer_attribution/research/dag_knobs || exit 1
ulimit -v 4000000
R=$HOME/runs/uh2
nice -n 19 taskset -c 10 python compare_runs.py apply_test_compare.json $R/tmp_dk_base $R/tmp_dk_mis $R/tmp_dk_set $R/tmp_dk_typea $R/tmp_dk_typeb > apply_test_compare.txt 2>&1
echo "rc=$?"
