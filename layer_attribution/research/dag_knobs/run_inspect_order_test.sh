#!/usr/bin/env bash
source ~/hydra_env.sh 2>/dev/null
cd /mnt/d/Hydra/layer_attribution/research/dag_knobs || exit 1
ulimit -v 3000000
nice -n 19 taskset -c 10 python inspect_order_test.py "$HOME/runs/uh2" order_test_result.json > order_test_result.txt 2>&1
echo "rc=$?"
cat order_test_result.txt | tail -40
