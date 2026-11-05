#!/usr/bin/env bash
source ~/hydra_env.sh 2>/dev/null
cd /mnt/d/Hydra/layer_attribution/research/dag_knobs || exit 1
ulimit -v 4000000
nice -n 19 taskset -c 10 python knob_val_summary.py "$HOME/runs/uh2" knob_val_result.json > knob_val_result.txt 2>&1
echo "rc=$?"
tail -150 knob_val_result.txt
