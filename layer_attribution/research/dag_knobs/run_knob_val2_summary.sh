#!/usr/bin/env bash
source ~/hydra_env.sh 2>/dev/null
cd /mnt/d/Hydra/layer_attribution/research/dag_knobs || exit 1
ulimit -v 4000000
nice -n 19 taskset -c 10 python knob_val_summary.py "$HOME/runs/uh2" knob_val2_result.json kv2_base kv2_meshres3 kv2_meshw20 kv2_meshw200 kv2_maxclu kv2_mergedist kv2_mindiff kv2_dsratio kv2_inert > knob_val2_result.txt 2>&1
echo "rc=$?"
tail -5 knob_val2_result.txt
