#!/usr/bin/env bash
source ~/hydra_env.sh 2>/dev/null
cd /mnt/d/Hydra/layer_attribution/research/dag_knobs || exit 1
ulimit -v 4000000
taskset -c 0 python probe_api.py "$HOME/runs/uh2/uh2_clean/backend/dsg.json"
