#!/usr/bin/env bash
source ~/hydra_env.sh 2>/dev/null
cd /mnt/d/Hydra/layer_attribution/research/dag_knobs || exit 1
ulimit -v 3000000
taskset -c 0 python dump_default_config.py default_config.yaml
