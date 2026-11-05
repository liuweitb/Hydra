#!/usr/bin/env bash
source ~/hydra_env.sh 2>/dev/null
python /mnt/d/Hydra/layer_attribution/research/signal_probe/explore_api.py "$1" 2>&1
