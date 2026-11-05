#!/usr/bin/env bash
# Feature extraction over run directories, low priority: wsl_features.sh --out CSV [--splits reference,repeat,benign,dev] [--jobs 2] ...
# A cold snapshot build peaks near 1.6 GB per worker; stay at --jobs 2 or less while a sweep runs.
source ~/hydra_env.sh 2>/dev/null
export PYTHONPATH=/mnt/d/Hydra/layer_attribution/src:$PYTHONPATH
exec nice -n 10 python -m layer_attribution.features_extract "$@"
