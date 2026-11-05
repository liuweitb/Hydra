#!/usr/bin/env bash
# usage: wsl_hydra_py_src.sh SCRIPT.py [args...]   as wsl_hydra_py.sh plus layer_attribution/src on PYTHONPATH
export PYTHONPATH=/mnt/d/Hydra/layer_attribution/src:$PYTHONPATH
exec bash /mnt/d/Hydra/layer_attribution/research/config_faults/wsl_hydra_py.sh "$@"
