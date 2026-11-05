#!/usr/bin/env bash
# Run python inside the Hydra env with the project on the path: wsl_py.sh ARGS...  (e.g. -m layer_attribution.query_table ...)
source ~/hydra_env.sh 2>/dev/null
export PYTHONPATH=/mnt/d/Hydra/layer_attribution/src:$PYTHONPATH
exec python "$@"
