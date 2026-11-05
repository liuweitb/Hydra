#!/usr/bin/env bash
# One core, 4 GB cap; reads only uh2_clean and the seed-0 severity-0.5 label-fault runs.
source ~/hydra_env.sh 2>/dev/null
cd /mnt/d/Hydra/layer_attribution/research/dag_knobs || exit 1
ulimit -v 4500000
taskset -c 0 python verify_claims.py "$HOME/runs/uh2" verify_claims_out.json > verify_claims_out.txt 2>&1
echo "rc=$?"
tail -5 verify_claims_out.txt
