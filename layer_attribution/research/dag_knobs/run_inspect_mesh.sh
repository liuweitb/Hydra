#!/usr/bin/env bash
# Mesh links of the clean reference run; one core, 4 GB virtual cap, low priority.
source ~/hydra_env.sh 2>/dev/null
cd /mnt/d/Hydra/layer_attribution/research/dag_knobs || exit 1
ulimit -v 4000000
nice -n 19 taskset -c 10 python inspect_mesh_links.py "$HOME/runs/uh2/uh2_clean" inspect_clean_mesh_links.json > inspect_clean_mesh_links.txt 2>&1
echo "rc=$?"
