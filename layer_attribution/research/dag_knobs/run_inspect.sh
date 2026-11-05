#!/usr/bin/env bash
# Structure of the reference run graphs; one core, capped memory.
source ~/hydra_env.sh 2>/dev/null
cd /mnt/d/Hydra/layer_attribution/research/dag_knobs || exit 1
ulimit -v 4000000
for stage in backend frontend; do
  taskset -c 0 python inspect_graph.py "$HOME/runs/uh2/uh2_clean/$stage/dsg.json" "inspect_clean_${stage}.json" > "inspect_clean_${stage}.txt" 2>&1
  echo "$stage rc=$?"
done
