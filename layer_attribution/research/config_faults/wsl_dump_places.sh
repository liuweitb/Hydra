#!/usr/bin/env bash
# usage: wsl_dump_places.sh NAME [NAME ...]    cache places graphs of ~/runs/uh2/NAME into D:/hydra_work/results/config_faults
H=/mnt/d/Hydra/layer_attribution/research/config_faults
args=()
for n in "$@"; do args+=("$HOME/runs/uh2/$n"); done
bash "$H/wsl_hydra_py_src.sh" "$H/dump_places.py" /mnt/d/hydra_work/results/config_faults "${args[@]}"
