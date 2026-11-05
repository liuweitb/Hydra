#!/usr/bin/env bash
# list finished (dsg_with_mesh.json present) run dirs that are development data: clean, legacy s50, benign s03 (r0/r1), seed-0 grid
for d in "$HOME"/runs/uh2/*; do
  n=$(basename "$d")
  [ -f "$d/backend/dsg_with_mesh.json" ] || continue
  case "$n" in
    tmp_cfg_*) continue;;
    uh2_clean|uh2_clean_b) echo "$n";;
    *_s03_r0|*_s03_r1) echo "$n";;
    *_r0) echo "$n";;
    *_r1|*_r2) ;;   # held-out seeds: not touched
    uh2_*_s50) echo "$n";;
  esac
done
