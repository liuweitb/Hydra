#!/usr/bin/env bash
# Feature table + query table for a set of splits (incremental: caches make re-runs cheap).
# usage: wsl_tables.sh NAME SPLITS [--final]      e.g.  wsl_tables.sh dev reference,repeat,benign,dev
# output: /mnt/d/hydra_work/results/tables/NAME/{features*.csv, queries*.csv, tables.log}
NAME=$1; SPLITS=$2; shift 2
OUT=/mnt/d/hydra_work/results/tables/$NAME
mkdir -p "$OUT"
HERE=$(dirname "$0")
{
  echo "== features $(date +%T)"
  bash "$HERE/wsl_features.sh" --out "$OUT/features.csv" --splits "$SPLITS" --jobs 2 "$@"
  echo "== queries $(date +%T)"
  bash "$HERE/wsl_py.sh" -m layer_attribution.query_table --root ~/runs/uh2 --out "$OUT/queries.csv" \
    --per-query "$OUT/queries_per_query.csv" --splits "$SPLITS" "$@"
  echo "== done $(date +%T)"
} > "$OUT/tables.log" 2>&1
echo "rc=$?"; tail -3 "$OUT/tables.log"
