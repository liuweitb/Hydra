#!/usr/bin/env bash
# usage: wsl_figures.sh EVAL_DIR QUERIES_CSV TAG   (EVAL_DIR holds predictions_TAG.csv and summary_TAG.json)
E=$1; Q=$2; TAG=$3
bash "$(dirname "$0")/wsl_py.sh" -m layer_attribution.evaluation.figures --predictions "$E/predictions_$TAG.csv" \
  --queries "$Q" --summary "$E/summary_$TAG.json" --out "$E/figures" --tag "$TAG"
echo "rc=$?"; ls "$E/figures"
