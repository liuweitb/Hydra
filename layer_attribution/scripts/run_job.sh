#!/usr/bin/env bash
# Run Hydra on one dataset folder; output goes to $RUNS_ROOT/NAME (default ~/runs/uh2).
# usage: run_job.sh NAME DATASET_DIR [CONFIG_PATH|-] [MAX_STEPS|-]
source ~/hydra_env.sh 2>/dev/null
NAME=$1; DATA=$2; CFG=${3:--}; STEPS=${4:--}
OUT=${RUNS_ROOT:-$HOME/runs/uh2}
SRC=/mnt/d/Hydra/layer_attribution/src
mkdir -p "$OUT" "$HOME/runs/logs"
if [ -f "$OUT/$NAME/backend/dsg_with_mesh.json" ]; then echo "done already: $NAME"; exit 0; fi
args=("$DATA" "$OUT/$NAME" --force)
[ "$CFG" != "-" ] && args+=(--config-path "$CFG")
[ "$STEPS" != "-" ] && args+=(--max-steps "$STEPS")
s=$(date +%s)
PYTHONPATH="$SRC:$PYTHONPATH" python -m layer_attribution.pipeline.run_hydra "${args[@]}" \
  > "$HOME/runs/logs/$NAME.log" 2> "$HOME/runs/logs/$NAME.err"
rc=$?
[ -f "$DATA/injection_manifest.json" ] && [ -d "$OUT/$NAME" ] && cp "$DATA/injection_manifest.json" "$OUT/$NAME/"
[ "$CFG" != "-" ] && [ -f "$CFG/config_fault.json" ] && [ -d "$OUT/$NAME" ] && cp "$CFG/config_fault.json" "$OUT/$NAME/injection_manifest.json"
echo "$NAME rc=$rc seconds=$(( $(date +%s) - s ))"
exit $rc
