#!/usr/bin/env bash
# Extract snapshots for all finished DEVELOPMENT runs: reference, clean repeat, benign runs (s03, seeds 0 and 1),
# seed-0 fault runs (s20/s50/s80). Held-out names (s20/s50/s80 with _r1/_r2) and tmp runs never match the pattern.
source ~/hydra_env.sh 2>/dev/null
RUNS=$HOME/runs/uh2
OUT=/mnt/d/hydra_work/results/signal_probe/snapshots
mkdir -p "$OUT"
F="label_confusion|label_dropout|depth_noise|depth_holes|pose_drift|frame_drop"
args=()
for d in "$RUNS"/*/; do
  n=$(basename "$d")
  if [[ "$n" =~ ^uh2_(clean|clean_b)$ ]] || [[ "$n" =~ ^uh2_($F)_s03_r[01]$ ]] || [[ "$n" =~ ^uh2_($F)_s(20|50|80)(_r0)?$ ]]; then
    f="$d/backend/dsg_with_mesh.json"
    [ -f "$f" ] || continue
    # finished = file untouched for 2 min (and manifest copied for fault runs)
    age=$(( $(date +%s) - $(stat -c %Y "$f") ))
    [ "$age" -gt 120 ] || continue
    if [[ "$n" != uh2_clean* ]] && [ ! -f "$d/injection_manifest.json" ]; then continue; fi
    args+=("${d%/}")
  fi
done
echo "candidates: ${#args[@]}"
nice -n 10 python /mnt/d/Hydra/layer_attribution/research/signal_probe/extract_snapshots.py "$OUT" "${args[@]}" 2>&1
