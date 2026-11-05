#!/usr/bin/env bash
# Run truncated Hydra jobs, at most 2 at a time, pinned to cores 8,9 (resource rule: 2 cores, 4 GB).
# usage: run_batch.sh JOBS_FILE MAX_STEPS [PARALLEL=2]      JOBS_FILE lines: NAME CONFIG_PATH|-
JOBS=$1; STEPS=$2; PAR=${3:-2}
[ "$PAR" -gt 2 ] && PAR=2
DATA=/mnt/d/hydra_work/data/uhumans2_office_mp3d
RUN_JOB=/mnt/d/Hydra/layer_attribution/scripts/run_job.sh
STATUS=$HOME/runs/logs/tmp_cfg_batch_status.txt
mkdir -p "$HOME/runs/logs"
export DATA RUN_JOB STEPS STATUS
run_one() {
  name=$1; cfg=$2
  s=$(date +%s)
  taskset -c 8,9 timeout 1500 bash "$RUN_JOB" "$name" "$DATA" "$cfg" "$STEPS" > /dev/null 2>&1
  rc=$?
  echo "$(date +%H:%M:%S) $name rc=$rc seconds=$(( $(date +%s) - s ))" >> "$STATUS"
}
export -f run_one
echo "$(date +%H:%M:%S) batch start: $JOBS steps=$STEPS par=$PAR" >> "$STATUS"
grep -v '^\s*$' "$JOBS" | xargs -P "$PAR" -L 1 bash -c 'run_one "$0" "$1"'
echo "$(date +%H:%M:%S) batch done: $JOBS" >> "$STATUS"
