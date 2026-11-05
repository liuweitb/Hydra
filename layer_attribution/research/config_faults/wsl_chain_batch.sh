#!/usr/bin/env bash
# usage: wsl_chain_batch.sh JOBS_FILE MAX_STEPS   detached: wait until no run_batch.sh is running, then run the next batch
H=/mnt/d/Hydra/layer_attribution/research/config_faults
JOBS=$1; STEPS=$2
if [ "$3" != "--detached" ]; then
  setsid nohup bash "$0" "$JOBS" "$STEPS" --detached > /dev/null 2>&1 < /dev/null &
  sleep 1; echo "chained: $JOBS after the running batch"; exit 0
fi
while pgrep -f "config_faults/run_batch.sh" > /dev/null; do sleep 20; done
exec bash "$H/run_batch.sh" "$JOBS" "$STEPS" 2
