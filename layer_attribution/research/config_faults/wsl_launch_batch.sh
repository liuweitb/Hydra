#!/usr/bin/env bash
# usage: wsl_launch_batch.sh JOBS_FILE MAX_STEPS   start run_batch.sh detached (own session, survives the caller)
H=/mnt/d/Hydra/layer_attribution/research/config_faults
setsid nohup bash "$H/run_batch.sh" "$1" "$2" 2 > /dev/null 2>&1 < /dev/null &
sleep 1
echo "launched pid(s):"; pgrep -f "run_batch.sh" | tr '\n' ' '; echo
