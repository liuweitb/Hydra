#!/usr/bin/env bash
# batch status: finished jobs of the latest batch, running jobs with step counter, load
S="$HOME/runs/logs/tmp_cfg_batch_status.txt"
start=$(grep -n "batch start" "$S" | tail -1 | cut -d: -f1)
tail -n +"$start" "$S"
echo "--- running tmp_cfg jobs"
for pid in $(pgrep -f "layer_attribution.pipeline.run_hydra.*tmp_cfg_"); do
  name=$(tr '\0' ' ' < /proc/$pid/cmdline | grep -o 'tmp_cfg_[A-Za-z0-9_]*' | head -1)
  step=$(tr '\r' '\n' < "$HOME/runs/logs/$name.err" | tail -1 | cut -c1-40)
  echo "$pid $name $(ps -o etime=,pcpu=,rss= -p $pid) $step"
done
echo "--- load"; uptime
echo "--- tmp_cfg run dirs"; ls "$HOME/runs/uh2" | grep -c '^tmp_cfg_'
