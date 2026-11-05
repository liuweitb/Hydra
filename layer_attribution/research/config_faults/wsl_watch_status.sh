#!/usr/bin/env bash
# emit each new line of the tmp_cfg batch status file; exit when a "batch done" line appears
S="$HOME/runs/logs/tmp_cfg_batch_status.txt"
n=$(wc -l < "$S")
while true; do
  m=$(wc -l < "$S")
  if [ "$m" -gt "$n" ]; then tail -n +$((n + 1)) "$S"; n=$m; fi
  if tail -n 1 "$S" | grep -q "batch done"; then break; fi
  sleep 5
done
