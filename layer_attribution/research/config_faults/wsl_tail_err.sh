#!/usr/bin/env bash
# usage: wsl_tail_err.sh NAME  -> last lines of the run's stderr, and the INFO log tail
n=$1
echo "=== $n .err (last progress line + tail)"
tr '\r' '\n' < "$HOME/runs/logs/$n.err" | tail -5 | cut -c1-200
echo "=== $n INFO tail"
tail -5 "$HOME/runs/uh2/$n/logs/hydra_python.INFO" | cut -c1-220
echo "=== $n files"
ls -la "$HOME/runs/uh2/$n" "$HOME/runs/uh2/$n/backend" | cut -c1-150
