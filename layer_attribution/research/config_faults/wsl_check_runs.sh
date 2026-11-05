#!/usr/bin/env bash
# list tmp_cfg_ run dirs with completeness evidence (dsg file, steps logged, last log line time, size)
for d in "$HOME"/runs/uh2/tmp_cfg_*; do
  n=$(basename "$d")
  f="$d/backend/dsg_with_mesh.json"
  if [ -f "$f" ]; then sz=$(stat -c %s "$f"); else sz=none; fi
  steps=$(grep -c "Got input @" "$d/logs/hydra_python.INFO" 2>/dev/null)
  mt=$(stat -c %y "$d" | cut -c1-19)
  echo "$n dsg=$sz steps_logged=$steps dir_mtime=$mt"
done
echo "--- logs"
ls -la "$HOME"/runs/logs | grep tmp_cfg_ || true
echo "--- procs"
ps -eo pid,etime,pcpu,rss,args | grep -E "run_hydra|hydra_python|run_job" | grep -v grep | cut -c1-200
echo "--- cores"
nproc; uptime; free -m | head -2
