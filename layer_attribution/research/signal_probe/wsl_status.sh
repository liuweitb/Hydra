#!/usr/bin/env bash
# print load, memory, python/hydra processes, and finished run dirs
uptime
free -m | head -2
ps -eo pid,pcpu,pmem,etime,cmd --sort=-pcpu | head -12 | cut -c1-200
echo "--- runs"
ls -la --time-style=long-iso ~/runs/uh2 | awk '{print $6, $7, $8}'
echo "--- finished (have dsg_with_mesh.json)"
for d in ~/runs/uh2/*/; do n=$(basename "$d"); if [ -f "$d/backend/dsg_with_mesh.json" ]; then echo "$n"; fi; done
