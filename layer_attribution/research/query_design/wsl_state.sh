#!/usr/bin/env bash
# which development runs are finished (have dsg_with_mesh.json); never touches non-benign _r1/_r2 run dirs
echo "--- load / mem"; uptime; free -m | sed -n 1,2p
echo "--- finished development runs"
for d in ~/runs/uh2/uh2_clean ~/runs/uh2/uh2_clean_b ~/runs/uh2/uh2_*_s03_r0 ~/runs/uh2/uh2_*_s03_r1 ~/runs/uh2/uh2_*_s50 ~/runs/uh2/uh2_*_s20_r0 ~/runs/uh2/uh2_*_s80_r0; do
  [ -d "$d" ] || continue
  n=$(basename "$d")
  if [ -f "$d/backend/dsg_with_mesh.json" ]; then echo "READY $n $(stat -c %s "$d/backend/dsg_with_mesh.json") $(stat -c %y "$d/backend/dsg_with_mesh.json" | cut -c1-19)"; else echo "pending $n"; fi
done
