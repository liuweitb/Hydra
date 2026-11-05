#!/usr/bin/env bash
# usage: wsl_viewer.sh RUN_NAME PORT   (serves ~/runs/uh2/RUN_NAME/backend/dsg_with_mesh.json with viser)
source ~/hydra_env.sh 2>/dev/null
RUN=${1:-uh2_clean}; PORT=${2:-8080}
mkdir -p ~/viewer
if [ -f ~/viewer/viser_$PORT.pid ]; then kill "$(cat ~/viewer/viser_$PORT.pid)" 2>/dev/null; fi
[ -f ~/viewer/viser.pid ] && [ "$PORT" = 8080 ] && kill "$(cat ~/viewer/viser.pid)" 2>/dev/null
sleep 1
nohup python -m spark_dsg visualize ~/runs/uh2/$RUN/backend/dsg_with_mesh.json --ip 0.0.0.0 --port $PORT > ~/viewer/viser_$PORT.log 2>&1 &
echo $! > ~/viewer/viser_$PORT.pid
sleep 8
tail -4 ~/viewer/viser_$PORT.log
