#!/usr/bin/env bash
# Locate the spark_dsg package and its C++ headers (layer ids/names) inside WSL.
source ~/hydra_env.sh 2>/dev/null
python - <<'EOF'
import spark_dsg, os
print("spark_dsg file", spark_dsg.__file__)
print("version", getattr(spark_dsg, "__version__", "?"))
print("dir", os.path.dirname(spark_dsg.__file__))
print(sorted(os.listdir(os.path.dirname(spark_dsg.__file__))))
EOF
echo "--- headers"
find ~/hydra_ws ~/venvs/hydra -name 'scene_graph_types.h' -o -name 'scene_graph_layer.h' -o -name 'layer_prefix.h' 2>/dev/null | head
find / -path /proc -prune -o -name 'scene_graph_types.h' -print 2>/dev/null | head
echo "--- src dirs"
ls ~/hydra_ws 2>/dev/null
ls ~/hydra_ws/src 2>/dev/null
