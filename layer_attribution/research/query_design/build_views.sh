#!/usr/bin/env bash
# build compact SceneView npz files for all finished DEVELOPMENT runs (clean, clean_b, s03_r0/r1, legacy s50, s20_r0, s80_r0)
# never touches non-benign _r1/_r2 runs; 2 cores, 1 BLAS thread
source ~/hydra_env.sh 2>/dev/null
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=/mnt/d/hydra_work/results/query_design/views
runs=()
for d in ~/runs/uh2/uh2_clean ~/runs/uh2/uh2_clean_b ~/runs/uh2/uh2_*_s03_r0 ~/runs/uh2/uh2_*_s03_r1 ~/runs/uh2/uh2_*_s50 ~/runs/uh2/uh2_*_s20_r0 ~/runs/uh2/uh2_*_s80_r0; do
  [ -f "$d/backend/dsg_with_mesh.json" ] && runs+=("$(basename "$d")")
done
echo "building ${#runs[@]} views: ${runs[*]}"
PYTHONPATH=/mnt/d/Hydra/layer_attribution/src exec taskset -c 10,11 python -m layer_attribution.query_views build "${runs[@]}" --out "$OUT" "$@"
