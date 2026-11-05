import sys, glob, os, time
sys.path.insert(0, "D:/Hydra/layer_attribution/src")
from layer_attribution.queries import *
V = "D:/hydra_work/results/query_design/views/"
ref = SceneView.load(V + "uh2_clean.npz")
runs = sorted(os.path.basename(p)[:-4] for p in glob.glob(V + "*.npz"))
t0 = time.time()
for name in runs:
    run = SceneView.load(V + name + ".npz")
    row = []
    for method in ("footprint", "centroid"):
        cfg = QueryConfig(room_method=method)
        ev = evaluate(ref, run, cfg)
        s = ev.summary(); st = ev.summary(strict=True)
        row.append(f"{method[:4]} n={s['n_queries']} fail={s['failure_rate']:.3f} strict={st['failure_rate']:.3f} {({k:v for k,v in s['modes'].items() if v})}")
    print(f"{name:32s}", " | ".join(row))
print("elapsed", round(time.time() - t0, 1), "s")
