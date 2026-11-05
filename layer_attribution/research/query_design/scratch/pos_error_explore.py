import sys, glob, os, collections
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "D:/Hydra/layer_attribution/src")
from layer_attribution.queries import SceneView, node_name
V = "D:/hydra_work/results/query_design/views/"
ref = SceneView.load(V + "uh2_clean.npz")
runs = sorted(os.path.basename(p)[:-4] for p in glob.glob(V + "*.npz"))
np.set_printoptions(linewidth=200, precision=2, suppress=True)
Q = [0, .25, .5, .75, .9, .95, .99, 1]
print("quantiles", Q)
rows = {}
for name in runs:
    run = SceneView.load(V + name + ".npz")
    same = np.full(ref.n_obj, np.inf); anyc = np.full(ref.n_obj, np.inf)
    kd_all = cKDTree(run.obj_pos)
    anyc = kd_all.query(ref.obj_pos)[0]
    for c in np.unique(ref.obj_label):
        sel = np.where(run.obj_label == c)[0]
        ri = np.where(ref.obj_label == c)[0]
        if len(sel):
            same[ri] = cKDTree(run.obj_pos[sel]).query(ref.obj_pos[ri])[0]
    rows[name] = (same, anyc)
    print(f"{name:34s} n_obj={run.n_obj:3d}  nearest same-class: q={np.round(np.quantile(same, Q),2)}  frac<=1.0: {(same<=1).mean():.2f}  <=0.5: {(same<=0.5).mean():.2f} | nearest any-class q50/q90={np.round(np.quantile(anyc,[.5,.9]),2)}")
