import sys, glob, os
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "D:/Hydra/layer_attribution/src")
from layer_attribution.queries import SceneView
V = "D:/hydra_work/results/query_design/views/"
ref = SceneView.load(V + "uh2_clean.npz")
runs = sorted(os.path.basename(p)[:-4] for p in glob.glob(V + "*.npz") if "clean" not in os.path.basename(p) or "confusion" in p)
H = 0.5
allxy = ref.place_pos[:, :2]; lo, hi = allxy.min(0) - 8, allxy.max(0) + 8
cells = np.stack(np.meshgrid(np.arange(lo[0], hi[0] + H, H), np.arange(lo[1], hi[1] + H, H), indexing="ij"), -1).reshape(-1, 2)
def lab(v, mode):
    d, i = cKDTree(v.place_pos[:, :2]).query(cells)
    r = np.maximum(v.place_radius[i].astype(float), 0.5) if mode == "radius" else np.full(len(i), mode)
    return np.where(d <= r, v.place_room[i], -1)
for mode in ("radius", 0.75, 1.0):
    Lref = lab(ref, mode); nref = np.array([(Lref == i).sum() for i in range(ref.n_room)])
    best, second, third = [], [], []
    for name in runs:
        run = SceneView.load(V + name + ".npz")
        if run.n_room == 0: continue
        Lrun = lab(run, mode); N = np.zeros((ref.n_room, run.n_room)); m = (Lref >= 0) & (Lrun >= 0); np.add.at(N, (Lref[m], Lrun[m]), 1)
        sh = np.sort(N / nref[:, None], axis=1)[:, ::-1]
        sh = np.pad(sh, ((0, 0), (0, max(0, 3 - sh.shape[1]))))
        best += list(sh[:, 0]); second += list(sh[:, 1]); third += list(sh[:, 2])
    best, second, third = map(np.array, (best, second, third))
    print(f"gate={mode}: ref cells {nref.tolist()}  (ref covered {(Lref>=0).sum()})")
    print("  best share quantiles   ", np.round(np.quantile(best, [0, .05, .25, .5, .75, 1]), 2))
    print("  second share histogram (bins 0,.05,.1,.15,.2,.25,.3,.4,.5,1):", np.histogram(second, bins=[0, .05, .1, .15, .2, .25, .3, .4, .5, 1.01])[0].tolist())
    print("  third  share histogram                                    :", np.histogram(third, bins=[0, .05, .1, .15, .2, .25, .3, .4, .5, 1.01])[0].tolist())
