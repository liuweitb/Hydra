import sys, glob, os
import numpy as np
sys.path.insert(0, "D:/Hydra/layer_attribution/src")
from layer_attribution.queries import *
from layer_attribution.queries import _grid, room_label_map
V = "D:/hydra_work/results/query_design/views/"
ref = SceneView.load(V + "uh2_clean.npz")
cfg = QueryConfig()
runs = [os.path.basename(p)[:-4] for p in sorted(glob.glob(V + "*.npz"))]
runs = [r for r in runs if r not in ("uh2_clean", "uh2_clean_b") and "label_" not in r]
allv = []
bins = [0.0001, .05, .1, .15, .2, .25, .3, .35, .4, .5, .6, .7, .8, .9, 1.01]
for name in runs:
    run = SceneView.load(V + name + ".npz")
    cells = _grid(ref, run, cfg)
    lref, lrun = room_label_map(ref, cells, cfg), room_label_map(run, cells, cfg)
    nr, nn = ref.n_room, run.n_room
    sz_ref = np.bincount(lref[lref >= 0], minlength=nr); sz_run = np.bincount(lrun[lrun >= 0], minlength=nn)
    both = (lref >= 0) & (lrun >= 0)
    N = np.bincount(lref[both] * nn + lrun[both], minlength=nr * nn).reshape(nr, nn)
    for i in range(nr):
        for j in range(nn):
            if N[i, j] > 0:
                allv.append((name, i, j, N[i, j] / sz_ref[i], N[i, j] / sz_run[j], N[i, j] / min(sz_ref[i], sz_run[j]), N[i, j]))
a = np.array([[v[3], v[4], v[5], v[6]] for v in allv])
print("pairs with any overlap:", len(a))
print("hist of share_ref (N/|ref|)   :", np.histogram(a[:, 0], bins=bins)[0].tolist())
print("hist of share_run (N/|run|)   :", np.histogram(a[:, 1], bins=bins)[0].tolist())
print("hist of overlap coeff (N/min) :", np.histogram(a[:, 2], bins=bins)[0].tolist())
print("bins:", bins)
# list pairs with ambiguous values: oc in [0.2, 0.7]
print("pairs with overlap coeff in [0.15,0.7]:")
for v in allv:
    if 0.15 <= v[5] <= 0.7:
        print(f"  {v[0]:28s} ref{v[1]} run{v[2]} share_ref={v[3]:.2f} share_run={v[4]:.2f} oc={v[5]:.2f} cells={v[6]}")

print("\npairs with oc >= 0.3 but share_ref < 0.15 (slivers / tiny fragments):")
k = 0
for v in allv:
    if v[5] >= 0.3 and v[3] < 0.15:
        k += 1
        print(f"  {v[0]:28s} ref{v[1]} run{v[2]} share_ref={v[3]:.3f} share_run={v[4]:.2f} oc={v[5]:.2f} cells={v[6]}")
print("count", k)
print("\nall pairs with oc>=0.3 : share_ref sorted ascending (first 25):", sorted(round(v[3], 2) for v in allv if v[5] >= 0.3)[:25])
