import sys, glob, os, collections
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, "D:/Hydra/layer_attribution/src")
from layer_attribution.queries import *
V = "D:/hydra_work/results/query_design/views/"
ref = SceneView.load(V + "uh2_clean.npz")
qs = build_queries(ref)
# distance from each object to nearest place of ANY OTHER room (3D and xy) in the reference
pl3, pl2 = ref.place_pos, ref.place_pos[:, :2]
d3o, d2o, d3own = [], [], []
for q in qs:
    other = ref.place_room != q.ref_room_row
    d3o.append(np.linalg.norm(pl3[other] - q.pos, axis=1).min())
    d2o.append(np.linalg.norm(pl2[other] - q.pos[:2], axis=1).min())
    d3own.append(np.linalg.norm(pl3[~other] - q.pos, axis=1).min())
d3o, d2o, d3own = map(np.array, (d3o, d2o, d3own))
print("distance object -> nearest place of another room (3D) quantiles:", np.round(np.quantile(d3o, [0, .05, .1, .25, .5]), 2))
print("distance object -> nearest place of another room (xy) quantiles:", np.round(np.quantile(d2o, [0, .05, .1, .25, .5]), 2))
print("distance object -> nearest place of own room (3D) quantiles:", np.round(np.quantile(d3own, [0, .25, .5, .75, .95, 1]), 2))
# benign flips
benign = [os.path.basename(p)[:-4] for p in sorted(glob.glob(V + "*_s03_r*.npz"))]
cnt = collections.Counter(); modes = collections.defaultdict(list)
for name in benign:
    run = SceneView.load(V + name + ".npz")
    ev = evaluate(ref, run)
    for q, r in zip(ev.queries, ev.results):
        if not r.ok:
            cnt[q.qid] += 1; modes[q.qid].append(r.mode)
print("benign runs:", len(benign))
print("objects failing in >=1 benign run:", len(cnt), " total failures:", sum(cnt.values()))
print("qid obj label n_fail  modes  d_other3D d_other_xy")
for qid, c in cnt.most_common(40):
    q = qs[qid]
    print(f"{qid:3d} {node_name(q.ref_id):8s} {q.label_name:9s} room={q.ref_room_row} fails={c:2d} {dict(collections.Counter(modes[qid]))}  d_other3D={d3o[qid]:.2f} d_other_xy={d2o[qid]:.2f}")
for thr in (0.5, 0.75, 1.0, 1.5, 2.0):
    print(f"ref objects with another room's place within {thr} m (3D): {(d3o < thr).sum()}   (xy): {(d2o < thr).sum()}")
