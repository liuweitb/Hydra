import sys, glob, os
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
sys.path.insert(0, "D:/Hydra/layer_attribution/src")
from layer_attribution.queries import SceneView, node_name
V = "D:/hydra_work/results/query_design/views/"
ref = SceneView.load(V + "uh2_clean.npz")
runs = sorted(os.path.basename(p)[:-4] for p in glob.glob(V + "*.npz"))

H = 0.5
allxy = ref.place_pos[:, :2]
lo, hi = allxy.min(0) - 6, allxy.max(0) + 6
gx = np.arange(lo[0], hi[0] + H, H); gy = np.arange(lo[1], hi[1] + H, H)
cells = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)

def label_map(v, pad=0.0, min_r=0.5):
    """room row per grid cell (-1 none): room of the nearest place when the cell lies within place radius (+pad)."""
    kd = cKDTree(v.place_pos[:, :2])
    d, i = kd.query(cells)
    r = np.maximum(v.place_radius[i].astype(float), min_r) + pad
    lab = np.where(d <= r, v.place_room[i], -1)
    return lab

Lref = label_map(ref)
nref = np.array([(Lref == i).sum() for i in range(ref.n_room)])
print("ref room cells", nref, "total covered", (Lref >= 0).sum(), "of", len(cells))

def hung_centroid(ref, run, gate=3.0):
    d = np.linalg.norm(ref.room_pos[:, None, :] - run.room_pos[None, :, :], axis=2)
    cost = np.where(d <= gate, d, 1e6)
    r, c = linear_sum_assignment(cost)
    return {int(i): int(j) for i, j in zip(r, c) if d[i, j] <= gate}

np.set_printoptions(linewidth=200, precision=2, suppress=True)
for name in runs:
    if name == "uh2_clean": continue
    run = SceneView.load(V + name + ".npz")
    Lrun = label_map(run)
    N = np.zeros((ref.n_room, run.n_room))
    m = (Lref >= 0) & (Lrun >= 0)
    np.add.at(N, (Lref[m], Lrun[m]), 1)
    frac = N / nref[:, None]   # share of ref room covered by run room j
    cover = np.array([((Lref == i) & (Lrun >= 0)).sum() / nref[i] for i in range(ref.n_room)])
    hc = hung_centroid(ref, run)
    jstar = frac.argmax(1) if run.n_room else np.full(ref.n_room, -1)
    out = []
    for i in range(ref.n_room):
        s = f"R{i}: cov {cover[i]:.2f} best j={jstar[i]} share {frac[i].max() if run.n_room else 0:.2f} | cent->{hc.get(i)}"
        out.append(s)
    print("=" * 30, name, "rooms", run.n_room)
    print(" ; ".join(out))
