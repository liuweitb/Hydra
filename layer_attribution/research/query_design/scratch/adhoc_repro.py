"""Re-implement the ad-hoc query test of analyze_runs.py on SceneViews to check the view builder against the recorded numbers."""
import sys, glob, os
import numpy as np
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, "D:/Hydra/layer_attribution/src")
from layer_attribution.queries import SceneView
V = "D:/hydra_work/results/query_design/views/"
ref = SceneView.load(V + "uh2_clean.npz")

def match(a, b, gate):
    if len(a) == 0 or len(b) == 0: return []
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    cost = np.where(d <= gate, d, 1e6)
    r, c = linear_sum_assignment(cost)
    return [(i, j) for i, j in zip(r, c) if d[i, j] <= gate]

def single_room(v, i):
    """has_parent() semantics: only objects with exactly one parent place resolve; place has exactly one room."""
    p = v.parents(i)
    if len(p) != 1: return None
    r = v.place_room[p[0]]
    return int(r) if r >= 0 else None

def adhoc(ref, run, gate_room=3.0, gate_q=1.0):
    pairs = match(ref.room_pos, run.room_pos, gate_room)
    room_map = {i: j for i, j in pairs}
    st = {"ok": 0, "wrong_object": 0, "target_missing": 0, "room_missing": 0}
    for o in range(ref.n_obj):
        room = single_room(ref, o)
        if room is None: continue
        rr = room_map.get(room)
        if rr is None: st["room_missing"] += 1; continue
        cands = [n for n in range(run.n_obj) if run.obj_label[n] == ref.obj_label[o] and single_room(run, n) == rr]
        if not cands: st["target_missing"] += 1
        elif min(np.linalg.norm(run.obj_pos[n] - ref.obj_pos[o]) for n in cands) <= gate_q: st["ok"] += 1
        else: st["wrong_object"] += 1
    nq = sum(st.values())
    return nq, round(st["ok"] / nq, 3), st

expected = {"label_confusion": .571, "label_dropout": .571, "depth_noise": .629, "depth_holes": .419, "pose_drift": .352, "frame_drop": .752}
for f, e in expected.items():
    run = SceneView.load(V + f"uh2_{f}_s50.npz")
    nq, s, st = adhoc(ref, run)
    print(f"{f:16s} queries={nq} success={s:.3f} (recorded {e:.3f}) {st}")
