import sys
import numpy as np
sys.path.insert(0, "D:/Hydra/layer_attribution/src")
from layer_attribution.queries import *
np.set_printoptions(linewidth=220, precision=2, suppress=True)
V = "D:/hydra_work/results/query_design/views/"
ref = SceneView.load(V + "uh2_clean.npz")
name = sys.argv[1]
run = SceneView.load(V + name + ".npz")
ev = evaluate(ref, run)
m = ev.match
print(name, "rooms", run.n_room, "structures", m.structures())
print("overlap-coefficient matrix (ref rows x run cols)\n", m.oc)
print("rep rooms per ref room:", [list(m.rep_rooms(i)) for i in range(ref.n_room)])
for q, r in zip(ev.queries, ev.results):
    if not r.ok:
        n = r.elsewhere_row
        rr = run.rooms_of_object(n).tolist() if n >= 0 else None
        print(f"q{q.qid:3d} {node_name(q.ref_id):8s} {q.label_name:9s} refroom={q.ref_room_row} pos={np.round(q.pos[:2],1)} mode={r.mode:15s} rep={list(r.rep_room_rows)} elsewhere_node_rooms={rr} err_in_rooms={r.err_in_rooms_m:.2f} err_any={r.err_any_m:.2f} n_same_in_rooms={r.n_same_class_in_rooms}")
