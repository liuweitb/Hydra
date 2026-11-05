"""Check the two SceneView builders against each other and against the DSG API on uh2_clean (WSL, spark_dsg).

view_from_dsg is exact; view_from_snapshot (prototype Snapshot) must agree on everything except the second parent place.
"""
import pathlib
import sys
import tempfile

import numpy as np
import spark_dsg as dsg

sys.path.insert(0, "/mnt/d/Hydra/layer_attribution/src")
sys.path.insert(0, "/mnt/d/Hydra/layer_attribution/research/dsg_fields")
import snapshot_prototype as sp  # noqa: E402
from layer_attribution.queries import SceneView  # noqa: E402
from layer_attribution.query_views import view_from_dsg, view_from_snapshot  # noqa: E402

run = sys.argv[1] if len(sys.argv) > 1 else "uh2_clean"
run_dir = sp.resolve_run(run)
src = run_dir / "backend" / "dsg_with_mesh.json"
v = view_from_dsg(src, name=run)
cp = sp.cache_path(run_dir)
snap = sp.Snapshot.load(cp) if cp.exists() else sp.Snapshot.open(run, cache_root=tempfile.mkdtemp())
w = view_from_snapshot(snap, name=run)
print(f"run {run}: dsg view objects={v.n_obj} places={v.n_place} rooms={v.n_room}; snapshot view objects={w.n_obj} places={w.n_place} rooms={w.n_room}")
for k in ("obj_id", "obj_pos", "obj_label", "place_id", "place_pos", "place_radius", "place_room", "room_id", "room_pos"):
    a, b = getattr(v, k), getattr(w, k)
    print(f"  {k:13s} equal={np.array_equal(a, b)}  shape={a.shape}")
first = np.array([int(v.parents(i)[0]) if len(v.parents(i)) else -1 for i in range(v.n_obj)])
print("  first parent place equal:", np.array_equal(first, w.obj_par_idx if len(w.obj_par_idx) == v.n_obj else -2))
two = [i for i in range(v.n_obj) if len(v.parents(i)) > 1]
print("  objects with two parent places in the dsg view:", [(int(v.obj_id[i]), [int(v.place_id[p]) for p in v.parents(i)]) for i in two])
print("  snapshot adapter keeps one parent for them:", [len(w.parents(i)) for i in two])

# has_parent()/get_parent() semantics of the DSG API (ad-hoc analyze_runs.py walks them)
g = dsg.DynamicSceneGraph.load(str(src))
objs = sorted(g.get_layer(dsg.DsgLayers.OBJECTS).nodes, key=lambda n: int(n.id.value))
hp = np.array([n.has_parent() for n in objs])
npar = np.array([len(n.parents()) for n in objs])
print(f"  has_parent() true for {hp.sum()} of {len(objs)} objects; parents()-count histogram {dict(zip(*np.unique(npar, return_counts=True)))}")
print("  has_parent() is True exactly when there is one parent:", bool(np.array_equal(hp, npar == 1)))
