import sys, pathlib, time
import numpy as np
import spark_dsg as dsg

t0 = time.time()
run = pathlib.Path(sys.argv[1])
g = dsg.DynamicSceneGraph.load(str(run / "backend" / "dsg_with_mesh.json"))
print("load s", round(time.time() - t0, 1))
for p in g.layer_partitions:
    try:
        print("partition", p.id, "nodes", p.num_nodes(), "edges", p.num_edges())
    except Exception as e:
        print("partition err", e, [a for a in dir(p) if not a.startswith("_")][:40])
print("layer_keys", list(g.layer_keys))
print("layer_names", g.layer_names)
try:
    mp = g.get_layer("MESH_PLACES")
    ns = list(mp.nodes)
    print("MESH_PLACES nodes", len(ns), "edges", mp.num_edges())
    if ns:
        a = ns[0].attributes
        print(type(a).__name__, [x for x in dir(a) if not x.startswith("_")])
        print("pos", a.position, "label", a.semantic_label, "parent?", ns[0].has_parent(), "children", len(ns[0].children()), "siblings", len(ns[0].siblings()))
        pos = np.array([n.attributes.position for n in ns])
        print("z pct", np.percentile(pos[:, 2], [0, 5, 50, 95, 100]))
except Exception as e:
    print("MESH_PLACES err", e)
# interlayer edges
ie = list(g.interlayer_edges)
print("interlayer edges", len(ie))
from collections import Counter
c = Counter()
for e in ie[:20000]:
    s = g.get_node(e.source); t = g.get_node(e.target)
    c[(str(s.layer), str(t.layer))] += 1
print(c)
ue = list(g.unpartitioned_edges) if hasattr(g, "unpartitioned_edges") else None
print("unpartitioned edges", None if ue is None else len(ue))
un = list(g.unpartitioned_nodes) if hasattr(g, "unpartitioned_nodes") else None
print("unpartitioned nodes", None if un is None else len(un))
# node layer attribute
n0 = next(iter(g.get_layer(dsg.DsgLayers.OBJECTS).nodes))
print("node attrs", [a for a in dir(n0) if not a.startswith("_")])
print("layer of node", n0.layer, type(n0.layer), n0.id.category if hasattr(n0.id, "category") else None)
# time stamps
m = g.mesh
ts = np.array([m.timestamp(i) for i in range(0, m.num_vertices(), 997)])
fs = np.array([m.first_seen_timestamp(i) for i in range(0, m.num_vertices(), 997)])
print("timestamp pct s", np.percentile(ts, [0, 25, 50, 75, 100]) / 1e9)
print("first_seen pct s", np.percentile(fs, [0, 25, 50, 75, 100]) / 1e9)
print("t-first_seen pct s", np.percentile(ts - fs, [0, 25, 50, 75, 100]) / 1e9)
