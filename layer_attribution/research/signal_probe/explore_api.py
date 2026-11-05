import sys, json, pathlib
import numpy as np
import spark_dsg as dsg

run = pathlib.Path(sys.argv[1])
g = dsg.DynamicSceneGraph.load(str(run / "backend" / "dsg_with_mesh.json"))
print("spark_dsg", getattr(dsg, "__version__", "?"), "file", dsg.__file__)
print("layers:", [(k, v) for k, v in vars(dsg.DsgLayers).items() if not k.startswith("_")])
print("num_nodes", g.num_nodes(), "num_edges", g.num_edges(), "has_mesh", g.has_mesh())
m = g.mesh
print("mesh type", type(m), [a for a in dir(m) if not a.startswith("_")])
print("num_vertices", m.num_vertices(), "num_faces", m.num_faces())
v = m.get_vertices(); print("vertices shape", np.asarray(v).shape, np.asarray(v).dtype)
try:
    l = m.get_labels(); print("labels shape", np.asarray(l).shape, np.asarray(l).dtype, np.unique(np.asarray(l))[:30])
except Exception as e:
    print("labels err", e)
try:
    f = m.get_faces(); print("faces shape", np.asarray(f).shape)
except Exception as e:
    print("faces err", e)
for name in ["SEGMENTS", "OBJECTS", "PLACES", "ROOMS", "BUILDINGS"]:
    try:
        lid = getattr(dsg.DsgLayers, name)
    except AttributeError:
        print(name, "missing"); continue
    if not g.has_layer(lid):
        print(name, "no layer"); continue
    layer = g.get_layer(lid)
    ns = list(layer.nodes)
    print("==", name, "id", lid, "nodes", len(ns), "edges", layer.num_edges())
    if ns:
        n = ns[0]
        a = n.attributes
        print("  attr type", type(a).__name__, [x for x in dir(a) if not x.startswith("_")])
        print("  id", n.id, n.id.value, "has_parent", n.has_parent(), "children", len(n.children()), "siblings", len(n.siblings()))
        try:
            print("  position", a.position)
        except Exception as e:
            print("  pos err", e)
        for key in ["semantic_label", "name", "distance", "radius", "bounding_box", "color", "is_active", "last_update_time_ns", "need_cleanup", "mesh_connections", "num_basis_points", "has_active_mesh_indices", "semantic_feature", "observed", "is_predicted"]:
            if hasattr(a, key):
                val = getattr(a, key)
                s = repr(val)
                print("  ", key, "=", s[:200])
