import sys, pathlib
import numpy as np
import spark_dsg as dsg

run = pathlib.Path(sys.argv[1])
g = dsg.DynamicSceneGraph.load(str(run / "backend" / "dsg_with_mesh.json"))
print([a for a in dir(g) if not a.startswith("_")])
try:
    for l in g.layers:
        print("layer", l.id, "nodes", l.num_nodes(), "edges", l.num_edges())
except Exception as e:
    print("layers iter err", e)
try:
    for p in g.layer_partitions:
        print("partition", p)
except Exception as e:
    print("partitions err", e)
for lid, part in [(dsg.DsgLayers.PLACES, 0), (dsg.DsgLayers.MESH_PLACES, None)]:
    try:
        if part is None:
            L = g.get_layer(lid, 1) if hasattr(g, "get_layer") else None
        else:
            L = g.get_layer(lid)
        print("layer", lid, part, L.num_nodes() if L is not None else None)
    except Exception as e:
        print("err", lid, part, e)
# dynamic layers
try:
    print("dynamic layers", g.dynamic_layers if hasattr(g, "dynamic_layers") else None)
except Exception as e:
    print("dyn err", e)
m = g.mesh
for fn in ("timestamp", "first_seen_timestamp"):
    try:
        print(fn, getattr(m, fn)(0), getattr(m, fn)(1000))
    except Exception as e:
        print(fn, "err", e)
# object connections sanity
objs = list(g.get_layer(dsg.DsgLayers.OBJECTS).nodes)
V = np.asarray(m.get_vertices()).T[:, :3]
L = np.asarray(m.get_labels())
for o in objs[:5]:
    a = o.attributes
    idx = np.asarray(a.mesh_connections, dtype=int)
    print("obj", o.id, "label", a.semantic_label, "nconn", len(idx), "max idx", idx.max() if len(idx) else None,
          "mean label of conn", np.mean(L[idx] == a.semantic_label) if len(idx) else None,
          "conn centroid", V[idx].mean(0) if len(idx) else None, "pos", a.position)
# place attrs
places = list(g.get_layer(dsg.DsgLayers.PLACES).nodes)
rp = [bool(p.attributes.real_place) for p in places[:2000]]
print("real_place frac", np.mean(rp), "active frontier frac", np.mean([bool(p.attributes.active_frontier) for p in places]))
print("place label unique", np.unique([p.attributes.semantic_label for p in places]))
print("place mesh_vertex_labels sample", places[0].attributes.mesh_vertex_labels, "num_basis", places[0].attributes.num_basis_points)
print("place voxblox_mesh_connections", len(places[0].attributes.voxblox_mesh_connections), "pcl", len(places[0].attributes.pcl_mesh_connections))
print("place dist stats", np.percentile([p.attributes.distance for p in places], [0, 5, 50, 95, 100]))
# room attrs
rooms = list(g.get_layer(dsg.DsgLayers.ROOMS).nodes)
for r in rooms:
    print("room", r.id, r.attributes.position, "children", len(r.children()), "siblings", len(r.siblings()), "parent", r.has_parent())
