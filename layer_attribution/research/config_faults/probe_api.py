#!/usr/bin/env python3
"""List spark_dsg attributes needed for exact per-layer comparison."""

import pathlib
import sys

import numpy as np
import spark_dsg as dsg

run = pathlib.Path(sys.argv[1])
g = dsg.DynamicSceneGraph.load(str(run / "backend" / "dsg_with_mesh.json"))
print("mesh attrs:", [x for x in dir(g.mesh) if not x.startswith("_")])
v = np.asarray(g.mesh.get_vertices())
print("vertices shape", v.shape, v.dtype)
for name in ("get_labels", "get_faces", "get_first_seen_stamps", "get_colors"):
    if hasattr(g.mesh, name):
        a = np.asarray(getattr(g.mesh, name)())
        print(name, a.shape, a.dtype)
for layer in ("OBJECTS", "PLACES", "ROOMS", "BUILDINGS"):
    nodes = list(g.get_layer(getattr(dsg.DsgLayers, layer)).nodes)
    print(layer, len(nodes))
    if nodes:
        n = nodes[0]
        print("  node attrs:", [x for x in dir(n) if not x.startswith("_")])
        print("  attrs:", [x for x in dir(n.attributes) if not x.startswith("_")])
        print("  id", n.id, type(n.id), n.id.value, "has_parent", n.has_parent(), "children", len(list(n.children())))
print("layers keys:", [x for x in dir(dsg.DsgLayers) if not x.startswith("_")])
print("partitions? PLACES/MESH_PLACES:", [x for x in dir(dsg.DsgLayers) if "PLACE" in x])
try:
    mp = list(g.get_layer(dsg.DsgLayers.MESH_PLACES).nodes)
    print("MESH_PLACES via get_layer:", len(mp))
except Exception as e:
    print("MESH_PLACES get_layer failed:", e)
print("num_nodes", g.num_nodes(), "num_edges", g.num_edges())
