#!/usr/bin/env python3
"""Describe the clean scene graph: object sizes, place radii, room sizes (guides severity ranges)."""

import json
import pathlib
import sys

import numpy as np
import spark_dsg as dsg

run = pathlib.Path(sys.argv[1])
g = dsg.DynamicSceneGraph.load(str(run / "backend" / "dsg_with_mesh.json"))


def layer_nodes(name):
    return list(g.get_layer(getattr(dsg.DsgLayers, name)).nodes)


out = {}
objs = layer_nodes("OBJECTS")
sizes = []
for n in objs:
    a = n.attributes
    mc = getattr(a, "mesh_connections", None)
    sizes.append(len(mc) if mc is not None else -1)
out["n_objects"] = len(objs)
out["object_labels"] = sorted({int(n.attributes.semantic_label) for n in objs})
out["object_mesh_conn_sizes_sorted"] = sorted(sizes)
out["object_attr_fields"] = [x for x in dir(objs[0].attributes) if not x.startswith("_")] if objs else []

pl = layer_nodes("PLACES")
dist = np.array([float(n.attributes.distance) for n in pl])
out["n_places"] = len(pl)
out["place_distance_quantiles"] = {str(q): round(float(np.quantile(dist, q)), 3) for q in (0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)}

rooms = layer_nodes("ROOMS")
out["n_rooms"] = len(rooms)
rs = []
for r in rooms:
    rs.append(len([c for c in r.children()]))
out["room_n_children"] = rs
out["room_positions"] = [[round(float(x), 2) for x in r.attributes.position] for r in rooms]

# parents of objects: place -> room chain
by_room = {}
for o in objs:
    n = o
    chain = []
    for _ in range(3):
        if not n.has_parent():
            break
        n = g.get_node(n.get_parent())
        chain.append(str(n.id))
    by_room.setdefault(chain[-1] if chain else "none", 0)
    by_room[chain[-1] if chain else "none"] += 1
out["objects_by_top_ancestor"] = by_room
print(json.dumps(out, indent=1))
