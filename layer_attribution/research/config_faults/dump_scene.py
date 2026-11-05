#!/usr/bin/env python3
"""Cache what the room/query emulation needs from a Hydra run: places (position, distance, edges), reference rooms, objects with parent place.

usage: dump_scene.py OUT_DIR RUN_DIR [RUN_DIR ...]      writes OUT_DIR/scene_<run name>.json
"""

import json
import pathlib
import sys

import numpy as np
import spark_dsg as dsg


def pos(n):
    return [float(x) for x in np.asarray(n.attributes.position, dtype=float).reshape(3)]


def dump(run):
    g = dsg.DynamicSceneGraph.load(str(pathlib.Path(run) / "backend" / "dsg_with_mesh.json"))
    places = g.get_layer(dsg.DsgLayers.PLACES)
    pl = {n.id.value: {"pos": pos(n), "dist": float(n.attributes.distance), "parent": n.get_parent() if n.has_parent() else None}
          for n in places.nodes}
    edges = [(e.source, e.target, float(e.info.weight)) for e in places.edges if e.source in pl and e.target in pl]
    rooms = {n.id.value: {"pos": pos(n), "n_children": len(list(n.children()))} for n in g.get_layer(dsg.DsgLayers.ROOMS).nodes}
    objs = []
    for n in g.get_layer(dsg.DsgLayers.OBJECTS).nodes:
        objs.append({"id": n.id.value, "label": int(n.attributes.semantic_label), "pos": pos(n),
                     "parent": n.get_parent() if n.has_parent() else None})
    return {"run": str(run), "places": pl, "edges": edges, "rooms": rooms, "objects": objs}


out = pathlib.Path(sys.argv[1])
for run in sys.argv[2:]:
    s = dump(run)
    (out / f"scene_{pathlib.Path(run).name}.json").write_text(json.dumps(s))
    print(run, len(s["places"]), "places", len(s["rooms"]), "rooms", len(s["objects"]), "objects", flush=True)
