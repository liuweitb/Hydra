"""Exploration of the clean reference graph for query design: objects, parents, rooms (WSL, spark_dsg)."""
import sys, json, pathlib
import numpy as np
import spark_dsg as dsg

run = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "~/runs/uh2/uh2_clean").expanduser()
g = dsg.DynamicSceneGraph.load(str(run / "backend" / "dsg_with_mesh.json"))
O = sorted(g.get_layer(dsg.DsgLayers.OBJECTS).nodes, key=lambda n: int(n.id.value))
P = sorted(g.get_layer(dsg.DsgLayers.PLACES).nodes, key=lambda n: int(n.id.value))
R = sorted(g.get_layer(dsg.DsgLayers.ROOMS).nodes, key=lambda n: int(n.id.value))
n0 = O[0]
print("id type", type(n0.id), "value", n0.id.value, "category", n0.id.category, "category_id", getattr(n0.id, "category_id", None), "str", str(n0.id))
print("parents()", type(n0.parents()), list(n0.parents())[:3], "children()", list(P[0].children())[:3])
names = {int(k): v for k, v in dict(g.get_labelspace(2, 0).labels_to_names).items()}
print("label names", names)
pid = {int(n.id.value): i for i, n in enumerate(P)}
rid = {int(n.id.value): i for i, n in enumerate(R)}
place_room = {}
for n in P:
    ps = list(n.parents())
    place_room[int(n.id.value)] = [rid[p] for p in ps if p in rid]
print("places with !=1 room parent:", sum(len(v) != 1 for v in place_room.values()))
print("rooms:")
for i, r in enumerate(R):
    ch = [int(c) for c in r.children()]
    print(i, str(r.id), np.round(r.attributes.position, 2), "n_children", len(ch))
print("objects:")
for i, n in enumerate(O):
    pars = sorted(int(p) for p in n.parents())
    rooms = sorted({rr for p in pars for rr in place_room.get(p, [])})
    bb = n.attributes.bounding_box
    print(i, str(n.id), names.get(n.attributes.semantic_label), np.round(n.attributes.position, 2), "bbox_dim", np.round(bb.dimensions, 2) if bb.type.name != "INVALID" else None,
          "parents", [str(dsg.NodeSymbol(p)) for p in pars], "rooms", rooms)
