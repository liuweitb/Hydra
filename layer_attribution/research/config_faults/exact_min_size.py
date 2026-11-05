#!/usr/bin/env python3
"""Exact outcome of rooms_min_size on a clean run: rooms are the clean rooms with >= T places, nothing else changes.

Query success = share of queries (objects whose parent place has a reference room) whose room survives. Windows python.

usage: exact_min_size.py SCENE_JSON [T ...]
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import emulate_queries as eq  # noqa: E402

scene = eq.load_scene(sys.argv[1])
sizes = {rid: r["n_children"] for rid, r in scene["rooms"].items()}
room_of_obj = []
for o in scene["objects"]:
    pl = scene["places"].get(o["parent"]) if o["parent"] is not None else None
    if pl is not None and pl["parent"] in sizes:
        room_of_obj.append(pl["parent"])
objs_per_room = {rid: room_of_obj.count(rid) for rid in sizes}
print("rooms (id: n places, n queries):", {rid: (sizes[rid], objs_per_room[rid]) for rid in sorted(sizes, key=lambda r: -sizes[r])})
print("queries:", len(room_of_obj))
thr = [int(t) for t in sys.argv[2:]] or [10, 44, 45, 53, 62, 63, 66, 67, 74, 76, 91, 110, 132, 147, 148, 159, 163, 164, 191, 195, 196, 230, 277, 317, 318, 400]
out = {}
for t in thr:
    kept = [rid for rid in sizes if sizes[rid] >= t]
    ok = sum(1 for r in room_of_obj if r in kept)
    out[t] = {"rooms_kept": len(kept), "query_success": round(ok / len(room_of_obj), 3), "kept_sizes": sorted((sizes[r] for r in kept), reverse=True)}
    print(f"T={t:>4}: rooms kept {len(kept)} sizes {out[t]['kept_sizes']} query success {out[t]['query_success']}")
if "--json" in sys.argv:
    pass
