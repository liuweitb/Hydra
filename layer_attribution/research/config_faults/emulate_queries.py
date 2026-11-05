#!/usr/bin/env python3
"""Room-fault emulation down to the query test: rooms from the room-finder emulator, room matching and the
"object of class c in room R" test exactly as in D:/hydra_work/tools/analyze_runs.py (gates: rooms 3 m, query 1 m).

Valid for faults that leave places and objects untouched (all rooms_* faults): objects keep their parent places, only the
place -> room assignment changes. Windows python, reads a scene cache written by dump_scene.py.

usage: emulate_queries.py SCENE_JSON [key=value ...]     room_finder overrides, e.g. min_room_size=100
"""

import json
import pathlib
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import emulate_rooms as er  # noqa: E402

ROOM_GATE_M, QUERY_GATE_M = 3.0, 1.0


def load_scene(path):
    s = json.loads(pathlib.Path(path).read_text())
    s["places"] = {int(k): v for k, v in s["places"].items()}
    s["rooms"] = {int(k): v for k, v in s["rooms"].items()}
    s["edges"] = [tuple(e) for e in s["edges"]]
    return s


def room_position(places, cluster):
    """hydra::getRoomPosition: centroid of the places, pushed onto the free-space sphere of the nearest place if outside all."""
    pts = np.array([places[n]["pos"] for n in cluster], dtype=float)
    c = pts.mean(axis=0)
    best_d, best_r, best_p = np.inf, 0.0, None
    for n in cluster:
        p = np.array(places[n]["pos"], dtype=float)
        d = float(np.linalg.norm(c - p))
        r = places[n]["dist"]
        if d <= r:
            return c
        if d - r < best_d:
            best_d, best_r, best_p = d - r, r, p
    u = c - best_p
    return best_p + u / np.linalg.norm(u) * best_r


def match(a, b, gate):
    if len(a) == 0 or len(b) == 0:
        return []
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    r, c = linear_sum_assignment(np.where(d <= gate, d, 1e6))
    return [(i, j) for i, j in zip(r, c) if d[i, j] <= gate]


def emulate(scene, overrides):
    places, edges = scene["places"], scene["edges"]
    dist = {n: p["dist"] for n, p in places.items()}
    res = er.find_rooms(dist, edges, return_labels=True, **overrides)
    labels = res.pop("labels", {})
    clusters = {}
    for n, c in labels.items():
        clusters.setdefault(c, []).append(n)
    run_cids = sorted(clusters)
    run_pos = np.array([room_position(places, clusters[c]) for c in run_cids]).reshape(-1, 3)
    ref_ids = sorted(scene["rooms"])
    ref_pos = np.array([scene["rooms"][i]["pos"] for i in ref_ids], dtype=float).reshape(-1, 3)
    pairs = match(ref_pos, run_pos, ROOM_GATE_M)
    room_map = {ref_ids[i]: run_cids[j] for i, j in pairs}  # ref room id -> run cluster id

    # per-place agreement: places with a reference room and an emulated room
    agree = tot = 0
    for n, p in places.items():
        if p["parent"] is None or n not in labels:
            continue
        tot += 1
        agree += room_map.get(p["parent"]) == labels[n]

    # queries
    ref_rooms = set(ref_ids)
    objs = scene["objects"]
    status = {"ok": 0, "wrong_object": 0, "target_missing": 0, "room_missing": 0}
    obj_cluster = []
    for o in objs:
        pl = places.get(o["parent"]) if o["parent"] is not None else None
        obj_cluster.append(labels.get(o["parent"]) if pl is not None else None)
    for o in objs:
        pl = places.get(o["parent"]) if o["parent"] is not None else None
        ref_room = pl["parent"] if pl is not None and pl["parent"] in ref_rooms else None
        if ref_room is None:
            continue
        run_room = room_map.get(ref_room)
        if run_room is None:
            status["room_missing"] += 1
            continue
        op = np.array(o["pos"], dtype=float)
        cands = [np.linalg.norm(np.array(q["pos"], dtype=float) - op) for q, c in zip(objs, obj_cluster)
                 if q["label"] == o["label"] and c == run_room]
        if not cands:
            status["target_missing"] += 1
        elif min(cands) <= QUERY_GATE_M:
            status["ok"] += 1
        else:
            status["wrong_object"] += 1
    nq = sum(status.values())
    return {"n_rooms": res["n_rooms"], "sizes": res["sizes"], "threshold": res["threshold"],
            "rooms_matched": round(len(pairs) / max(1, len(ref_ids)), 3),
            "room_assignment_agreement": round(agree / tot, 3) if tot else None,
            "queries": nq, "query_success": round(status["ok"] / nq, 3) if nq else None, "query_status": status}


def main():
    scene = load_scene(sys.argv[1])
    over = {}
    for kv in sys.argv[2:]:
        k, v = kv.split("=")
        over[k] = int(v) if k in ("min_room_size", "min_component_size") else float(v)
    print(json.dumps(emulate(scene, over)))


if __name__ == "__main__":
    main()
