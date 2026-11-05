#!/usr/bin/env python3
"""Room and object emulators at the DEFAULT config against the real final graphs of dev-data runs (full length, 2077 frames).

Each run's own places layer is fed to the room emulator and compared with that run's real rooms; each run's own labeled
mesh is fed to the object emulator and compared with that run's real objects. Only dev data is read (clean, benign, seed 0,
legacy severity-0.5 runs). Results are appended to OUT_JSON per run.

usage: validate_emulators_default.py RUNS_ROOT OUT_JSON NAME [NAME ...]
"""

import json
import pathlib
import re
import sys

import numpy as np
import spark_dsg as dsg
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import emulate_objects as eo  # noqa: E402
import emulate_rooms as er  # noqa: E402

RX_BEST = re.compile(r"Best threshold: ([\d.e+-]+) \((\d+) components\)")


def last_best(info_log):
    best = None
    with open(info_log, errors="replace") as f:
        for line in f:
            if "No rooms found" in line:
                best = None
            m = RX_BEST.search(line)
            if m:
                best = (float(m.group(1)), int(m.group(2)))
    return best


def agreement(real_place_room, emu_labels):
    """Fraction of really-assigned places that land in the same room after the best one-to-one room matching."""
    rooms = sorted(set(real_place_room.values()))
    clus = sorted(set(emu_labels.values()))
    if not rooms or not clus:
        return 0.0 if rooms else None, len(real_place_room), len(emu_labels)
    ri, ci = {r: i for i, r in enumerate(rooms)}, {c: i for i, c in enumerate(clus)}
    cont = np.zeros((len(rooms), len(clus)), dtype=int)
    for p, r in real_place_room.items():
        if p in emu_labels:
            cont[ri[r], ci[emu_labels[p]]] += 1
    a, b = linear_sum_assignment(-cont)
    agree = int(cont[a, b].sum())
    only_real = sum(1 for p in real_place_room if p not in emu_labels)
    only_emu = sum(1 for p in emu_labels if p not in real_place_room)
    return agree / max(1, len(real_place_room)), only_real, only_emu


def one(root, name):
    run = root / name
    g = dsg.DynamicSceneGraph.load(str(run / "backend" / "dsg_with_mesh.json"))
    # rooms
    layer = g.get_layer(dsg.DsgLayers.PLACES)
    dist = {n.id.value: float(n.attributes.distance) for n in layer.nodes}
    edges = [(e.source, e.target, float(e.info.weight)) for e in layer.edges if e.source in dist and e.target in dist]
    real_place_room, real_sizes = {}, []
    for r in g.get_layer(dsg.DsgLayers.ROOMS).nodes:
        kids = [int(getattr(c, "value", c)) for c in r.children()]
        real_sizes.append(len(kids))
        for c in kids:
            real_place_room[c] = r.id.value
    real_sizes.sort(reverse=True)
    emu = er.find_rooms(dist, edges, return_labels=True)
    rb = last_best(run / "logs" / "hydra_python.INFO")
    agree, only_real, only_emu = agreement(real_place_room, emu.pop("labels", {}))
    same_n = emu["n_rooms"] == len(real_sizes)
    rooms = {
        "n_places": len(dist), "n_edges": len(edges),
        "real": {"n_rooms": len(real_sizes), "sizes": real_sizes, "threshold": rb[0] if rb else None, "components": rb[1] if rb else None},
        "emulated": emu,
        "n_rooms_equal": same_n,
        "threshold_equal": (rb is not None and emu["threshold"] is not None and abs(rb[0] - emu["threshold"]) < 1e-4) or (rb is None and emu["threshold"] is None),
        "size_abs_diff_max": max(abs(a - b) for a, b in zip(emu["sizes"], real_sizes)) if same_n and real_sizes else None,
        "size_abs_diff_sum": sum(abs(a - b) for a, b in zip(emu["sizes"], real_sizes)) if same_n else None,
        "place_assignment_agreement": None if agree is None else round(agree, 4),
        "places_assigned_only_in_real": only_real, "places_assigned_only_in_emulation": only_emu,
    }
    # objects
    v = np.asarray(g.mesh.get_vertices(), dtype=float).T[:, :3]
    lab = np.asarray(g.mesh.get_labels()).astype(int)
    objs = np.array([(int(n.attributes.semantic_label), *[float(x) for x in np.asarray(n.attributes.position).reshape(3)])
                     for n in g.get_layer(dsg.DsgLayers.OBJECTS).nodes], dtype=float).reshape(-1, 4)
    r = eo.run_for(v, lab, 0.25, 40)
    matched = 0
    for l, (c, _) in r.items():
        ro = objs[objs[:, 0] == l][:, 1:]
        if len(ro) and len(c):
            d = np.linalg.norm(ro[:, None, :] - c[None, :, :], axis=2)
            rr, cc = linear_sum_assignment(np.where(d <= 1.0, d, 1e6))
            matched += int(sum(d[i, j] <= 1.0 for i, j in zip(rr, cc)))
    objects = {
        "real_n": int(len(objs)), "emulated_n": int(sum(len(s) for _, s in r.values())),
        "real_by_label": {int(l): int((objs[:, 0] == l).sum()) for l in eo.OBJECT_LABELS},
        "emulated_by_label": {int(l): int(len(s)) for l, (_, s) in r.items()},
        "real_found_one_to_one_by_emulated": round(matched / max(1, len(objs)), 3),
    }
    return {"rooms": rooms, "objects": objects}


def main():
    root, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    res = json.loads(out.read_text()) if out.exists() else {}
    for name in sys.argv[3:]:
        if name in res:
            continue
        res[name] = one(root, name)
        out.write_text(json.dumps(res, indent=1))
        r = res[name]["rooms"]
        o = res[name]["objects"]
        print(name, "| rooms real", r["real"]["n_rooms"], "emu", r["emulated"]["n_rooms"], "thr_eq", r["threshold_equal"],
              "assign", r["place_assignment_agreement"], "| objects real", o["real_n"], "emu", o["emulated_n"], "found", o["real_found_one_to_one_by_emulated"], flush=True)


if __name__ == "__main__":
    main()
