#!/usr/bin/env python3
"""Object survival under frontend.objects.min_cluster_size, predicted from a clean run's object sizes.

Prediction = objects whose final mesh_connections count is >= threshold. Checked against the real truncated
min_size runs (same objects, matched by label and position), then tabulated for the full-length clean run.

usage: objects_size_emulation.py RUNS_ROOT TRUNC_REF [FULL_CLEAN] OUT_JSON [REAL_RUN:THRESHOLD ...]
"""

import json
import pathlib
import sys

import numpy as np
import spark_dsg as dsg

NAMES = {5: "chair", 6: "bench", 7: "couch", 10: "computer", 12: "painting", 13: "plant", 18: "trashcan"}
THRESHOLDS = [40, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 600, 800, 1000, 1500, 2000, 3000]


def objects(run):
    g = dsg.DynamicSceneGraph.load(str(pathlib.Path(run) / "backend" / "dsg_with_mesh.json"))
    rows = []
    for n in g.get_layer(dsg.DsgLayers.OBJECTS).nodes:
        a = n.attributes
        rows.append((int(a.semantic_label), len(a.mesh_connections), *[float(x) for x in np.asarray(a.position).reshape(3)]))
    return np.array(rows, dtype=float).reshape(-1, 5)


def table(rows):
    out = {}
    for t in THRESHOLDS:
        keep = rows[:, 1] >= t
        out[t] = {"n": int(keep.sum()), "by_class": {NAMES.get(int(c), str(int(c))): int((keep & (rows[:, 0] == c)).sum())
                                                     for c in sorted(set(rows[:, 0]))}}
    return out


def main():
    root = pathlib.Path(sys.argv[1])
    args = sys.argv[2:]
    real = [a for a in args if ":" in a]
    args = [a for a in args if ":" not in a]
    trunc_ref, out_json = args[0], pathlib.Path(args[-1])
    full = args[1] if len(args) == 3 else None
    res = {}
    ref = objects(root / trunc_ref)
    res["trunc_ref"] = {"n_objects": len(ref), "sizes_sorted": sorted(int(s) for s in ref[:, 1]), "survivors_by_threshold": table(ref),
                        "class_counts": {NAMES.get(int(c), str(int(c))): int((ref[:, 0] == c).sum()) for c in sorted(set(ref[:, 0]))}}
    res["real_vs_predicted"] = {}
    for item in real:
        name, thr = item.rsplit(":", 1)
        thr = int(thr)
        run = objects(root / name)
        pred = ref[ref[:, 1] >= thr]
        # match real survivors to predicted survivors by label + position (0.5 m)
        used, hit = set(), 0
        for r in run:
            d = [(np.linalg.norm(r[2:] - p[2:]), j) for j, p in enumerate(pred) if p[0] == r[0] and j not in used]
            if d and min(d)[0] <= 0.5:
                used.add(min(d)[1])
                hit += 1
        res["real_vs_predicted"][name] = {"threshold": thr, "real_n": len(run), "predicted_n": len(pred), "real_matched_to_predicted": hit,
                                          "real_sizes_sorted": sorted(int(s) for s in run[:, 1]),
                                          "predicted_sizes_sorted": sorted(int(s) for s in pred[:, 1])}
    if full:
        fr = objects(root / full)
        res["full_clean"] = {"run": full, "n_objects": len(fr), "sizes_sorted": sorted(int(s) for s in fr[:, 1]),
                             "survivors_by_threshold": table(fr),
                             "class_counts": {NAMES.get(int(c), str(int(c))): int((fr[:, 0] == c).sum()) for c in sorted(set(fr[:, 0]))}}
    out_json.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "full_clean"}, indent=1)[:3500])
    if full:
        print("full clean survivors:", {t: v["n"] for t, v in res["full_clean"]["survivors_by_threshold"].items()})


if __name__ == "__main__":
    main()
