#!/usr/bin/env python3
"""Offline screen of frontend.objects.{cluster_tolerance,min_cluster_size}: euclidean clustering of the final labeled mesh.

Hydra clusters only the active mesh window of each update and merges clusters across updates, so this is an
approximation of the incremental result; it is checked against real truncated runs (compare mode) and used only to pick
severities. Mesh vertices are stored (6, N) = xyz + rgb.

usage: emulate_objects.py RUN_DIR OUT_JSON [--tols 0.05,0.1,...] [--cases tol:min_size,tol:min_size,...]
"""

import json
import pathlib
import sys

import numpy as np
import spark_dsg as dsg
from scipy.sparse import coo_matrix
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

OBJECT_LABELS = [5, 6, 7, 10, 12, 13, 18]
TOLS = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.4, 0.6124, 1.0, 1.5, 2.0, 3.0, 4.0]


def load(run):
    g = dsg.DynamicSceneGraph.load(str(pathlib.Path(run) / "backend" / "dsg_with_mesh.json"))
    v = np.asarray(g.mesh.get_vertices(), dtype=float).T[:, :3]
    lab = np.asarray(g.mesh.get_labels()).astype(int)
    objs = []
    for n in g.get_layer(dsg.DsgLayers.OBJECTS).nodes:
        a = n.attributes
        objs.append((int(a.semantic_label), *[float(x) for x in np.asarray(a.position).reshape(3)]))
    return v, lab, np.array(objs, dtype=float).reshape(-1, 4)


def clusters(points, tol, min_size):
    """(centroids, sizes) of euclidean clusters with >= min_size points."""
    if len(points) < min_size:
        return np.zeros((0, 3)), np.zeros(0, dtype=int)
    pairs = cKDTree(points).query_pairs(tol, output_type="ndarray")
    n = len(points)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    k, comp = connected_components(g, directed=False)
    sizes = np.bincount(comp, minlength=k)
    keep = np.flatnonzero(sizes >= min_size)
    cents = np.array([points[comp == c].mean(axis=0) for c in keep]).reshape(-1, 3)
    return cents, sizes[keep]


def run_for(v, lab, tol, min_size):
    out = {}
    for l in OBJECT_LABELS:
        c, s = clusters(v[lab == l], tol, min_size)
        out[l] = (c, s)
    return out


def main():
    run, out_json = sys.argv[1], pathlib.Path(sys.argv[2])
    cases = [(t, 40) for t in TOLS]
    if "--tols" in sys.argv:
        cases = [(float(x), 40) for x in sys.argv[sys.argv.index("--tols") + 1].split(",")]
    if "--cases" in sys.argv:  # tol:min_size,...
        cases = [(float(c.split(":")[0]), int(c.split(":")[1])) for c in sys.argv[sys.argv.index("--cases") + 1].split(",")]
    v, lab, objs = load(run)
    res = {"run": str(run), "real_objects": int(len(objs)), "labeled_vertices": {int(l): int((lab == l).sum()) for l in OBJECT_LABELS}, "by_tol": {}}
    for tol, min_size in cases:
        r = run_for(v, lab, tol, min_size)
        n = sum(len(s) for _, s in r.values())
        # share of emulated clusters whose centroid has a real object of the same label within 1 m
        near = tot = 0
        for l, (c, _) in r.items():
            ro = objs[objs[:, 0] == l][:, 1:]
            for p in c:
                tot += 1
                near += bool(len(ro)) and float(np.min(np.linalg.norm(ro - p, axis=1))) <= 1.0
        # one-to-one match (1 m gate, same label) of the run's real objects to emulated centroids: analogue of objects_found_with_label
        matched = 0
        for l, (c, _) in r.items():
            ro = objs[objs[:, 0] == l][:, 1:]
            if len(ro) and len(c):
                d = np.linalg.norm(ro[:, None, :] - c[None, :, :], axis=2)
                rr, cc = linear_sum_assignment(np.where(d <= 1.0, d, 1e6))
                matched += int(sum(d[i, j] <= 1.0 for i, j in zip(rr, cc)))
        key = f"tol{tol}_min{min_size}"
        res["by_tol"][key] = {"tol": tol, "min_size": min_size, "n_clusters": int(n),
                              "by_label": {int(l): int(len(s)) for l, (_, s) in r.items()},
                              "centroid_has_real_object_within_1m": round(near / max(1, tot), 3),
                              "real_objects_found_one_to_one": round(matched / max(1, len(objs)), 3)}
        print(key, res["by_tol"][key], flush=True)
    out_json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
