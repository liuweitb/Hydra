#!/usr/bin/env python3
"""Per-layer comparison of truncated config-fault runs against a truncated clean reference.

Exact identity per layer (Hydra is deterministic) plus the tolerance metrics of D:/hydra_work/tools/analyze_runs.py
(object/place/room matching and the "object of class c in room R" query test). Results are merged into OUT_JSON.

usage: analyze_cfg_runs.py RUNS_ROOT CFG_ROOT OUT_JSON REF_NAME NAME [NAME ...] [--redo]
"""

import hashlib
import json
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, "/mnt/d/hydra_work/tools")
import analyze_runs as ar  # noqa: E402
from layer_attribution.pipeline import config_faults as cf  # noqa: E402

LAYERS = ["OBJECTS", "PLACES", "MESH_PLACES", "ROOMS", "BUILDINGS"]


def node_rows(g, layer, with_parent):
    rows = []
    for n in ar.nodes(g, layer):
        a = n.attributes
        row = [*ar.pos(n), float(ar.label(n))]
        if layer == "OBJECTS":
            row.append(float(len(a.mesh_connections)))
        if layer == "PLACES":
            row.append(float(a.distance))
        if with_parent:
            if n.has_parent():
                row += [*ar.pos(g.get_node(n.get_parent()))]
            else:
                row += [np.nan] * 3
        rows.append(row)
    if not rows:
        return np.zeros((0, 0))
    arr = np.array(rows, dtype=float)
    return arr[np.lexsort(arr.T[::-1])]


def same(a, b):
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def mesh_identical(ref, run):
    if ref.mesh.num_vertices() != run.mesh.num_vertices() or ref.mesh.num_faces() != run.mesh.num_faces():
        return False
    return same(np.asarray(ref.mesh.get_vertices()), np.asarray(run.mesh.get_vertices())) and same(
        np.asarray(ref.mesh.get_labels()), np.asarray(run.mesh.get_labels())
    )


def room_sizes(g):
    return sorted((len(list(r.children())) for r in ar.nodes(g, "ROOMS")), reverse=True)


def log_stats(info_log):
    """Last room detection and error lines of a run."""
    st = {"n_error_lines": 0, "errors": [], "n_no_rooms": 0}
    rx_nodes = re.compile(r"Detecting rooms for (\d+) nodes")
    rx_bounds = re.compile(r"Bounds: \[([\d.e+-]+), ([\d.e+-]+)\],\s+window: \[(\d+), (\d+)\]")
    rx_best = re.compile(r"Best threshold: ([\d.e+-]+) \((\d+) components\)")
    n_steps = 0
    with open(info_log, errors="replace") as f:
        for line in f:
            if "Got input @" in line:
                n_steps += 1
            elif line.startswith("E2"):
                st["n_error_lines"] += 1
                msg = line.split("]", 1)[-1].strip()[:140]
                if msg not in st["errors"] and len(st["errors"]) < 5:
                    st["errors"].append(msg)
            elif "No rooms found" in line:
                st["n_no_rooms"] += 1
                st["last_best"] = None
            else:
                m = rx_nodes.search(line)
                if m:
                    st["last_places_for_rooms"] = int(m.group(1))
                    continue
                m = rx_bounds.search(line)
                if m:
                    st["last_window_idx"] = [int(m.group(3)), int(m.group(4))]
                    continue
                m = rx_best.search(line)
                if m:
                    st["last_best"] = {"threshold_m": float(m.group(1)), "components": int(m.group(2))}
    st["steps_logged"] = n_steps
    return st


def read_status(path):
    out = {}
    p = pathlib.Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            m = re.match(r"\S+ (\S+) rc=(\d+) seconds=(\d+)", line)
            if m:
                out[m.group(1)] = {"rc": int(m.group(2)), "seconds": int(m.group(3))}
    return out


def compare(ref, run):
    ident, counts = {}, {}
    ident["mesh"] = mesh_identical(ref, run)
    counts["mesh_vertices"] = [int(ref.mesh.num_vertices()), int(run.mesh.num_vertices())]
    for layer in LAYERS:
        a_p, b_p = node_rows(ref, layer, True), node_rows(run, layer, True)
        a_n, b_n = node_rows(ref, layer, False), node_rows(run, layer, False)
        ident[layer.lower()] = {"nodes_and_parent_edge": same(a_p, b_p), "nodes": same(a_n, b_n)}
        counts[layer.lower()] = [int(len(a_n)), int(len(b_n))]
    return ident, counts


def main():
    args = [a for a in sys.argv[1:] if a != "--redo"]
    redo = "--redo" in sys.argv
    runs_root, cfg_root, out_json, ref_name, names = (
        pathlib.Path(args[0]), pathlib.Path(args[1]), pathlib.Path(args[2]), args[3], args[4:])
    status = read_status(pathlib.Path.home() / "runs/logs/tmp_cfg_batch_status.txt")
    res = json.loads(out_json.read_text()) if out_json.exists() else {"runs": {}}
    ref = ar.load(runs_root / ref_name)
    res["reference"] = {
        "name": ref_name,
        "counts": {l.lower(): len(ar.nodes(ref, l)) for l in LAYERS} | {"mesh_vertices": int(ref.mesh.num_vertices())},
        "room_sizes": room_sizes(ref),
        "log": log_stats(runs_root / ref_name / "logs/hydra_python.INFO"),
        "hydra": status.get(ref_name),
    }
    for name in names:
        if name in res["runs"] and not redo:
            continue
        rd = runs_root / name
        if not (rd / "backend/dsg_with_mesh.json").exists():
            res["runs"][name] = {"hydra": status.get(name), "error": "no dsg_with_mesh.json"}
            continue
        run = ar.load(rd)
        rec = {"hydra": status.get(name)}
        man_path = cfg_root / name / cf.MANIFEST
        if man_path.exists():
            man = json.loads(man_path.read_text())
            rec.update(fault=man["fault"], severity=man["severity"], expected_first_layer=man["expected_first_layer"],
                       params=man["params"])
            expected = {k: v["value"] for k, v in man["params"].items()}
            chk = cf.check_effective(rd / "logs/hydra_python.INFO", expected)
            rec["effective"] = {k: {"expected": expected[k], "echoed": t, "ok": ok} for k, (t, ok) in chk.items()}
        hc = (rd / "hydra_config.yaml").read_text()
        rec["hydra_config_yaml_has_frontend_or_backend_keys"] = bool(re.search(r"^(frontend|backend):", hc, re.M))
        rec["identical"], rec["counts"] = compare(ref, run)
        rec["room_sizes"] = room_sizes(run)
        rec["log"] = log_stats(rd / "logs/hydra_python.INFO")
        rec["metrics"] = ar.metrics(ref, run)
        res["runs"][name] = rec
        out_json.write_text(json.dumps(res, indent=1))
        print(name, "done", flush=True)
    out_json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
