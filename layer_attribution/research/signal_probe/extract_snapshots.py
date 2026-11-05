#!/usr/bin/env python3
"""Dump one Hydra run (backend/dsg_with_mesh.json + side files) to a compact npz for features_proto.

Runs inside WSL (needs spark_dsg). One run at a time, single process, so peak RAM stays near one graph.
usage: extract_snapshots.py OUT_DIR RUN_DIR [RUN_DIR ...]   (existing snapshots are skipped)
"""

import gc
import json
import pathlib
import sys

import numpy as np
import spark_dsg as dsg

OBJECTS, PLACES, ROOMS, BUILDINGS = (getattr(dsg.DsgLayers, k) for k in ("OBJECTS", "PLACES", "ROOMS", "BUILDINGS"))


def _id_index(nodes):
    return {n.id.value: i for i, n in enumerate(nodes)}


def _parent_index(nodes, parent_lookup):
    """Index of each node's parent in the parent layer array, -1 if no parent or parent in another layer."""
    out = np.full(len(nodes), -1, dtype=np.int32)
    for i, n in enumerate(nodes):
        if n.has_parent():
            out[i] = parent_lookup.get(n.get_parent(), -1)
    return out


def _layer_edges(layer, lookup):
    uv = [(lookup[e.source], lookup[e.target]) for e in layer.edges if e.source in lookup and e.target in lookup]
    return np.asarray(uv, dtype=np.int32).reshape(-1, 2)


def _by_value(layer):
    return {n.id.value: n for n in layer.nodes}


def snapshot(run: pathlib.Path) -> dict:
    g = dsg.DynamicSceneGraph.load(str(run / "backend" / "dsg_with_mesh.json"))
    out = {}

    m = g.mesh
    n_v = m.num_vertices()
    v = np.asarray(m.get_vertices(), dtype=np.float64)  # (6, N): xyz + rgb
    out["mesh_xyz"] = np.ascontiguousarray(v[:3].T, dtype=np.float32)
    out["mesh_label"] = np.asarray(m.get_labels(), dtype=np.int16)
    out["mesh_faces"] = np.ascontiguousarray(np.asarray(m.get_faces()).T, dtype=np.int32)
    out["mesh_t_last"] = np.fromiter((m.timestamp(i) for i in range(n_v)), dtype=np.int64, count=n_v)
    out["mesh_t_first"] = np.fromiter((m.first_seen_timestamp(i) for i in range(n_v)), dtype=np.int64, count=n_v)

    objs = list(g.get_layer(OBJECTS).nodes)
    places = list(g.get_layer(PLACES).nodes)
    rooms = list(g.get_layer(ROOMS).nodes)
    blds = list(g.get_layer(BUILDINGS).nodes) if g.has_layer(BUILDINGS) else []
    o_idx, p_idx, r_idx = _id_index(objs), _id_index(places), _id_index(rooms)

    k = len(objs)
    out["obj_pos"] = np.array([n.attributes.position for n in objs], dtype=np.float64).reshape(k, 3)
    out["obj_label"] = np.array([n.attributes.semantic_label for n in objs], dtype=np.int32)
    bb = [n.attributes.bounding_box for n in objs]
    out["obj_bbox_center"] = np.array([b.world_P_center for b in bb], dtype=np.float64).reshape(k, 3)
    out["obj_bbox_dim"] = np.array([b.dimensions for b in bb], dtype=np.float64).reshape(k, 3)
    out["obj_bbox_R"] = np.array([b.world_R_center for b in bb], dtype=np.float64).reshape(k, 3, 3)
    out["obj_parent"] = _parent_index(objs, p_idx)
    conn = [np.asarray(n.attributes.mesh_connections, dtype=np.int64) for n in objs]
    out["obj_conn_offsets"] = np.concatenate([[0], np.cumsum([len(c) for c in conn])]).astype(np.int64)
    out["obj_conn_vertices"] = np.concatenate(conn) if conn else np.zeros(0, dtype=np.int64)
    out["obj_last_update_ns"] = np.array([n.attributes.last_update_time_ns for n in objs], dtype=np.int64)

    k = len(places)
    out["place_pos"] = np.array([n.attributes.position for n in places], dtype=np.float64).reshape(k, 3)
    out["place_radius"] = np.array([n.attributes.distance for n in places], dtype=np.float64)
    out["place_basis"] = np.array([n.attributes.num_basis_points for n in places], dtype=np.int32)
    out["place_parent"] = _parent_index(places, r_idx)
    out["place_edges"] = _layer_edges(g.get_layer(PLACES), p_idx)
    out["place_last_update_ns"] = np.array([n.attributes.last_update_time_ns for n in places], dtype=np.int64)
    obj_by_val = _by_value(g.get_layer(OBJECTS))
    out["place_n_obj_children"] = np.array(
        [sum(1 for c in n.children() if c in obj_by_val) for n in places], dtype=np.int32)

    k = len(rooms)
    out["room_pos"] = np.array([n.attributes.position for n in rooms], dtype=np.float64).reshape(k, 3)
    out["room_has_parent"] = np.array([n.has_parent() for n in rooms], dtype=bool)
    out["room_edges"] = _layer_edges(g.get_layer(ROOMS), r_idx)
    out["building_pos"] = np.array([n.attributes.position for n in blds], dtype=np.float64).reshape(-1, 3)

    if g.has_layer("MESH_PLACES") if hasattr(g, "has_layer") else False:
        mp = list(g.get_layer("MESH_PLACES").nodes)
    else:
        mp = []
    out["meshplace_pos"] = np.array([n.attributes.position for n in mp], dtype=np.float64).reshape(-1, 3)

    out["graph_counts"] = np.array([g.num_nodes(), g.num_edges()], dtype=np.int64)
    del g, m
    gc.collect()

    traj = run / "backend" / "trajectory.csv"
    out["traj"] = np.loadtxt(traj, delimiter=",", skiprows=1, ndmin=2) if traj.exists() else np.zeros((0, 8))

    for layer in (2, 3, 4):  # cumulative churn counters, last row: active, removed, merged, w_parents, w_children, edges
        f = run / "backend" / f"layer_{layer}_statistics.csv"
        rows = np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2) if f.exists() else np.zeros((0, 6))
        out[f"stats_layer{layer}_last"] = rows[-1] if len(rows) else np.full(6, np.nan)

    man = run / "injection_manifest.json"
    meta = {"run": run.name}
    if man.exists():
        full = json.loads(man.read_text())
        meta["manifest"] = {k2: full.get(k2) for k2 in (
            "fault", "severity", "seed", "params", "expected_first_layer", "also_hits_directly", "corrupted_input",
            "n_frames_in", "n_frames_out")}
    out["meta_json"] = np.array(json.dumps(meta))
    return out


def is_development(run: pathlib.Path) -> bool:
    """Seed 0 or benign (severity <= 0.05) from the manifest copy; clean runs have no manifest. Held-out guard."""
    man = run / "injection_manifest.json"
    if not man.exists():
        return run.name in ("uh2_clean", "uh2_clean_b")
    m = json.loads(man.read_text())
    return int(m.get("seed", -1)) == 0 or float(m.get("severity", 1.0)) <= 0.05


def main():
    out_dir = pathlib.Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in sys.argv[2:]:
        run = pathlib.Path(r)
        dst = out_dir / f"{run.name}.npz"
        if dst.exists():
            print("skip", run.name, flush=True)
            continue
        if not is_development(run):
            print("held-out, refused", run.name, flush=True)
            continue
        if not (run / "backend" / "dsg_with_mesh.json").exists():
            print("not finished", run.name, flush=True)
            continue
        snap = snapshot(run)
        np.savez_compressed(str(dst) + ".tmp.npz", **snap)
        pathlib.Path(str(dst) + ".tmp.npz").rename(dst)
        print("wrote", dst.name, {k: v.shape for k, v in snap.items() if k.startswith(("obj_pos", "place_pos", "room_pos", "mesh_xyz"))}, flush=True)
        del snap
        gc.collect()


if __name__ == "__main__":
    main()
