"""Empirical checks of source-level claims on saved dev graphs (uh2_clean and seed-0 / severity-0.5 label-fault runs).

usage: python verify_claims.py RUNS_ROOT OUT_JSON
Reads only uh2_clean, uh2_label_confusion_s50, uh2_label_dropout_s50 (seed 0 = development data).
"""
import collections
import json
import sys

import numpy as np
import spark_dsg as dsg


def key_str(k):
    return f"{k.layer}" if k.partition == 0 else f"{k.layer}[{k.partition}]"


def nid(x):
    return int(x.value) if hasattr(x, "value") else int(x)


def pos_of(layer):
    ids = [nid(n.id) for n in layer.nodes]
    xyz = np.array([np.asarray(n.attributes.position, dtype=float) for n in layer.nodes])
    return ids, xyz


def parent_checks(G):
    """object -> parent place: nearest place by 3D distance? distance to parent, room via parent."""
    objs = list(G.get_layer("OBJECTS").nodes)
    pids, pxyz = pos_of(G.get_layer("PLACES"))
    pidx = {i: k for k, i in enumerate(pids)}
    radius = np.array([float(G.get_node(i).attributes.distance) for i in pids])
    out = collections.defaultdict(list)
    n_two_parents = 0
    two_parent_diff_room = 0
    for o in objs:
        ps = [nid(p) for p in o.parents()]
        if not ps:
            out["no_parent"].append(1)
            continue
        x = np.asarray(o.attributes.position, dtype=float)
        d = np.linalg.norm(pxyz - x, axis=1)
        order = np.argsort(d)
        rank_of_parent = min(int(np.where(order == pidx[p])[0][0]) for p in ps if p in pidx)
        out["parent_rank"].append(rank_of_parent)
        best_parent = min(ps, key=lambda p: d[pidx[p]])
        out["dist_to_parent"].append(float(d[pidx[best_parent]]))
        out["parent_radius"].append(float(radius[pidx[best_parent]]))
        out["inside_parent_sphere"].append(int(d[pidx[best_parent]] <= radius[pidx[best_parent]]))
        near_id = pids[order[0]]
        room_of = lambda pid: [nid(r) for r in G.get_node(pid).parents()]
        out["same_room_as_nearest_place"].append(int(room_of(best_parent) == room_of(near_id)))
        if len(ps) > 1:
            n_two_parents += 1
            rooms = {tuple(room_of(p)) for p in ps}
            two_parent_diff_room += int(len(rooms) > 1)
    r = {
        "n_objects": len(objs),
        "no_parent": len(out["no_parent"]),
        "n_two_parents": n_two_parents,
        "two_parents_in_different_rooms": two_parent_diff_room,
    }
    pr = np.array(out["parent_rank"])
    r["parent_is_nearest_place_frac"] = float((pr == 0).mean())
    r["parent_in_3_nearest_frac"] = float((pr < 3).mean())
    r["parent_rank_max"] = int(pr.max())
    dp = np.array(out["dist_to_parent"])
    r["dist_object_to_parent_place_m"] = [float(dp.min()), float(np.median(dp)), float(dp.max())]
    r["inside_parent_sphere_frac"] = float(np.mean(out["inside_parent_sphere"]))
    r["room_via_parent_equals_room_of_nearest_place_frac"] = float(np.mean(out["same_room_as_nearest_place"]))
    return r


def building_checks(G):
    out = {}
    b = [n for n in G.nodes if n.layer.layer == 5]
    rooms = list(G.get_layer("ROOMS").nodes)
    out["n_buildings"] = len(b)
    out["n_rooms"] = len(rooms)
    if b:
        bn = b[0]
        out["building_id"] = str(bn.id)
        out["building_num_children"] = len(list(bn.children()))
        out["building_num_parents"] = len(list(bn.parents()))
        rc = np.mean([np.asarray(r.attributes.position, dtype=float) for r in rooms], axis=0) if rooms else None
        out["building_position"] = [float(v) for v in np.asarray(bn.attributes.position)]
        out["rooms_centroid_at_save"] = [float(v) for v in rc] if rc is not None else None
        out["abs_diff_building_vs_room_centroid_m"] = float(np.abs(np.asarray(bn.attributes.position) - rc).max()) if rc is not None else None
    out["room_ids"] = [str(r.id) for r in rooms]
    out["room_num_parents"] = [len(list(r.parents())) for r in rooms]
    return out


def mesh_arrays(G):
    m = G.mesh
    V = np.asarray(m.get_vertices())
    L = np.asarray(m.get_labels(), dtype=np.int64)
    return V[:3].T if V.shape[0] in (3, 6) else V[:, :3], L


def layer_snapshot(G):
    snap = {}
    for k in ("OBJECTS", "PLACES", "MESH_PLACES", "ROOMS"):
        ids, xyz = pos_of(G.get_layer(k))
        order = np.argsort(ids)
        snap[k] = (np.array(ids)[order], xyz[order] if len(xyz) else xyz)
    return snap


def compare_runs(clean_snap, clean_mesh, G):
    V0, L0 = clean_mesh
    V1, L1 = mesh_arrays(G)
    r = {"mesh": {"n_vertices": [int(len(V0)), int(len(V1))]}}
    if V0.shape == V1.shape:
        r["mesh"]["max_abs_vertex_diff_m"] = float(np.abs(V0 - V1).max())
        ch = L0 != L1
        r["mesh"]["frac_vertices_label_changed"] = float(ch.mean())
        trans = collections.Counter((int(a), int(b)) for a, b in zip(L0[ch], L1[ch]))
        r["mesh"]["top_label_transitions"] = [[a, b, c] for (a, b), c in trans.most_common(12)]
        r["mesh"]["label_hist_clean"] = {int(k): int(v) for k, v in sorted(collections.Counter(L0.tolist()).items())}
        r["mesh"]["label_hist_run"] = {int(k): int(v) for k, v in sorted(collections.Counter(L1.tolist()).items())}
    snap = layer_snapshot(G)
    for k, (ids0, xyz0) in clean_snap.items():
        ids1, xyz1 = snap[k]
        e = {"n": [int(len(ids0)), int(len(ids1))], "same_ids": bool(len(ids0) == len(ids1) and (ids0 == ids1).all())}
        if e["same_ids"] and len(xyz0):
            e["max_abs_pos_diff_m"] = float(np.abs(xyz0 - xyz1).max())
        r[k] = e
    return r


def main(root, out_path):
    res = {}
    G = dsg.DynamicSceneGraph.load(f"{root}/uh2_clean/backend/dsg_with_mesh.json")
    res["clean_parent_checks"] = parent_checks(G)
    res["clean_building_checks"] = building_checks(G)
    clean_snap = layer_snapshot(G)
    clean_mesh = mesh_arrays(G)
    res["clean_attribute_semantics"] = {
        "object_semantic_labels": sorted({int(n.attributes.semantic_label) for n in G.get_layer("OBJECTS").nodes}),
        "mesh_places_semantic_labels": sorted({int(n.attributes.semantic_label) for n in G.get_layer("MESH_PLACES").nodes}),
    }
    del G
    for run in ("uh2_label_confusion_s50", "uh2_label_dropout_s50"):
        H = dsg.DynamicSceneGraph.load(f"{root}/{run}/backend/dsg_with_mesh.json")
        res[run] = compare_runs(clean_snap, clean_mesh, H)
        del H
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print(json.dumps(res, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
