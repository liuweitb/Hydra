"""Synthetic scenes for the feature tests.

A scene is a dict of prototype-style arrays (the names features_proto.Snap exposed). `snapshot_from_arrays` turns one into a
canonical Snapshot without spark_dsg, so the tests cover Snapshot -> FeatureView -> feature end to end.
"""

import importlib.util
import pathlib
import sys

import numpy as np
from scipy.spatial import cKDTree

from layer_attribution import snapshot as sn

OBJ_LABELS = (5, 6, 7, 10, 12, 13, 18)
SIZE = 6.0
ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTO = ROOT / "research" / "signal_probe" / "features_proto.py"

DEGENERATE = ("no_rooms", "no_objects", "no_places", "empty_mesh", "empty")


def _rng(seed, part):
    return np.random.default_rng([seed, part])


def _grid_faces(nx, ny, base):
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    a = (base + i * ny + j).ravel()
    b, c, d = a + 1, a + ny, a + ny + 1
    return np.concatenate([np.stack([a, b, c], 1), np.stack([b, d, c], 1)]).astype(np.int32)


def empty_arrays():
    z3, z2 = np.zeros((0, 3)), np.zeros((0, 2), np.int32)
    return dict(
        mesh_xyz=z3.astype(np.float32), mesh_label=np.zeros(0, np.int16), mesh_faces=np.zeros((0, 3), np.int32),
        mesh_t_first=np.zeros(0, np.int64), mesh_t_last=np.zeros(0, np.int64),
        obj_pos=z3, obj_label=np.zeros(0, np.int32), obj_bbox_center=z3, obj_bbox_dim=z3, obj_bbox_R=np.zeros((0, 3, 3)),
        obj_parent=np.zeros(0, np.int32), obj_conn_offsets=np.zeros(1, np.int64), obj_conn_vertices=np.zeros(0, np.int64),
        obj_last_update_ns=np.zeros(0, np.int64),
        place_pos=z3, place_radius=np.zeros(0), place_basis=np.zeros(0, np.int32), place_parent=np.zeros(0, np.int32), place_edges=z2,
        place_last_update_ns=np.zeros(0, np.int64), place_n_obj_children=np.zeros(0, np.int32),
        room_pos=z3, room_has_parent=np.zeros(0, bool), room_edges=z2, building_pos=z3, meshplace_pos=z3,
        graph_counts=np.zeros(2, np.int64), traj=np.zeros((0, 8)),
        stats_layer2_last=np.full(6, np.nan), stats_layer3_last=np.full(6, np.nan), stats_layer4_last=np.full(6, np.nan))


def complete(a):
    """All arrays of the prototype contract; per-row arrays missing next to their primary array get neutral values."""
    d = empty_arrays()
    d.update(a)
    rows = {"obj_pos": dict(obj_label=(np.int32, 5), obj_parent=(np.int32, -1), obj_last_update_ns=(np.int64, 0)),
            "place_pos": dict(place_radius=(float, 1.0), place_basis=(np.int32, 0), place_parent=(np.int32, -1),
                              place_last_update_ns=(np.int64, 0), place_n_obj_children=(np.int32, 0)),
            "mesh_xyz": dict(mesh_label=(np.int16, 3), mesh_t_first=(np.int64, 0), mesh_t_last=(np.int64, 0)),
            "room_pos": dict(room_has_parent=(bool, False))}
    for primary, fields in rows.items():
        for k, (dt, fill) in fields.items():
            if k not in a:
                d[k] = np.full(len(d[primary]), fill, dt)
    n = len(d["obj_pos"])
    if "obj_bbox_center" not in a:
        d["obj_bbox_center"] = np.array(d["obj_pos"], float)
    if "obj_bbox_dim" not in a:
        d["obj_bbox_dim"] = np.ones((n, 3))
    if "obj_bbox_R" not in a:
        d["obj_bbox_R"] = np.tile(np.eye(3), (n, 1, 1))
    if "obj_conn_offsets" not in a:
        d["obj_conn_offsets"] = np.zeros(n + 1, np.int64)
    return d


def make_scene(seed=0, n_obj=35, n_place=120, g=64, jitter=0.0, relabel=0.0, drop_objects=0, drop_frames=False, drift=0.0, n_keyframes=100):
    """Floor grid, two walls, object clusters, void speckle, places in 4 rooms, circular trajectory.

    The base geometry depends on `seed` only (one rng stream per component), so a perturbed copy of a scene shares
    everything the perturbation does not touch: jitter (std, m) on all vertices, relabel (fraction of object-cluster
    vertices), drop_objects (last k objects), drop_frames (every 5th keyframe), drift (z offset growing with first-seen time).
    """
    L = SIZE
    rm, ro, rp, rt, rj = (_rng(seed, k) for k in range(5))
    xs, zs = np.linspace(0, L, g), np.linspace(0, 2.5, 16)
    fx, fy = np.meshgrid(xs, xs, indexing="ij")
    parts = [np.stack([fx.ravel(), fy.ravel(), 0.002 * rm.standard_normal(g * g)], 1)]
    labels, faces, base = [np.full(g * g, 3)], [_grid_faces(g, g, 0)], g * g
    u, v = np.meshgrid(xs, zs, indexing="ij")
    u, v = u.ravel(), v.ravel()
    for wall in (np.stack([np.zeros_like(u), u, v], 1), np.stack([u, np.zeros_like(u), v], 1)):
        parts.append(wall)
        labels.append(np.full(len(wall), 19))
        faces.append(_grid_faces(g, len(zs), base))
        base += len(wall)

    obj_pos = np.c_[ro.uniform(0.6, L - 0.6, (n_obj, 2)), np.full(n_obj, 0.4)]
    obj_label = np.array([OBJ_LABELS[i % len(OBJ_LABELS)] for i in range(n_obj)], np.int32)
    obj_dim = ro.uniform(0.25, 1.2, (n_obj, 3))
    obj_dim[::6] = 0.2
    per = 30
    conn = []
    for i in range(n_obj):
        parts.append(obj_pos[i] + ro.uniform(-0.2, 0.2, (per, 3)))
        labels.append(np.full(per, obj_label[i]))
        conn.append(np.arange(base, base + per))
        base += per
    parts.append(rm.uniform(0, L, (150, 3)))
    labels.append(np.zeros(150, int))
    xyz, lab = np.concatenate(parts), np.concatenate(labels)
    n = len(xyz)
    first = np.where(rm.random(n) < 0.5, rm.uniform(0, 100, n), rm.uniform(200, 300, n)) * 1e9
    last = first + 5e9
    if relabel > 0:
        cl = np.concatenate(conn)
        pick = cl[rj.random(len(cl)) < relabel]
        lab[pick] = rj.choice(OBJ_LABELS, len(pick))
    xyz = xyz + (jitter * rj.standard_normal(xyz.shape) if jitter > 0 else 0.0)
    xyz[:, 2] += drift * first / 300e9
    keep = n_obj - drop_objects
    conn = conn[:keep]

    plc = np.c_[rp.uniform(0.3, L - 0.3, (n_place, 2)), np.full(n_place, 1.0)]
    place_room = (plc[:, 0] > L / 2).astype(np.int32) + 2 * (plc[:, 1] > L / 2).astype(np.int32)
    radius = rp.uniform(0.3, 1.5, n_place)
    _, nb = cKDTree(plc).query(plc, k=4)
    pe = np.unique(np.sort(np.stack([np.repeat(np.arange(n_place), 3), nb[:, 1:].ravel()], 1), axis=1), axis=0).astype(np.int32)
    parent = cKDTree(plc).query(obj_pos[:keep])[1].astype(np.int32)
    parent[::7] = -1
    rooms = np.array([[L / 4, L / 4, 1.0], [3 * L / 4, L / 4, 1.0], [L / 4, 3 * L / 4, 1.0], [3 * L / 4, 3 * L / 4, 1.0]])

    idx = np.arange(n_keyframes)
    th = idx * 0.06
    pos = np.c_[L / 2 + 2 * np.cos(th), L / 2 + 2 * np.sin(th), np.full(n_keyframes, 1.2)] + 0.01 * rt.standard_normal((n_keyframes, 3))
    quat = np.c_[np.cos(th / 2), np.zeros(n_keyframes), np.zeros(n_keyframes), np.sin(th / 2)]
    traj = np.c_[idx * 1e9, pos, quat]
    if drop_frames:
        traj = traj[idx % 5 != 0]
    offs = np.concatenate([[0], np.cumsum([len(c) for c in conn])]).astype(np.int64)
    return dict(
        mesh_xyz=xyz.astype(np.float32), mesh_label=lab.astype(np.int16), mesh_faces=np.concatenate(faces),
        mesh_t_first=first.astype(np.int64), mesh_t_last=last.astype(np.int64),
        obj_pos=obj_pos[:keep], obj_label=obj_label[:keep], obj_bbox_center=obj_pos[:keep].astype(np.float32).astype(np.float64),
        obj_bbox_dim=obj_dim[:keep].astype(np.float32).astype(np.float64), obj_bbox_R=np.tile(np.eye(3), (keep, 1, 1)),
        obj_parent=parent, obj_conn_offsets=offs, obj_conn_vertices=np.concatenate(conn) if conn else np.zeros(0, np.int64),
        obj_last_update_ns=np.arange(keep, dtype=np.int64),
        place_pos=plc, place_radius=radius.astype(np.float32).astype(np.float64), place_basis=np.full(n_place, 3, np.int32),
        place_parent=place_room, place_edges=pe, place_last_update_ns=np.arange(n_place, dtype=np.int64),
        place_n_obj_children=np.bincount(parent[parent >= 0], minlength=n_place).astype(np.int32),
        room_pos=rooms, room_has_parent=np.zeros(4, bool), room_edges=np.array([[0, 1], [1, 3], [3, 2], [2, 0]], np.int32),
        building_pos=np.array([[L / 2, L / 2, 1.0]]), meshplace_pos=rp.uniform(0, L, (30, 3)),
        graph_counts=np.array([keep + n_place + 4 + len(traj) + 1, len(pe) + 10]),
        traj=traj, stats_layer2_last=np.array([keep, 8, 3, keep, 0, 0.0]), stats_layer3_last=np.array([n_place, 300, 40, n_place, 10, 400.0]),
        stats_layer4_last=np.array([4, 0, 0, 0, 4, 4.0]))


def degenerate(a, kind):
    """The scene with one layer removed, arrays kept mutually consistent."""
    a = dict(a)
    e = empty_arrays()
    if kind in ("no_rooms", "empty"):
        a.update({k: e[k] for k in ("room_pos", "room_has_parent", "room_edges")})
        a["place_parent"] = np.full(len(a["place_pos"]), -1, np.int32)
    if kind in ("no_objects", "empty"):
        a.update({k: e[k] for k in ("obj_pos", "obj_label", "obj_bbox_center", "obj_bbox_dim", "obj_bbox_R", "obj_parent",
                                    "obj_conn_offsets", "obj_conn_vertices", "obj_last_update_ns")})
        a["place_n_obj_children"] = np.zeros(len(a["place_pos"]), np.int32)
    if kind in ("no_places", "empty"):
        a.update({k: e[k] for k in ("place_pos", "place_radius", "place_basis", "place_parent", "place_edges", "place_last_update_ns",
                                    "place_n_obj_children")})
        a["obj_parent"] = np.full(len(a["obj_pos"]), -1, np.int32)
    if kind in ("empty_mesh", "empty"):
        a.update({k: e[k] for k in ("mesh_xyz", "mesh_label", "mesh_faces", "mesh_t_first", "mesh_t_last")})
        a["obj_conn_offsets"] = np.zeros(len(a["obj_pos"]) + 1, np.int64)
        a["obj_conn_vertices"] = np.zeros(0, np.int64)
    if kind == "empty":
        a.update({k: e[k] for k in ("meshplace_pos", "building_pos", "traj")})
    return a


def snapshot_from_arrays(a, name="syn", extra_parents=None):
    """Canonical Snapshot with all tables and fields; `extra_parents` {object row: place row} gives those objects a second parent."""
    a = complete(a)
    no, npl, nr, nm = len(a["obj_pos"]), len(a["place_pos"]), len(a["room_pos"]), len(a["mesh_xyz"])
    extra = extra_parents or {}
    par = [sorted({int(a["obj_parent"][i]), *([int(extra[i])] if i in extra else [])} - {-1}) for i in range(no)]
    par_ip = np.concatenate([[0], np.cumsum([len(p) for p in par])]).astype(np.int64)
    par_ix = np.array([p for x in par for p in x], np.int32)
    parent_place = np.array([p[0] if p else -1 for p in par], np.int32)
    place_room = np.asarray(a["place_parent"], np.int32)
    ids = lambda ch, n: (ord(ch) << 56) + np.arange(n, dtype=np.int64)  # noqa: E731
    conn_ip, conn_ix = np.asarray(a["obj_conn_offsets"], np.int64), np.asarray(a["obj_conn_vertices"], np.int32)
    child = [np.flatnonzero(place_room == r).astype(np.int32) for r in range(nr)]
    ch_ip = np.concatenate([[0], np.cumsum([len(c) for c in child])]).astype(np.int64)
    deg = lambda ij, n: np.bincount(np.asarray(ij).ravel(), minlength=n).astype(np.int32)  # noqa: E731
    pos_p = a["place_pos"]
    T = {
        "objects": dict(id=ids("O", no), pos=a["obj_pos"], label=a["obj_label"], bbox_type=np.ones(no, np.int8),
                        bbox_center=np.asarray(a["obj_bbox_center"], np.float32), bbox_dim=np.asarray(a["obj_bbox_dim"], np.float32),
                        bbox_rot=np.asarray(a["obj_bbox_R"], np.float32).reshape(no, 3, 3), last_update_ns=a["obj_last_update_ns"],
                        is_active=np.zeros(no, bool), n_mesh_conn=np.diff(conn_ip).astype(np.int32), mesh_conn_indptr=conn_ip,
                        mesh_conn_idx=conn_ix, n_parents=np.array([len(p) for p in par], np.int16), parent_indptr=par_ip, parent_idx=par_ix,
                        parent_place=parent_place, room=np.where(parent_place >= 0, place_room[np.maximum(parent_place, 0)] if npl else -1, -1).astype(np.int32)),
        "places": dict(id=ids("p", npl), pos=pos_p, dist=np.asarray(a["place_radius"], np.float32), n_basis=np.asarray(a["place_basis"], np.int16),
                       last_update_ns=a["place_last_update_ns"], is_active=np.zeros(npl, bool), basis_indptr=np.zeros(npl + 1, np.int64),
                       basis_pos=np.zeros((0, 3), np.float32), room=place_room, n_room_parents=(place_room >= 0).astype(np.int16),
                       deg=deg(a["place_edges"], npl), n_obj=a["place_n_obj_children"], n_agent=np.zeros(npl, np.int32)),
        "place_edges": dict(ij=np.asarray(a["place_edges"], np.int32).reshape(-1, 2), w=np.ones(len(a["place_edges"]), np.float32)),
        "mplaces": dict(id=ids("P", len(a["meshplace_pos"])), pos=a["meshplace_pos"], label=np.zeros(len(a["meshplace_pos"]), np.int32),
                        ellipse_centroid=np.zeros((len(a["meshplace_pos"]), 3), np.float32),
                        ellipse_expand=np.zeros((len(a["meshplace_pos"]), 2, 2), np.float32), is_active=np.zeros(len(a["meshplace_pos"]), bool),
                        mesh_conn_indptr=np.zeros(len(a["meshplace_pos"]) + 1, np.int64), mesh_conn_idx=np.zeros(0, np.int32),
                        boundary_indptr=np.zeros(len(a["meshplace_pos"]) + 1, np.int64), boundary_xyz=np.zeros((0, 3), np.float32),
                        deg=np.zeros(len(a["meshplace_pos"]), np.int32)),
        "mplace_edges": dict(ij=np.zeros((0, 2), np.int32), w=np.zeros(0, np.float32)),
        "rooms": dict(id=ids("R", nr), pos=a["room_pos"], child_indptr=ch_ip, child_idx=np.concatenate(child) if child else np.zeros(0, np.int32),
                      n_children=np.diff(ch_ip).astype(np.int32),
                      ext_min=np.array([pos_p[c].min(0) if len(c) else [np.nan] * 3 for c in child], np.float32).reshape(nr, 3),
                      ext_max=np.array([pos_p[c].max(0) if len(c) else [np.nan] * 3 for c in child], np.float32).reshape(nr, 3),
                      has_parent=np.asarray(a["room_has_parent"], bool), deg=deg(a["room_edges"], nr)),
        "room_edges": dict(ij=np.asarray(a["room_edges"], np.int32).reshape(-1, 2), w=np.ones(len(a["room_edges"]), np.float32)),
        "building": dict(id=ids("B", len(a["building_pos"])), pos=a["building_pos"], label=np.full(len(a["building_pos"]), -1, np.int32)),
        "traj": dict(t_ns=np.rint(a["traj"][:, 0]).astype(np.int64), pos=a["traj"][:, 1:4], quat_wxyz=a["traj"][:, 4:8],
                     agent_id=ids("a", len(a["traj"])), place=np.full(len(a["traj"]), -1, np.int32)),
        "mesh": dict(xyz=np.asarray(a["mesh_xyz"], np.float32), rgb=np.zeros((nm, 3), np.uint8),
                     label=np.where(a["mesh_label"] < 0, sn.MESH_NO_LABEL, a["mesh_label"].astype(np.int32)).astype(np.uint16),
                     stamp_ns=a["mesh_t_last"], first_seen_ns=a["mesh_t_first"], faces=np.asarray(a["mesh_faces"], np.int32).reshape(-1, 3)),
        "stats": {f"layer_{k}_last": np.asarray(a[f"stats_layer{k}_last"], np.float64) for k in sn.STATS_LAYERS},
    }
    meta = {"schema": sn.SCHEMA, "run": name, "src": {"path": "", "size": 0, "mtime_ns": 0, "side": {f: None for f in sn.SIDE_FILES}},
            "counts": {"objects": no, "places": npl, "mplaces": len(a["meshplace_pos"]), "rooms": nr, "agents": len(a["traj"]), "mesh_vertices": nm,
                       "mesh_faces": len(a["mesh_faces"]), "place_edges": len(a["place_edges"]), "mplace_edges": 0, "room_edges": len(a["room_edges"])},
            "graph_counts": {"nodes": int(a["graph_counts"][0]), "edges": int(a["graph_counts"][1])}, "edge_counts": {}, "edges_unrepresented": 0,
            "mesh_flags": {}, "label_names": {}, "bbox_type_names": {}, "no_label": sn.NO_LABEL, "mesh_no_label": sn.MESH_NO_LABEL,
            "traj_source": "trajectory.csv", "traj_matches_agents": False, "warnings": [], "build_seconds": {"dsg_load": 0.0, "total": 0.0}}
    return sn.Snapshot(meta, {k: sn.Table(k, v) for k, v in T.items()})


def load_prototype():
    """research/signal_probe/features_proto.py as a module (None when the file is gone)."""
    if not PROTO.is_file():
        return None
    name = "features_proto_legacy"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PROTO)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass resolves string annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


def prototype_snap(mod, a, name="syn"):
    s = mod.Snap.__new__(mod.Snap)
    s.a = complete(a)
    s.name = name
    s.manifest = {}
    return s
