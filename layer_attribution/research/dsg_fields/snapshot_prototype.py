#!/usr/bin/env python3
"""Snapshot: numpy view of one saved Hydra run, built once from backend/dsg_with_mesh.json and cached as .npz.

build  needs spark_dsg (WSL venv): DynamicSceneGraph.load + attribute extraction, ~1 GB peak
load   needs numpy only (also works from Windows python through \\\\wsl.localhost\\...); arrays are read lazily
Rows of every node table are sorted by raw NodeId (symbol char << 56 | index). Relations are indices into the target
table (-1 = none) or CSR pairs. Unset labels are -1. A missing or empty layer gives empty arrays with the full field set,
so feature code needs no existence checks. Ground truth (injection_manifest.json) is never read here.

tables (arrays; "->T" = row index into table T; csr "<x>_{indptr,idx}" = ragged lists):
    objects   id pos label bbox_type bbox_center bbox_dim last_update_ns is_active n_mesh_conn mesh_conn_{indptr,idx}->mesh
              n_parents parent_{indptr,idx}->places parent_place->places room->rooms
    places    id pos dist n_basis basis_{indptr,pos} last_update_ns is_active room->rooms n_room_parents deg n_obj n_agent
    mplaces   id pos label ellipse_centroid ellipse_expand is_active mesh_conn_{indptr,idx}->mesh boundary_{indptr,xyz} deg
    rooms     id pos child_{indptr,idx}->places n_children ext_min ext_max (xyz extent of child places) deg
    place_edges / mplace_edges / room_edges   ij (E,2) rows of the own table, w
    building  id pos label     traj  t_ns pos quat_wxyz agent_id place->places
    mesh      xyz rgb label (uint16, 65535 = none) stamp_ns first_seen_ns faces (F,3)
meta: counts, edge_counts by category pair, edges_unrepresented, mesh_flags, label_names, traj_source, warnings, build_seconds
env: LA_RUNS_ROOT, LA_SNAPSHOT_CACHE override the default roots (needed from Windows python over the UNC path).

usage (WSL, via research/dsg_fields/wsl_py.sh):
    snapshot_prototype.py build RUN [--rebuild]      RUN = run name under ~/runs/uh2 or a run dir
    snapshot_prototype.py bench RUN                  cold/warm build and cache-load timing, peak RSS (fresh processes)
    snapshot_prototype.py selftest [--run RUN]       synthetic round trip + random degenerate graphs (+ real-run checks)
    snapshot_prototype.py verify RUN [RUN ...]       every stored value against the DSG API (exhaustive)
    snapshot_prototype.py mutate RUN                 degenerate versions of a real graph (no rooms / objects / places / mesh)
    snapshot_prototype.py survey [--out-dir DIR]     every finished development run: table + invariants
"""

import argparse
import gc
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
import zipfile

import numpy as np

SCHEMA = 2
RUNS_ROOT = pathlib.Path(os.environ.get("LA_RUNS_ROOT", "~/runs/uh2")).expanduser()  # Windows: \\wsl.localhost\Ubuntu-24.04\home\hydra\runs\uh2
CACHE_ROOT = pathlib.Path(os.environ.get("LA_SNAPSHOT_CACHE", "~/runs/cache")).expanduser()
STATUS_LOG = pathlib.Path("/mnt/d/hydra_work/sweep_logs/status_sweep_main.jsonl")
REPO_SRC = pathlib.Path(__file__).resolve().parents[2] / "src"
TABLES = ("objects", "places", "place_edges", "mplaces", "mplace_edges", "rooms", "room_edges", "building", "traj", "mesh")
CAT = {"O": "objects", "p": "places", "P": "mplaces", "R": "rooms", "B": "building", "a": "agents"}
NO_LABEL = -1  # DSG stores unset labels as uint32 4294967295
MESH_NO_LABEL = 65535
ID_MASK = (1 << 56) - 1
# category pairs the tables represent; any other edge is only counted in meta["edges_unrepresented"]
REPRESENTED_EDGES = {("O", "p"), ("R", "p"), ("a", "p"), ("a", "a"), ("p", "p"), ("P", "P"), ("R", "R")}

ROW_FIELDS = {
    "objects": ("id", "pos", "label", "bbox_type", "bbox_center", "bbox_dim", "last_update_ns", "is_active", "n_mesh_conn",
                "n_parents", "parent_place", "room"),
    "places": ("id", "pos", "dist", "n_basis", "last_update_ns", "is_active", "room", "n_room_parents", "deg", "n_obj", "n_agent"),
    "mplaces": ("id", "pos", "label", "ellipse_centroid", "ellipse_expand", "is_active", "deg"),
    "rooms": ("id", "pos", "n_children", "ext_min", "ext_max", "deg"),
    "building": ("id", "pos", "label"),
    "traj": ("t_ns", "pos", "quat_wxyz", "agent_id", "place"),
    "mesh": ("xyz", "rgb", "label", "stamp_ns", "first_seen_ns"),
}
# (table, indptr field, payload field, table the payload indexes into, or None when the payload is data)
CSR_FIELDS = (
    ("objects", "mesh_conn_indptr", "mesh_conn_idx", "mesh"),
    ("objects", "parent_indptr", "parent_idx", "places"),
    ("places", "basis_indptr", "basis_pos", None),
    ("mplaces", "mesh_conn_indptr", "mesh_conn_idx", "mesh"),
    ("mplaces", "boundary_indptr", "boundary_xyz", None),
    ("rooms", "child_indptr", "child_idx", "places"),
)


class SnapshotError(RuntimeError):
    """Run directory unusable: json missing or unreadable."""


def symbol(node_id):
    node_id = int(node_id)
    return f"{chr(node_id >> 56)}{node_id & ID_MASK}"


class Table:
    """Named numpy arrays of one entity; arrays load from the npz on first access."""

    def __init__(self, name, arrays=None, npz=None):
        self._name, self._arrays, self._npz = name, dict(arrays or {}), npz

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError(key)
        if key not in self._arrays:
            if self._npz is None or f"{self._name}.{key}" not in self._npz.files:
                raise AttributeError(f"{self._name}.{key}")
            self._arrays[key] = self._npz[f"{self._name}.{key}"]
        return self._arrays[key]

    def fields(self):
        names = set(self._arrays)
        if self._npz is not None:
            names |= {k.split(".", 1)[1] for k in self._npz.files if k.startswith(self._name + ".")}
        return sorted(names)

    def __len__(self):
        names = self.fields()
        for k in ("id", "t_ns", "xyz", "ij"):
            if k in names:
                return len(getattr(self, k))
        return 0

    def csr(self, indptr, idx, i):
        """Row i of a CSR pair stored as fields `<indptr>` / `<idx>`."""
        return getattr(self, idx)[getattr(self, indptr)[i]:getattr(self, indptr)[i + 1]]


class Snapshot:
    from_cache = None  # set by open()

    def __init__(self, meta, tables, npz=None):
        self.meta, self._tables, self._npz = meta, tables, npz

    def __getattr__(self, key):
        if key in TABLES:
            return self.__dict__["_tables"][key]
        raise AttributeError(key)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._npz is not None:
            self._npz.close()
            self._npz = None

    def materialize(self):
        """Read every array into memory and release the npz file handle."""
        for t in self._tables.values():
            for f in t.fields():
                getattr(t, f)
        self.close()
        return self

    @property
    def label_names(self):
        return {int(k): v for k, v in self.meta["label_names"].items()}

    def mesh_subsample(self, n=50000):
        """Deterministic stride subsample of mesh vertex indices (same rule for every run)."""
        nv = len(self.mesh.xyz)
        return np.arange(0, nv, max(1, nv // n))

    def flat(self):
        out = {"meta": np.array(json.dumps(self.meta))}
        for tn, t in self._tables.items():
            for f in t.fields():
                out[f"{tn}.{f}"] = getattr(t, f)
        return out

    def save(self, path, compress=False):
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            (np.savez_compressed if compress else np.savez)(f, **self.flat())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        npz = np.load(path, allow_pickle=False)
        meta = json.loads(str(npz["meta"]))
        return cls(meta, {tn: Table(tn, npz=npz) for tn in TABLES}, npz)

    @classmethod
    def open(cls, run, cache_root=CACHE_ROOT, rebuild=False, cache=True):
        """Cache hit when schema and source (size, mtime) match, else build from the json and write the cache."""
        run_dir = resolve_run(run)
        src = run_dir / "backend" / "dsg_with_mesh.json"
        if not src.is_file():
            raise SnapshotError(f"{src} missing (run incomplete?)")
        cp = cache_path(run_dir, cache_root)
        if cache and cp.exists() and not rebuild:
            try:
                snap = cls.load(cp)
                st, s = src.stat(), snap.meta["src"]
                # 1 us tolerance: mtime read through the Windows UNC path is rounded to 100 ns
                if snap.meta["schema"] == SCHEMA and s["size"] == st.st_size and abs(s["mtime_ns"] - st.st_mtime_ns) < 1000:
                    snap.from_cache = True
                    return snap
                snap.close()
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                pass  # stale or damaged cache: rebuild below
        snap = build_snapshot(run_dir)
        snap.from_cache = False
        if cache:
            try:
                snap.save(cp)
            except OSError as e:
                snap.meta["warnings"].append(f"cache not written: {e}")
        return snap

    def check(self):
        return check_invariants(self)

    def summary(self):
        return (f"Snapshot({self.meta['run']}): " + ", ".join(f"{k}={v}" for k, v in self.meta["counts"].items())
                + f"; traj matches agents: {self.meta['traj_matches_agents']}")


def resolve_run(run):
    p = pathlib.Path(str(run)).expanduser()
    return p if (p / "backend").is_dir() else RUNS_ROOT / str(run)


def cache_path(run_dir, cache_root=CACHE_ROOT):
    return pathlib.Path(cache_root).expanduser() / f"{pathlib.Path(run_dir).name}.snap{SCHEMA}.npz"


# ------------------------------------------------------------------------------------------------ invariants
def check_invariants(s):
    """Structural violations of the Snapshot contract (empty list = sound). Never raises on a built Snapshot."""
    bad = []
    n = {tn: len(s._tables[tn]) for tn in TABLES}

    def arr(tn, f):
        try:
            return getattr(s._tables[tn], f)
        except AttributeError:
            bad.append(f"{tn}.{f} missing")
            return None

    for tn, fields in ROW_FIELDS.items():
        for f in fields:
            a = arr(tn, f)
            if a is not None and len(a) != n[tn]:
                bad.append(f"{tn}.{f} has {len(a)} rows, table has {n[tn]}")
    for tn, ip_f, ix_f, target in CSR_FIELDS:
        ip, ix = arr(tn, ip_f), arr(tn, ix_f)
        if ip is None or ix is None:
            continue
        if len(ip) != n[tn] + 1 or ip[0] != 0 or ip[-1] != len(ix) or (np.diff(ip) < 0).any():
            bad.append(f"{tn}.{ip_f} is not a valid CSR pointer for {len(ix)} entries")
        elif target is not None and len(ix) and (ix.min() < 0 or ix.max() >= n[target]):
            bad.append(f"{tn}.{ix_f} outside {target} [0, {n[target]})")
    for tn, target in (("place_edges", "places"), ("mplace_edges", "mplaces"), ("room_edges", "rooms")):
        ij, w = arr(tn, "ij"), arr(tn, "w")
        if ij is not None and w is not None:
            if ij.ndim != 2 or ij.shape[1] != 2 or len(w) != len(ij):
                bad.append(f"{tn} shape {ij.shape} / weights {w.shape}")
            elif len(ij) and (ij.min() < 0 or ij.max() >= n[target]):
                bad.append(f"{tn}.ij outside {target}")
    for tn, f, target in (("objects", "parent_place", "places"), ("objects", "room", "rooms"), ("places", "room", "rooms"),
                          ("traj", "place", "places")):
        a = arr(tn, f)
        if a is not None and len(a) and (a.min() < -1 or a.max() >= n[target]):
            bad.append(f"{tn}.{f} outside [-1, {n[target]})")
    for tn in ("objects", "places", "mplaces", "rooms", "building"):
        ids = arr(tn, "id")
        if ids is not None and len(ids) > 1 and (np.diff(ids) <= 0).any():
            bad.append(f"{tn}.id not strictly increasing")
    for tn, f in (("objects", "pos"), ("places", "pos"), ("mplaces", "pos"), ("rooms", "pos"), ("traj", "pos"), ("traj", "quat_wxyz"),
                  ("mesh", "xyz"), ("places", "dist")):
        a = arr(tn, f)
        if a is not None and a.size and not np.isfinite(a).all():
            bad.append(f"{tn}.{f} has {int((~np.isfinite(a)).sum())} non-finite values")
    faces = arr("mesh", "faces")
    if faces is not None and len(faces) and (faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= n["mesh"]):
        bad.append("mesh.faces outside [0, n_vertices)")
    nrp, room, ci = arr("places", "n_room_parents"), arr("places", "room"), arr("rooms", "child_idx")
    if nrp is not None and room is not None and ci is not None and len(nrp) == len(room) and len(ci) and ci.max() < len(nrp):
        if not np.array_equal(np.bincount(ci, minlength=len(nrp)), nrp):
            bad.append("places.n_room_parents disagrees with rooms.child_idx")
        if not np.array_equal(room >= 0, nrp >= 1):
            bad.append("places.room set without a room parent (or the reverse)")
    return bad


# ------------------------------------------------------------------------------------------------ build
def _indptr(lens):
    ip = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(np.asarray(lens, dtype=np.int64), out=ip[1:])
    return ip


def _csr(lists, dtype):
    ip = _indptr([len(x) for x in lists])
    return ip, np.fromiter((v for x in lists for v in x), dtype=dtype, count=int(ip[-1]))


def _rows(values, n, width, dtype):
    return np.array(values, dtype=dtype).reshape(n, width)


def _label(v):
    v = int(v)
    return v if 0 <= v < 2**31 else NO_LABEL


def _edge_table(edges, imap):
    """(E,2) index pairs + float32 weights for edges with both endpoints in imap (id -> row)."""
    ij, w = [], []
    for s, t, wt in edges:
        if s in imap and t in imap:
            ij.append((imap[s], imap[t]))
            w.append(wt)
    return {"ij": np.array(ij, dtype=np.int32).reshape(-1, 2), "w": np.array(w, dtype=np.float32)}


def _degree(ij, n):
    return np.bincount(ij.ravel(), minlength=n).astype(np.int32)


def _mesh_tables(g, warns):
    """Mesh arrays plus flags for the optional per-vertex channels; absent channels are zero-filled."""
    flags = {"has_mesh": False, "labels": False, "colors": False, "stamps": False, "first_seen": False}
    m = g.mesh if g.has_mesh() else None
    nv = m.num_vertices() if m is not None else 0
    out = {"xyz": np.zeros((0, 3), np.float32), "rgb": np.zeros((0, 3), np.uint8), "label": np.zeros(0, np.uint16),
           "stamp_ns": np.zeros(0, np.int64), "first_seen_ns": np.zeros(0, np.int64), "faces": np.zeros((0, 3), np.int32)}
    if m is None:
        warns.append("no mesh in file")
        return out, flags
    flags["has_mesh"] = True
    V = np.asarray(m.get_vertices())
    out["xyz"] = np.ascontiguousarray(V[:3].T, dtype=np.float32)
    out["faces"] = np.ascontiguousarray(np.asarray(m.get_faces()).T, dtype=np.int32).reshape(-1, 3)
    if nv == 0:
        warns.append("mesh has no vertices")
        return out, flags

    def probe(fn):
        try:
            fn(0)
            return True
        except IndexError:
            return False

    labels = list(m.get_labels())
    flags["labels"] = len(labels) == nv
    flags["colors"] = probe(m.color)
    flags["stamps"] = probe(m.timestamp)
    flags["first_seen"] = probe(m.first_seen_timestamp)
    out.update(rgb=np.zeros((nv, 3), np.uint8), label=np.full(nv, MESH_NO_LABEL, np.uint16), stamp_ns=np.zeros(nv, np.int64),
               first_seen_ns=np.zeros(nv, np.int64))
    if flags["labels"]:
        out["label"] = np.minimum(np.array(labels, dtype=np.int64), MESH_NO_LABEL).astype(np.uint16)
    if flags["colors"] and V.shape[0] >= 6:
        out["rgb"] = np.rint(np.clip(V[3:6].T, 0, 1) * 255).astype(np.uint8)
    if flags["stamps"]:
        out["stamp_ns"] = np.fromiter((m.timestamp(i) for i in range(nv)), dtype=np.int64, count=nv)
    if flags["first_seen"]:
        out["first_seen_ns"] = np.fromiter((m.first_seen_timestamp(i) for i in range(nv)), dtype=np.int64, count=nv)
    missing = [k for k in ("labels", "colors", "stamps", "first_seen") if not flags[k]]
    if missing:
        warns.append("mesh channels absent (filled with 0, labels with MESH_NO_LABEL): " + ",".join(missing))
    return out, flags


def _trajectory(run_dir, A, apos, place_of_agent, agent_ids, warns):
    """Trajectory from backend/trajectory.csv (exact ns timestamps) when it agrees with the agent layer, else from the agents."""
    csv = run_dir / "backend" / "trajectory.csv"
    d = None
    if csv.exists():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # "input contained no data" for a header-only file
                d = np.loadtxt(csv, delimiter=",", skiprows=1, ndmin=2)
        except (ValueError, OSError) as e:
            warns.append(f"trajectory.csv unreadable: {e}")
        if d is not None and (d.ndim != 2 or d.shape[1] < 8 or len(d) == 0):
            warns.append("trajectory.csv has no usable rows")
            d = None
    if d is None:
        t = {"t_ns": np.array([int(round(n.attributes.timestamp.total_seconds() * 1e9)) for n in A], dtype=np.int64),
             "pos": apos, "quat_wxyz": _rows([[q.w, q.x, q.y, q.z] for q in (n.attributes.world_R_body for n in A)], len(A), 4, np.float64)}
        t.update(agent_id=agent_ids, place=place_of_agent)
        return t, "agents", None
    t = {"t_ns": np.rint(d[:, 0]).astype(np.int64), "pos": d[:, 1:4].copy(), "quat_wxyz": d[:, 4:8].copy()}
    match = bool(len(d) == len(A) and (len(A) == 0 or np.abs(d[:, 1:4] - apos).max() < 1e-3))
    if match:
        t.update(agent_id=agent_ids, place=place_of_agent)
    else:
        warns.append(f"trajectory.csv ({len(d)} rows) does not match the agent layer ({len(A)} nodes): no place link")
        t.update(agent_id=np.full(len(d), -1, dtype=np.int64), place=np.full(len(d), -1, dtype=np.int32))
    return t, "trajectory.csv", match


def build_snapshot(run_dir):
    try:
        import spark_dsg as dsg
    except ImportError as e:
        raise SnapshotError("spark_dsg not importable (build needs the WSL Hydra venv)") from e

    t0 = time.time()
    run_dir = pathlib.Path(run_dir)
    src = run_dir / "backend" / "dsg_with_mesh.json"
    if not src.is_file():
        raise SnapshotError(f"{src} missing")
    try:
        g = dsg.DynamicSceneGraph.load(str(src))
    except Exception as e:  # spark_dsg reports bad json as assorted RuntimeError / ValueError types
        raise SnapshotError(f"cannot load {src}: {type(e).__name__}: {e}") from e
    t_load = time.time() - t0
    warns = []
    with np.errstate(all="ignore"):  # default-constructed attributes can hold huge doubles
        return _extract(g, dsg, run_dir, src, t0, t_load, warns)


def _extract(g, dsg, run_dir, src, t0, t_load, warns):
    nodes = {c: [] for c in CAT.values()}
    cat_of, unknown = {}, {}
    for key in g.layer_keys:
        for n in g.get_layer(key.layer, key.partition).nodes:
            ch = n.id.category
            cat_of[int(n.id.value)] = ch
            if ch in CAT:
                nodes[CAT[ch]].append(n)
            else:
                unknown[ch] = unknown.get(ch, 0) + 1
    for v in nodes.values():
        v.sort(key=lambda n: int(n.id.value))
    ids = {c: np.array([int(n.id.value) for n in v], dtype=np.int64) for c, v in nodes.items()}
    imap = {c: {int(v): i for i, v in enumerate(ids[c])} for c in nodes}
    edges = [(int(e.source), int(e.target), float(e.info.weight)) for e in g.edges]
    edge_counts = {}
    for s, t, _ in edges:
        pair = tuple(sorted((cat_of.get(s, "?"), cat_of.get(t, "?"))))
        edge_counts["-".join(pair)] = edge_counts.get("-".join(pair), 0) + 1
    unrepresented = sum(c for k, c in edge_counts.items() if tuple(k.split("-")) not in REPRESENTED_EDGES)

    def rows_of(node_ids, cat):
        m = imap[cat]
        return sorted({m[int(p)] for p in node_ids if int(p) in m})

    T = {}
    O, P, M, R, B, A = (nodes[k] for k in ("objects", "places", "mplaces", "rooms", "building", "agents"))

    # rooms first: places and objects point at them; lowest room row wins when a place has several room parents
    r_child = [rows_of(n.children(), "places") for n in R]
    r_ip, r_ix = _csr(r_child, np.int32)
    place_room = np.full(len(P), -1, dtype=np.int32)
    for ri in range(len(R) - 1, -1, -1):
        place_room[r_child[ri]] = ri
    pos_p = _rows([n.attributes.position for n in P], len(P), 3, np.float64)
    ext_min = _rows([pos_p[c].min(0) if len(c) else [np.nan] * 3 for c in r_child], len(R), 3, np.float32)
    ext_max = _rows([pos_p[c].max(0) if len(c) else [np.nan] * 3 for c in r_child], len(R), 3, np.float32)
    re_ = _edge_table(edges, imap["rooms"])
    T["rooms"] = {"id": ids["rooms"], "pos": _rows([n.attributes.position for n in R], len(R), 3, np.float64),
                  "child_indptr": r_ip, "child_idx": r_ix, "n_children": np.diff(r_ip).astype(np.int32),
                  "ext_min": ext_min, "ext_max": ext_max, "deg": _degree(re_["ij"], len(R))}
    T["room_edges"] = re_

    # GVD places; voxblox block/vertex are uninitialised memory in the saved json, only voxel_pos is kept
    a = [n.attributes for n in P]
    basis = [[tuple(v.voxel_pos) for v in x.voxblox_mesh_connections] for x in a]
    pe = _edge_table(edges, imap["places"])
    par = [rows_of(n.parents(), "places") for n in O]
    n_obj = np.bincount(np.fromiter((p for x in par for p in x), dtype=np.int64), minlength=len(P)).astype(np.int32)
    place_of_agent = np.full(len(A), -1, dtype=np.int32)
    n_agent = np.zeros(len(P), dtype=np.int32)
    for k, n in enumerate(A):
        pp = rows_of(n.parents(), "places")
        if pp:
            place_of_agent[k] = pp[0]
            n_agent[pp[0]] += 1
    T["places"] = {"id": ids["places"], "pos": pos_p, "dist": np.array([x.distance for x in a], dtype=np.float32),
                   "n_basis": np.array([x.num_basis_points for x in a], dtype=np.int16),
                   "last_update_ns": np.array([x.last_update_time_ns for x in a], dtype=np.int64),
                   "is_active": np.array([x.is_active for x in a], dtype=bool),
                   "basis_indptr": _indptr([len(b) for b in basis]), "basis_pos": _rows([p for b in basis for p in b], sum(map(len, basis)), 3, np.float32),
                   "room": place_room, "n_room_parents": np.bincount(r_ix, minlength=len(P)).astype(np.int16),
                   "deg": _degree(pe["ij"], len(P)), "n_obj": n_obj, "n_agent": n_agent}
    T["place_edges"] = pe

    # objects
    a = [n.attributes for n in O]
    mc_ip, mc_ix = _csr([x.mesh_connections for x in a], np.int32)
    par_ip, par_ix = _csr(par, np.int32)
    parent_place = np.array([p[0] if p else -1 for p in par], dtype=np.int32)
    bb = [x.bounding_box for x in a]
    ok = [b.type.name != "INVALID" for b in bb]
    T["objects"] = {"id": ids["objects"], "pos": _rows([x.position for x in a], len(O), 3, np.float64),
                    "label": np.array([_label(x.semantic_label) for x in a], dtype=np.int32),
                    "bbox_type": np.array([int(b.type) for b in bb], dtype=np.int8),
                    "bbox_center": _rows([b.world_P_center if v else [np.nan] * 3 for b, v in zip(bb, ok)], len(O), 3, np.float32),
                    "bbox_dim": _rows([b.dimensions if v else [np.nan] * 3 for b, v in zip(bb, ok)], len(O), 3, np.float32),
                    "last_update_ns": np.array([x.last_update_time_ns for x in a], dtype=np.int64),
                    "is_active": np.array([x.is_active for x in a], dtype=bool),
                    "n_mesh_conn": np.diff(mc_ip).astype(np.int32), "mesh_conn_indptr": mc_ip, "mesh_conn_idx": mc_ix,
                    "n_parents": np.array([len(n.parents()) for n in O], dtype=np.int16), "parent_indptr": par_ip, "parent_idx": par_ix,
                    "parent_place": parent_place,
                    "room": np.where(parent_place >= 0, place_room[np.maximum(parent_place, 0)] if len(P) else -1, -1).astype(np.int32)}

    # 2D mesh places (surface patches on floor / stairs labelled mesh)
    a = [n.attributes for n in M]
    pc_ip, pc_ix = _csr([x.pcl_mesh_connections for x in a], np.int32)
    bpts = [p for x in a for p in x.boundary]
    me = _edge_table(edges, imap["mplaces"])
    T["mplaces"] = {"id": ids["mplaces"], "pos": _rows([x.position for x in a], len(M), 3, np.float64),
                    "label": np.array([_label(x.semantic_label) for x in a], dtype=np.int32),
                    "ellipse_centroid": _rows([x.ellipse_centroid for x in a], len(M), 3, np.float32),
                    "ellipse_expand": _rows([np.asarray(x.ellipse_matrix_expand).ravel() for x in a], len(M), 4, np.float32).reshape(len(M), 2, 2),
                    "is_active": np.array([x.is_active for x in a], dtype=bool), "mesh_conn_indptr": pc_ip, "mesh_conn_idx": pc_ix,
                    "boundary_indptr": _indptr([len(x.boundary) for x in a]), "boundary_xyz": _rows(bpts, len(bpts), 3, np.float32),
                    "deg": _degree(me["ij"], len(M))}
    T["mplace_edges"] = me

    # building: one node, no edges in the saved graphs
    T["building"] = {"id": ids["building"], "pos": _rows([n.attributes.position for n in B], len(B), 3, np.float64),
                     "label": np.array([_label(n.attributes.semantic_label) for n in B], dtype=np.int32)}

    apos = _rows([n.attributes.position for n in A], len(A), 3, np.float64)
    T["traj"], traj_source, traj_match = _trajectory(run_dir, A, apos, place_of_agent, ids["agents"], warns)
    T["mesh"], mesh_flags = _mesh_tables(g, warns)
    nv = len(T["mesh"]["xyz"])

    try:
        names = {int(k): v for k, v in dict(g.get_labelspace(2, 0).labels_to_names).items()}
    except Exception:  # absent labelspace in degenerate graphs
        names = {}
    try:
        bbox_names = {int(v): k for k, v in dsg.BoundingBoxType.__members__.items()}
    except Exception:
        bbox_names = {}
    if unknown:
        warns.append(f"nodes with unknown id category ignored: {unknown}")
    if unrepresented:
        warns.append(f"{unrepresented} edges between categories without a table")
    st = src.stat()
    meta = {"schema": SCHEMA, "run": run_dir.name, "src": {"path": str(src), "size": st.st_size, "mtime_ns": st.st_mtime_ns},
            "counts": {"objects": len(O), "places": len(P), "mplaces": len(M), "rooms": len(R), "agents": len(A), "mesh_vertices": nv,
                       "mesh_faces": len(T["mesh"]["faces"]), "place_edges": len(pe["w"]), "mplace_edges": len(me["w"]),
                       "room_edges": len(re_["w"])},
            "edge_counts": edge_counts, "edges_unrepresented": unrepresented, "mesh_flags": mesh_flags,
            "label_names": names, "bbox_type_names": bbox_names, "no_label": NO_LABEL, "mesh_no_label": MESH_NO_LABEL,
            "traj_source": traj_source, "traj_matches_agents": traj_match, "warnings": warns,
            "build_seconds": {"dsg_load": round(t_load, 2), "total": round(time.time() - t0, 2)}}
    return Snapshot(meta, {k: Table(k, v) for k, v in T.items()})


# ------------------------------------------------------------------------------------------------ bench
def drop_file_cache(path):
    """Evict one file from the Linux page cache (clean pages only, no root needed)."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _child(mode, run, path=None):
    import resource  # unix only, kept out of the module import so Windows python can load snapshots

    t0 = time.time()
    extra = {}
    if mode == "build":
        snap = build_snapshot(resolve_run(run))
        t_work = time.time() - t0
        t1 = time.time()
        snap.save(cache_path(resolve_run(run)))
        extra["save_seconds"] = round(time.time() - t1, 3)
    else:
        snap = Snapshot.load(path or cache_path(resolve_run(run)))
        for tn in TABLES:
            if mode == "load_graph_only" and tn == "mesh":
                continue
            for f in snap._tables[tn].fields():
                getattr(snap._tables[tn], f)
        t_work = time.time() - t0
    print(json.dumps({"mode": mode, "work_seconds": round(t_work, 3),
                      "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1), **extra}))


def bench(run):
    run_dir = resolve_run(run)
    src = run_dir / "backend" / "dsg_with_mesh.json"
    cp = cache_path(run_dir)
    rows = []

    def spawn(mode, cold_path=None, path=None, label=None):
        if cold_path is not None:
            drop_file_cache(cold_path)
        t = time.time()
        cmd = [sys.executable, __file__, "_child", mode, str(run)] + ([str(path)] if path else [])
        out = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.time() - t
        if out.returncode:
            print(out.stderr[-2000:])
            raise SystemExit(out.returncode)
        r = json.loads(out.stdout.strip().splitlines()[-1])
        r["process_wall_seconds"] = round(wall, 3)
        r["page_cache"] = "cold" if cold_path is not None else "warm"
        r["mode"] = label or r["mode"]
        rows.append(r)

    spawn("build", cold_path=src)
    spawn("build")
    spawn("load")
    spawn("load", cold_path=cp)
    spawn("load_graph_only")
    spawn("load_graph_only", cold_path=cp)
    with tempfile.TemporaryDirectory() as td:
        zp = pathlib.Path(td) / "c.npz"
        with Snapshot.load(cp) as s:
            t = time.time()
            s.save(zp, compress=True)
            zsave = time.time() - t
        spawn("load", path=zp, label="load(zip)")
        spawn("load", cold_path=zp, path=zp, label="load(zip)")
        zsize = zp.stat().st_size
    print(f"json {src.stat().st_size / 1e6:.1f} MB, cache {cp.stat().st_size / 1e6:.1f} MB (np.savez), "
          f"{zsize / 1e6:.1f} MB (np.savez_compressed, write {zsave:.1f} s)")
    print(f"{'mode':16s} {'page cache':10s} {'work s':>8s} {'process wall s':>15s} {'peak RSS MB':>12s}")
    for r in rows:
        print(f"{r['mode']:16s} {r['page_cache']:10s} {r['work_seconds']:8.3f} {r['process_wall_seconds']:15.3f} {r['peak_rss_mb']:12.1f}")
    return rows


# ------------------------------------------------------------------------------------------------ synthetic graphs
def _symbol(dsg, ch, i):
    return dsg.NodeSymbol(ch, int(i))


def _synthetic_json(path):
    """Small hand-checkable graph; nodes inserted unsorted on purpose."""
    import spark_dsg as dsg

    g = dsg.DynamicSceneGraph()
    for sym, pos, lab, conn in (("O1", [4.0, 1.0, 0.4], 7, [3]), ("O0", [1.0, 2.0, 0.5], 5, [0, 1, 2])):
        a = dsg.ObjectNodeAttributes()
        a.position, a.semantic_label, a.mesh_connections = np.array(pos), lab, conn
        a.bounding_box = dsg.BoundingBox(np.array([0.5, 0.5, 0.5], dtype=np.float32), np.array(pos, dtype=np.float32))
        g.add_node(dsg.DsgLayers.OBJECTS, dsg.NodeSymbol("O", int(sym[1])), a)
    for i, (pos, d) in enumerate((([1.0, 2.0, 1.0], 0.6), ([4.0, 1.0, 1.0], 0.9), ([6.0, 1.0, 1.0], 0.7))):
        p = dsg.PlaceNodeAttributes()
        p.position, p.distance, p.num_basis_points = np.array(pos), d, 2
        g.add_node(dsg.DsgLayers.PLACES, dsg.NodeSymbol("p", i), p)
    for i, pos in enumerate(([2.0, 2.0, 1.0], [6.0, 1.0, 1.0])):
        r = dsg.RoomNodeAttributes()
        r.position = np.array(pos)
        g.add_node(dsg.DsgLayers.ROOMS, dsg.NodeSymbol("R", i), r)
    for c1, i1, c2, i2 in (("p", 0, "O", 0), ("p", 1, "O", 1), ("R", 0, "p", 0), ("R", 0, "p", 1), ("R", 1, "p", 2), ("p", 0, "p", 1),
                           ("p", 1, "p", 2), ("R", 0, "R", 1)):
        g.insert_edge(dsg.NodeSymbol(c1, i1), dsg.NodeSymbol(c2, i2))
    g.add_layer(3, dsg.PartitionId(1), "MESH_PLACES")
    q = dsg.Place2dNodeAttributes()
    q.position, q.semantic_label, q.pcl_mesh_connections = np.array([3.0, 3.0, 0.0]), 3, [0, 1]
    q.ellipse_centroid, q.ellipse_matrix_compress, q.ellipse_matrix_expand = np.zeros(3), np.eye(2), np.eye(2)
    g.add_node(dsg.LayerKey(3, 1), dsg.NodeSymbol("P", 7), q)
    g.add_layer(2, dsg.PartitionId(97), "AGENTS")
    for i in range(2):
        ag = dsg.AgentNodeAttributes()
        ag.position = np.array([float(i), 0.0, 1.0])
        ag.world_R_body = dsg.Quaternion(1.0, 0.0, 0.0, 0.0)
        g.add_node(dsg.LayerKey(2, 97), dsg.NodeSymbol("a", i), ag)
        g.insert_edge(dsg.NodeSymbol("p", i), dsg.NodeSymbol("a", i))
    m = dsg.Mesh(True, True, True, True)
    m.resize_vertices(4)
    for i, xyz in enumerate([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1)]):
        m.set_pos(i, np.array(xyz, dtype=np.float32))
        c = dsg.Color()
        c.r, c.g, c.b, c.a = 10 * i, 20, 30, 255
        m.set_color(i, c)
        m.set_label(i, i % 3)
        m.set_timestamp(i, 1000 + i)
        m.set_first_seen_timestamp(i, 500 + i)
    m.resize_faces(2)
    m.set_face(0, [0, 1, 2])
    m.set_face(1, [1, 2, 3])
    g.mesh = m
    g.save(str(path))


def _random_graph(dsg, rng):
    """Random degenerate graph: any layer may be empty, nodes may be orphans or multi-parent, odd edge types appear."""
    g = dsg.DynamicSceneGraph()
    n = {k: int(rng.integers(0, hi)) for k, hi in (("O", 5), ("p", 7), ("R", 4), ("P", 4), ("a", 4), ("B", 2))}
    ids = {k: [int(i) for i in rng.choice(40, size=v, replace=False)] for k, v in n.items()}
    mesh_kind = ["none", "empty", "full", "labels_only", "stamps_only", "bare"][int(rng.integers(0, 6))]
    nv = int(rng.integers(1, 7))
    hi = nv if mesh_kind not in ("none", "empty") else 1
    g.add_layer(3, dsg.PartitionId(1), "MESH_PLACES")
    g.add_layer(2, dsg.PartitionId(97), "AGENTS")
    for i in ids["O"]:
        a = dsg.ObjectNodeAttributes()
        a.position = rng.normal(size=3)
        a.semantic_label = int(rng.choice([5, 7, 4294967295]))
        a.mesh_connections = [int(x) for x in rng.integers(0, hi, size=int(rng.integers(0, 4)))] if mesh_kind not in ("none", "empty") else []
        if rng.random() < 0.7:
            a.bounding_box = dsg.BoundingBox(np.abs(rng.normal(size=3)).astype(np.float32), rng.normal(size=3).astype(np.float32))
        g.add_node(dsg.DsgLayers.OBJECTS, _symbol(dsg, "O", i), a)
    for i in ids["p"]:
        a = dsg.PlaceNodeAttributes()
        a.position, a.distance, a.num_basis_points = rng.normal(size=3), float(rng.random()), int(rng.integers(0, 6))
        g.add_node(dsg.DsgLayers.PLACES, _symbol(dsg, "p", i), a)
    for i in ids["R"]:
        a = dsg.RoomNodeAttributes()
        a.position = rng.normal(size=3)
        g.add_node(dsg.DsgLayers.ROOMS, _symbol(dsg, "R", i), a)
    for i in ids["P"]:
        a = dsg.Place2dNodeAttributes()
        a.position, a.semantic_label = rng.normal(size=3), int(rng.choice([3, 15, 4294967295]))
        a.ellipse_centroid, a.ellipse_matrix_compress, a.ellipse_matrix_expand = np.zeros(3), np.eye(2), np.eye(2)
        g.add_node(dsg.LayerKey(3, 1), _symbol(dsg, "P", i), a)
    for i in ids["a"]:
        a = dsg.AgentNodeAttributes()
        a.position = rng.normal(size=3)
        a.world_R_body = dsg.Quaternion(1.0, 0.0, 0.0, 0.0)  # default is uninitialised; NaN is written as null and cannot be reloaded
        g.add_node(dsg.LayerKey(2, 97), _symbol(dsg, "a", i), a)
    for i in ids["B"]:
        g.add_node(dsg.DsgLayers.BUILDINGS, _symbol(dsg, "B", i), dsg.SemanticNodeAttributes())

    def link(c1, i1, c2, i2):
        try:
            g.insert_edge(_symbol(dsg, c1, i1), _symbol(dsg, c2, i2))
        except Exception:
            pass  # generator-side: duplicate or rejected edge

    for i in ids["O"]:
        if ids["p"] and rng.random() < 0.8:
            for j in rng.choice(ids["p"], size=min(len(ids["p"]), int(rng.integers(1, 4))), replace=False):
                link("p", j, "O", i)
        if ids["R"] and rng.random() < 0.2:
            link("R", rng.choice(ids["R"]), "O", i)
    for i in ids["p"]:
        if ids["R"] and rng.random() < 0.85:
            for j in rng.choice(ids["R"], size=min(len(ids["R"]), 1 + int(rng.random() < 0.15)), replace=False):
                link("R", j, "p", i)
        for j in rng.choice(ids["p"], size=min(len(ids["p"]), 2), replace=False):
            if j != i:
                link("p", i, "p", j)
    for i in ids["a"]:
        if ids["p"] and rng.random() < 0.7:
            link("p", rng.choice(ids["p"]), "a", i)
    if len(ids["R"]) > 1:
        link("R", ids["R"][0], "R", ids["R"][1])
    if len(ids["P"]) > 1:
        link("P", ids["P"][0], "P", ids["P"][1])
    if ids["B"] and ids["R"]:
        link("B", ids["B"][0], "R", ids["R"][0])
    if mesh_kind != "none":
        # Mesh(has_colors, has_timestamps, has_labels, has_first_seen)
        flags = {"empty": (True,) * 4, "full": (True,) * 4, "labels_only": (False, False, True, False),
                 "stamps_only": (False, True, False, True), "bare": (False,) * 4}[mesh_kind]
        m = dsg.Mesh(*flags)
        if mesh_kind != "empty":
            m.resize_vertices(nv)
            for i in range(nv):
                m.set_pos(i, rng.normal(size=3).astype(np.float32))
            if nv >= 3:
                m.resize_faces(1)
                m.set_face(0, [0, 1, 2])
            if mesh_kind in ("full", "labels_only"):
                for i in range(nv):
                    m.set_label(i, int(rng.choice([0, 3, 19, 4294967295])))
            if mesh_kind in ("full", "stamps_only"):
                for i in range(nv):
                    m.set_timestamp(i, int(i))
                    m.set_first_seen_timestamp(i, int(i))
        g.mesh = m
    return g, len(ids["a"]), mesh_kind


def _nonfinite_graph_json(path, what):
    import spark_dsg as dsg

    g = dsg.DynamicSceneGraph()
    if what == "position":
        a = dsg.ObjectNodeAttributes()
        a.position = np.array([np.nan, 0.0, 0.0])
        g.add_node(dsg.DsgLayers.OBJECTS, dsg.NodeSymbol("O", 0), a)
    else:
        a = dsg.AgentNodeAttributes()
        a.position = np.zeros(3)
        a.world_R_body = dsg.Quaternion(np.nan, 0.0, 0.0, 0.0)
        g.add_layer(2, dsg.PartitionId(97), "AGENTS")
        g.add_node(dsg.LayerKey(2, 97), dsg.NodeSymbol("a", 0), a)
    g.save(str(path))


def _write_run(run, g, csv_text=None):
    (run / "backend").mkdir(parents=True, exist_ok=True)
    g.save(str(run / "backend" / "dsg_with_mesh.json"))
    if csv_text is not None:
        (run / "backend" / "trajectory.csv").write_text(csv_text)


def _csv_variants(n_agents, rng):
    hdr = "timestamp[ns],x,y,z,qw,qx,qy,qz\n"
    rows = lambda k: "".join(f"{i * 10**9},{i},0,0,1,0,0,0\n" for i in range(k))
    return [None, "", hdr, hdr + rows(n_agents), hdr + rows(n_agents + 2), "t,x,y\n1,2,3\n", "garbage\nnot,a,number\n"][int(rng.integers(0, 7))]


def edge_cases(n_random=60, seed=0):
    """Degenerate inputs must build, pass the invariants and round-trip through the cache; unusable inputs raise SnapshotError."""
    import spark_dsg as dsg

    rng = np.random.default_rng(seed)
    kinds = {}
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        cache = td / "cache"
        for k in range(n_random + 1):
            if k == 0:
                g, na, mk = dsg.DynamicSceneGraph(), 0, "none"  # completely empty graph
            else:
                g, na, mk = _random_graph(dsg, rng)
            run = td / f"r{k}"
            _write_run(run, g, _csv_variants(na, rng))
            s = Snapshot.open(run, cache_root=cache)
            bad = s.check()
            assert not bad, (k, mk, bad)
            s2 = Snapshot.open(run, cache_root=cache)
            assert s2.from_cache
            for tn in TABLES:
                for f in s._tables[tn].fields():
                    assert np.array_equal(getattr(s._tables[tn], f), getattr(s2._tables[tn], f), equal_nan=True), (k, tn, f)
            s2.close()
            c = s.meta["counts"]
            for name, val in (("zero_objects", c["objects"] == 0), ("zero_places", c["places"] == 0), ("zero_rooms", c["rooms"] == 0),
                              ("orphan_objects", bool(len(s.objects) and (s.objects.parent_place < 0).any())),
                              ("orphan_places", bool(len(s.places) and (s.places.room < 0).any())),
                              ("multi_parent_objects", bool(len(s.objects) and (s.objects.n_parents > 1).any())),
                              ("multi_room_places", bool(len(s.places) and (s.places.n_room_parents > 1).any())),
                              ("unset_object_label", bool(len(s.objects) and (s.objects.label < 0).any())),
                              ("invalid_bbox", bool(len(s.objects) and np.isnan(s.objects.bbox_dim).any())),
                              ("unrepresented_edges", s.meta["edges_unrepresented"] > 0),
                              ("mesh_" + mk, True), ("traj_" + s.meta["traj_source"], True),
                              ("traj_mismatch", s.meta["traj_matches_agents"] is False),
                              ("mesh_missing_channel", any("mesh channels absent" in w for w in s.meta["warnings"]))):
                kinds[name] = kinds.get(name, 0) + int(val)
            s.close()
        # unusable inputs
        bad_run = td / "bad"
        (bad_run / "backend").mkdir(parents=True)
        for label, setup in (("missing json", lambda: None), ("truncated json", lambda: (bad_run / "backend" / "dsg_with_mesh.json").write_text('{"nodes": [')),
                             ("empty file", lambda: (bad_run / "backend" / "dsg_with_mesh.json").write_text("")),
                             ("binary junk", lambda: (bad_run / "backend" / "dsg_with_mesh.json").write_bytes(os.urandom(512)))):
            setup()
            try:
                Snapshot.open(bad_run, cache_root=cache)
            except SnapshotError:
                continue
            raise AssertionError(f"{label}: expected SnapshotError")
        # NaN attributes are written as json null: either a typed load error or a snapshot flagged as non-finite
        for what in ("position", "quaternion"):
            run = td / f"nan_{what}"
            (run / "backend").mkdir(parents=True)
            _nonfinite_graph_json(run / "backend" / "dsg_with_mesh.json", what)
            try:
                s = Snapshot.open(run, cache_root=cache)
            except SnapshotError:
                kinds[f"nan_{what}: typed load error"] = 1
            else:
                assert any("non-finite" in b for b in s.check()), what
                kinds[f"nan_{what}: built, flagged non-finite"] = 1
                s.close()
        # damaged cache file is rebuilt, not fatal
        good = td / "r1"
        cp = cache_path(good, cache)
        cp.write_bytes(cp.read_bytes()[:1000])
        s = Snapshot.open(good, cache_root=cache)
        assert not s.from_cache and not s.check()
        s.close()
    return kinds


# ------------------------------------------------------------------------------------------------ self test
def selftest(run=None):
    import spark_dsg as dsg

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        (td / "run" / "backend").mkdir(parents=True)
        src = td / "run" / "backend" / "dsg_with_mesh.json"
        _synthetic_json(src)
        (td / "run" / "backend" / "trajectory.csv").write_text(
            "timestamp[ns],x,y,z,qw,qx,qy,qz\n499300019999,0,0,1,1,0,0,0\n501100019999,1,0,1,1,0,0,0\n")
        s = Snapshot.open(td / "run", cache_root=td / "cache")
        assert not s.check(), s.check()
        o, p, r, mp, mesh, tr = s.objects, s.places, s.rooms, s.mplaces, s.mesh, s.traj
        g = dsg.DynamicSceneGraph.load(str(src))
        assert list(o.id) == sorted(o.id) and list(o.label) == [5, 7], o.label  # sorted by id although inserted unsorted
        assert symbol(o.id[0]) == "O0" and symbol(p.id[2]) == "p2"
        for i in range(2):
            bb = g.get_node(int(o.id[i])).attributes.bounding_box
            assert np.allclose(o.bbox_dim[i], bb.dimensions) and np.allclose(o.bbox_center[i], bb.world_P_center) and o.bbox_type[i] == int(bb.type)
        assert list(o.parent_place) == [0, 1] and list(o.room) == [0, 0] and list(o.n_parents) == [1, 1]
        assert list(o.csr("parent_indptr", "parent_idx", 1)) == [1]
        assert list(o.n_mesh_conn) == [3, 1] and list(o.csr("mesh_conn_indptr", "mesh_conn_idx", 0)) == [0, 1, 2]
        assert len(p) == 3 and np.allclose(p.dist, [0.6, 0.9, 0.7]) and list(p.room) == [0, 0, 1] and list(p.deg) == [1, 2, 1]
        assert list(p.n_obj) == [1, 1, 0] and list(p.n_agent) == [1, 1, 0] and list(p.n_room_parents) == [1, 1, 1]
        assert len(r) == 2 and list(r.n_children) == [2, 1] and list(r.csr("child_indptr", "child_idx", 0)) == [0, 1] and list(r.deg) == [1, 1]
        assert np.allclose(r.ext_min[0], [1.0, 1.0, 1.0]) and np.allclose(r.ext_max[0], [4.0, 2.0, 1.0])
        assert len(mp) == 1 and mp.label[0] == 3 and list(mp.csr("mesh_conn_indptr", "mesh_conn_idx", 0)) == [0, 1]
        assert mesh.xyz.shape == (4, 3) and mesh.faces.shape == (2, 3) and list(mesh.label) == [0, 1, 2, 0]
        assert list(mesh.stamp_ns) == [1000, 1001, 1002, 1003] and list(mesh.first_seen_ns) == [500, 501, 502, 503]
        assert list(mesh.rgb[:, 0]) == [0, 10, 20, 30] and tuple(mesh.rgb[0, 1:]) == (20, 30)
        assert list(tr.t_ns) == [499300019999, 501100019999] and list(tr.place) == [0, 1] and s.meta["traj_matches_agents"] is True
        assert s.meta["edge_counts"] == {"O-p": 2, "R-p": 3, "R-R": 1, "p-p": 2, "a-p": 2} and s.meta["edges_unrepresented"] == 0
        # second open must hit the cache and return identical arrays
        s2 = Snapshot.open(td / "run", cache_root=td / "cache")
        assert s2.from_cache and s2.meta["build_seconds"] == s.meta["build_seconds"]
        for tn in TABLES:
            for f in s._tables[tn].fields():
                assert np.array_equal(getattr(s._tables[tn], f), getattr(s2._tables[tn], f)), (tn, f)
        # a changed source file invalidates the cache
        os.utime(src, ns=(src.stat().st_atime_ns, src.stat().st_mtime_ns + 10**9))
        assert Snapshot.open(td / "run", cache_root=td / "cache").meta["src"]["mtime_ns"] == src.stat().st_mtime_ns
    print("synthetic self-test OK")
    kinds = edge_cases()
    print("degenerate-graph self-test OK (61 graphs: builds, invariants, cache round trip, errors typed); coverage:",
          ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
    if run is None:
        return

    t0 = time.time()
    s = Snapshot.open(run)
    print(s.summary(), f"(open {time.time() - t0:.2f}s)")
    assert not s.check(), s.check()
    g = dsg.DynamicSceneGraph.load(str(resolve_run(run) / "backend" / "dsg_with_mesh.json"))
    c = s.meta["counts"]
    assert c["objects"] == g.get_layer(2, 0).num_nodes() and c["places"] == g.get_layer(3, 0).num_nodes()
    assert c["rooms"] == g.get_layer(4, 0).num_nodes() and c["mesh_vertices"] == g.mesh.num_vertices()
    for i in sorted({0, len(s.objects) // 2, len(s.objects) - 1}) if len(s.objects) else []:
        n = g.get_node(int(s.objects.id[i]))
        assert np.array_equal(n.attributes.position, s.objects.pos[i]) and n.attributes.semantic_label == s.objects.label[i]
        assert list(n.attributes.mesh_connections) == list(s.objects.csr("mesh_conn_indptr", "mesh_conn_idx", i))
        pa = sorted(n.parents())
        assert (int(s.places.id[s.objects.parent_place[i]]) if s.objects.parent_place[i] >= 0 else None) == (pa[0] if pa else None)
        assert [int(s.places.id[j]) for j in s.objects.csr("parent_indptr", "parent_idx", i)] == pa
    for i in sorted({0, len(s.places) // 2, len(s.places) - 1}) if len(s.places) else []:
        n = g.get_node(int(s.places.id[i]))
        assert abs(n.attributes.distance - s.places.dist[i]) < 1e-6
        assert (int(s.rooms.id[s.places.room[i]]) if s.places.room[i] >= 0 else None) == (next(iter(n.parents())) if n.parents() else None)
    assert np.array_equal(s.mesh.label, np.array(g.mesh.get_labels(), dtype=np.uint16))
    assert np.allclose(s.mesh.xyz[::997], g.mesh.get_vertices()[:3].T[::997].astype(np.float32))
    assert s.meta["traj_matches_agents"] is True and len(s.traj.t_ns) == c["agents"]
    assert len(s.place_edges.w) == g.get_layer(3, 0).num_edges() and len(s.mplace_edges.w) == g.get_layer(3, 1).num_edges()
    assert s.meta["edges_unrepresented"] == 0 and sum(s.meta["edge_counts"].values()) == g.num_edges()
    print(f"real-run self-test OK ({run})")


# ------------------------------------------------------------------------------------------------ exhaustive check against the DSG API
def verify(run):
    """Every stored value of one run against spark_dsg (all nodes, edges, mesh vertices); raises on the first 10 mismatches."""
    import spark_dsg as dsg

    run_dir = resolve_run(run)
    g = dsg.DynamicSceneGraph.load(str(run_dir / "backend" / "dsg_with_mesh.json"))
    with Snapshot.open(run_dir) as s:
        s.materialize()
    nodes = {c: [] for c in CAT}
    for key in g.layer_keys:
        for n in g.get_layer(key.layer, key.partition).nodes:
            nodes.setdefault(n.id.category, []).append(n)
    for v in nodes.values():
        v.sort(key=lambda n: int(n.id.value))
    O, P, M, R, B, A = (nodes[c] for c in "OpPRBa")
    o, p, r, mp, b, tr, mesh = s.objects, s.places, s.rooms, s.mplaces, s.building, s.traj, s.mesh
    bad = []

    def expect(cond, msg):
        if not cond:
            bad.append(msg)

    pid = {int(n.id.value): i for i, n in enumerate(P)}
    rid = {int(n.id.value): i for i, n in enumerate(R)}

    def room_row(parents):
        rp = sorted(int(x) for x in parents if int(x) in rid)
        return len(rp), (rid[rp[0]] if rp else -1)

    for tab, ns, cats in ((o, O, "objects"), (p, P, "places"), (r, R, "rooms"), (mp, M, "mplaces"), (b, B, "building")):
        expect([int(n.id.value) for n in ns] == tab.id.tolist(), f"{cats} ids")
        expect(np.array_equal(np.array([n.attributes.position for n in ns]).reshape(-1, 3), tab.pos), f"{cats} pos")
    for i, n in enumerate(O):
        a = n.attributes
        bb = a.bounding_box
        expect(_label(a.semantic_label) == o.label[i], f"object {i} label")
        expect(bb.type.name == "INVALID" and np.isnan(o.bbox_dim[i]).all() or np.allclose(bb.dimensions, o.bbox_dim[i]) and np.allclose(bb.world_P_center, o.bbox_center[i]), f"object {i} bbox")
        expect(list(a.mesh_connections) == o.csr("mesh_conn_indptr", "mesh_conn_idx", i).tolist(), f"object {i} mesh_connections")
        par = sorted(int(x) for x in n.parents())
        pp = [x for x in par if x in pid]
        expect([int(p.id[j]) for j in o.csr("parent_indptr", "parent_idx", i)] == pp and o.n_parents[i] == len(par), f"object {i} parents")
        expect(o.parent_place[i] == (pid[pp[0]] if pp else -1), f"object {i} parent_place")
        expect(o.room[i] == (room_row(g.get_node(pp[0]).parents())[1] if pp else -1), f"object {i} room")
        expect(bool(a.is_active) == o.is_active[i] and a.last_update_time_ns == o.last_update_ns[i], f"object {i} flags")
    for i, n in enumerate(P):
        a = n.attributes
        nrp, room = room_row(n.parents())
        ch = [int(x) for x in n.children()]
        basis = np.array([tuple(v.voxel_pos) for v in a.voxblox_mesh_connections], dtype=np.float32).reshape(-1, 3)
        expect(np.float32(a.distance) == p.dist[i] and a.num_basis_points == p.n_basis[i], f"place {i} distance / basis count")
        expect(np.array_equal(basis, p.csr("basis_indptr", "basis_pos", i)), f"place {i} basis points")
        expect(p.n_room_parents[i] == nrp and p.room[i] == room, f"place {i} room")
        expect(p.deg[i] == len(n.siblings()), f"place {i} degree")
        expect(p.n_obj[i] == sum(chr(x >> 56) == "O" for x in ch) and p.n_agent[i] == sum(chr(x >> 56) == "a" for x in ch), f"place {i} children")
        expect(bool(a.is_active) == p.is_active[i] and a.last_update_time_ns == p.last_update_ns[i], f"place {i} flags")
    for i, n in enumerate(R):
        expect(sorted(int(x) for x in n.children() if int(x) in pid) == sorted(int(p.id[j]) for j in r.csr("child_indptr", "child_idx", i)), f"room {i} children")
        expect(r.deg[i] == len(n.siblings()), f"room {i} degree")
    for i, n in enumerate(M):
        a = n.attributes
        expect(_label(a.semantic_label) == mp.label[i] and list(a.pcl_mesh_connections) == mp.csr("mesh_conn_indptr", "mesh_conn_idx", i).tolist(), f"mplace {i} label / mesh connections")
        expect(np.array_equal(np.array(a.boundary, dtype=np.float32).reshape(-1, 3), mp.csr("boundary_indptr", "boundary_xyz", i)), f"mplace {i} boundary")
        expect(np.allclose(a.ellipse_centroid, mp.ellipse_centroid[i]) and np.allclose(np.asarray(a.ellipse_matrix_expand), mp.ellipse_expand[i]), f"mplace {i} ellipse")
        expect(mp.deg[i] == len(n.siblings()), f"mplace {i} degree")
    for i, n in enumerate(B):
        expect(_label(n.attributes.semantic_label) == b.label[i], f"building {i} label")
    # edges: multiset of (source id, target id, weight) per table
    for tab, ns, ch in ((s.place_edges, P, "p"), (s.mplace_edges, M, "P"), (s.room_edges, R, "R")):
        ids = np.array([int(n.id.value) for n in ns], dtype=np.int64)
        want = sorted((int(e.source), int(e.target), float(np.float32(e.info.weight))) for e in g.edges
                      if chr(int(e.source) >> 56) == ch and chr(int(e.target) >> 56) == ch)
        have = sorted((int(ids[i]), int(ids[j]), float(w)) for (i, j), w in zip(tab.ij, tab.w))
        expect(want == have, f"{ch}-{ch} edges")
    # trajectory
    if s.meta["traj_source"] == "trajectory.csv" and s.meta["traj_matches_agents"]:
        ts = np.array([int(round(n.attributes.timestamp.total_seconds() * 1e9)) for n in A], dtype=np.int64)
        expect(len(ts) == len(tr.t_ns) and (len(ts) == 0 or np.abs(ts - tr.t_ns).max() <= 2000), "trajectory timestamps")
        expect(len(A) == 0 or np.abs(np.array([n.attributes.position for n in A]) - tr.pos).max() < 1e-3, "trajectory positions")
        expect([(pid[int(x)] if int(x) in pid else -1) for n in A for x in sorted(n.parents())[:1]] == [int(v) for v in tr.place[[bool(n.parents()) for n in A]]], "trajectory places")
    # mesh
    m = g.mesh
    if m is not None and m.num_vertices():
        V = m.get_vertices()
        nv = m.num_vertices()
        expect(np.array_equal(mesh.xyz, V[:3].T.astype(np.float32)) and np.array_equal(mesh.label, np.minimum(np.array(m.get_labels(), dtype=np.int64), MESH_NO_LABEL)), "mesh vertices / labels")
        expect(np.array_equal(mesh.faces, m.get_faces().T) and np.abs(mesh.rgb.astype(float) - V[3:6].T * 255).max() < 0.51, "mesh faces / colours")
        expect(np.array_equal(mesh.stamp_ns, [m.timestamp(i) for i in range(nv)]) and np.array_equal(mesh.first_seen_ns, [m.first_seen_timestamp(i) for i in range(nv)]), "mesh timestamps")
    c = s.meta["counts"]
    print(f"{run_dir.name}: " + ("MISMATCH " + "; ".join(bad[:10]) if bad else f"every stored value equals the DSG API ({c['objects']} objects, {c['places']} places, "
                                   f"{c['mplaces']} mesh places, {c['rooms']} rooms, {c['agents']} poses, {c['mesh_vertices']} mesh vertices, {sum(len(t.w) for t in (s.place_edges, s.mplace_edges, s.room_edges))} edges)"))
    if bad:
        raise AssertionError(f"{run_dir.name}: {len(bad)} mismatches")
    return True


# ------------------------------------------------------------------------------------------------ mutations of a real graph
def mutate(run):
    """Real-size degenerate graphs: JSON surgery on one run, then Snapshot build + invariants for each variant."""
    import resource

    src = resolve_run(run) / "backend" / "dsg_with_mesh.json"
    doc = json.loads(src.read_text())

    def cat(nid):  # raw id -> symbol char
        return chr(int(nid) >> 56)

    def drop(node_cats=(), edge_cats=(), mesh=True):
        d = dict(doc)
        d["nodes"] = [n for n in doc["nodes"] if cat(n["id"]) not in node_cats]
        keep = {n["id"] for n in d["nodes"]}
        d["edges"] = [e for e in doc["edges"] if e["source"] in keep and e["target"] in keep
                      and tuple(sorted((cat(e["source"]), cat(e["target"])))) not in edge_cats]
        if not mesh:
            d.pop("mesh", None)
        return d

    variants = {"no_rooms": dict(node_cats=("R",)), "no_objects": dict(node_cats=("O",)), "no_places": dict(node_cats=("p",)),
                "no_mesh": dict(mesh=False), "no_room_place_edges": dict(edge_cats=(("R", "p"),)),
                "no_object_place_edges": dict(edge_cats=(("O", "p"),)), "only_building": dict(node_cats=("O", "p", "P", "R", "a"), mesh=False)}
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, kw in variants.items():
        td = pathlib.Path(tempfile.mkdtemp(prefix="mut_", dir=str(CACHE_ROOT)))
        try:
            (td / "backend").mkdir()
            (td / "backend" / "dsg_with_mesh.json").write_text(json.dumps(drop(**kw)))
            tj = resolve_run(run) / "backend" / "trajectory.csv"
            if tj.exists():
                shutil.copy(tj, td / "backend" / "trajectory.csv")
            t = time.time()
            with Snapshot.open(td, cache_root=td / "cache") as s:
                rows.append((name, s.meta["counts"], s.check(), s.meta["warnings"], round(time.time() - t, 1)))
        finally:
            shutil.rmtree(td, ignore_errors=True)
        gc.collect()
    for name, c, bad, w, secs in rows:
        print(f"{name:22s} {secs:5.1f}s  objects={c['objects']:4d} places={c['places']:5d} mplaces={c['mplaces']:5d} rooms={c['rooms']:3d} "
              f"agents={c['agents']:4d} mesh_v={c['mesh_vertices']:7d} | invariants: {'OK' if not bad else bad}")
        for x in w:
            print(f"{'':22s} warning: {x}")
    print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
    return rows


# ------------------------------------------------------------------------------------------------ survey of development runs
def _rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    return float("nan")


def _status_log(path):
    """name -> status for the dev runs only is looked up by the caller; the whole log is not printed."""
    out = {}
    try:
        for line in pathlib.Path(path).read_text(encoding="utf8").splitlines():
            try:
                d = json.loads(line)
                out[d["name"]] = d.get("status")
            except (ValueError, KeyError):
                continue
    except OSError:
        pass
    return out


def finished_dev_runs(root=RUNS_ROOT, min_age_s=600):
    """Development-split runs (reference, repeat, benign, seed 0) with a complete json, not marked failed in the sweep log and
    not still being written. Returns (runs, skipped counts); test-split runs are only counted, never listed."""
    sys.path.insert(0, str(REPO_SRC))
    from layer_attribution import runs as runs_mod

    runs = runs_mod.index_runs(root)
    in_dev = [r for r in runs if r.split in runs_mod.DEV_SPLITS]
    log = _status_log(STATUS_LOG)
    skipped = {"not development split (test, or still running without manifest)": len(runs) - len(in_dev),
               "development, json missing": len([r for r in in_dev if not r.complete]), "development, status not ok / still being written": 0}
    out = []
    for r in runs_mod.select(runs, runs_mod.DEV_SPLITS):
        if log.get(r.name, "ok") != "ok" or (r.name not in log and time.time() - r.dsg_path.stat().st_mtime < min_age_s):
            skipped["development, status not ok / still being written"] += 1
        else:
            out.append(r)
    return out, skipped


def survey(root=RUNS_ROOT, out_dir=None):
    import resource

    dev, skipped = finished_dev_runs(root)
    rows = []
    for r in dev:
        row = {"run": r.name, "split": r.split, "fault": r.fault or "-", "sev": r.severity, "json_mb": round(r.dsg_path.stat().st_size / 1e6, 1)}
        try:
            t = time.perf_counter()
            with Snapshot.open(r.path) as s:
                row["open_s"], row["from_cache"] = round(time.perf_counter() - t, 2), s.from_cache
                row["build_s"] = s.meta["build_seconds"]["total"]
                bad = s.check()
                row.update(s.meta["counts"])
                row["orph_obj"] = int((s.objects.parent_place < 0).sum())
                row["multi_par"] = int((s.objects.n_parents > 1).sum())
                row["orph_pl"] = int((s.places.room < 0).sum())
                row["multi_room"] = int((s.places.n_room_parents > 1).sum())
                row["unset_lbl"] = int((s.objects.label < 0).sum())
                row["bad_bbox"] = int(np.isnan(s.objects.bbox_dim).any(axis=1).sum())
                row["rooms_no_obj"] = int((np.bincount(s.objects.room[s.objects.room >= 0], minlength=len(s.rooms)) == 0).sum()) if len(s.rooms) else 0
                row["traj_ok"] = s.meta["traj_matches_agents"]
                row["warnings"] = s.meta["warnings"]
                row["edges_unrep"] = s.meta["edges_unrepresented"]
                row["invariants"] = "OK" if not bad else bad
            t = time.perf_counter()
            with Snapshot.open(r.path) as s2:
                s2.materialize()
            row["load_ms"] = round((time.perf_counter() - t) * 1000)
            row["rss_mb"] = round(_rss_mb())
            row["status"] = "ok"
        except Exception as e:  # the survey must report, not die
            row["status"] = f"FAILED {type(e).__name__}: {str(e)[:120]}"
        rows.append(row)
        gc.collect()
    hdr = ("run", "split", "fault", "sev", "json_mb", "objects", "places", "mplaces", "rooms", "agents", "mesh_vertices", "build_s", "load_ms",
           "orph_obj", "multi_par", "orph_pl", "multi_room", "unset_lbl", "bad_bbox", "rooms_no_obj", "invariants", "rss_mb")
    lines = [" ".join(f"{h:>8s}" if h != "run" else f"{h:34s}" for h in hdr)]
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"{r['run']:34s} {r['status']}")
            continue
        lines.append(" ".join(f"{str(r.get(h, '')):>8s}" if h != "run" else f"{r[h]:34s}" for h in hdr))
    ok = [r for r in rows if r["status"] == "ok"]
    lines.append("")
    lines.append(f"{len(rows)} development runs loaded, {len(ok)} ok, {len(rows) - len(ok)} failed; "
                 f"invariant violations in {sum(r['invariants'] != 'OK' for r in ok)} runs; skipped (counts only): {skipped}")
    for k in ("objects", "places", "mplaces", "rooms", "agents", "mesh_vertices", "build_s", "load_ms"):
        v = [r[k] for r in ok]
        if v:
            lines.append(f"  {k:14s} min {min(v):>9} median {float(np.median(v)):>11.1f} max {max(v):>9}")
    w = {x for r in ok for x in r["warnings"]}
    lines.append(f"  warnings emitted: {sorted(w) if w else 'none'}; runs with unrepresented edges: {sum(r['edges_unrep'] > 0 for r in ok)}; "
                 f"traj mismatch: {sum(r['traj_ok'] is False for r in ok)}")
    lines.append(f"  survey process peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB; RSS after last run {rows[-1].get('rss_mb') if rows else '-'} MB")
    lines.append("  columns: build_s = json load + extraction (first build); load_ms = cache open + all arrays in memory (warm page cache); orph_obj = objects without "
                 "parent place; multi_par = objects with >1 parent; orph_pl = places without room; multi_room = places with >1 room; unset_lbl = objects with label unset; bad_bbox = objects with "
                 "invalid bbox; rooms_no_obj = rooms without objects; rss_mb = process RSS after the run")
    print("\n".join(lines))
    if out_dir:
        out_dir = pathlib.Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "survey_dev_runs.txt").write_text("\n".join(lines) + "\n", encoding="utf8")
        (out_dir / "survey_dev_runs.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf8")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("run")
    b.add_argument("--rebuild", action="store_true")
    sub.add_parser("bench").add_argument("run")
    sub.add_parser("mutate").add_argument("run")
    sub.add_parser("verify").add_argument("runs", nargs="+")
    sub.add_parser("selftest").add_argument("--run", default=None)
    sv = sub.add_parser("survey")
    sv.add_argument("--root", type=pathlib.Path, default=RUNS_ROOT)
    sv.add_argument("--out-dir", type=pathlib.Path, default=None)
    ch = sub.add_parser("_child")
    ch.add_argument("mode")
    ch.add_argument("run")
    ch.add_argument("path", nargs="?", default=None)
    args = ap.parse_args()
    if args.cmd == "build":
        t = time.time()
        s = Snapshot.open(args.run, rebuild=args.rebuild)
        print(s.summary(), f"| {time.time() - t:.2f}s | cache {cache_path(resolve_run(args.run))}")
    elif args.cmd == "bench":
        bench(args.run)
    elif args.cmd == "mutate":
        mutate(args.run)
    elif args.cmd == "verify":
        for run in args.runs:
            verify(run)
    elif args.cmd == "survey":
        survey(args.root, args.out_dir)
    elif args.cmd == "selftest":
        selftest(args.run)
    else:
        _child(args.mode, args.run, args.path)


if __name__ == "__main__":
    sys.exit(main())
