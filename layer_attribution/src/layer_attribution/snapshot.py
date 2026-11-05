"""Snapshot: numpy view of one saved Hydra run, built once from backend/dsg_with_mesh.json and cached as .npz.

build  needs spark_dsg (WSL venv; imported lazily, only on the build path)
load   needs numpy only (also works from Windows python through \\\\wsl.localhost\\...); arrays are read lazily
Rows of every node table are sorted by raw NodeId (symbol char << 56 | index). Relations are indices into the target
table (-1 = none) or CSR pairs. Unset labels are -1. A missing or empty layer gives empty arrays with the full field set,
so feature code needs no existence checks. Ground truth (injection_manifest.json) is never read here.

tables (arrays; "->T" = row index into table T; csr "<x>_{indptr,idx}" = ragged lists):
    objects   id pos label bbox_type bbox_center bbox_dim bbox_rot last_update_ns is_active n_mesh_conn mesh_conn_{indptr,idx}->mesh
              n_parents parent_{indptr,idx}->places parent_place->places room->rooms
    places    id pos dist n_basis basis_{indptr,pos} last_update_ns is_active room->rooms n_room_parents deg n_obj n_agent
    mplaces   id pos label ellipse_centroid ellipse_expand is_active mesh_conn_{indptr,idx}->mesh boundary_{indptr,xyz} deg
    rooms     id pos child_{indptr,idx}->places n_children ext_min ext_max has_parent deg
    place_edges / mplace_edges / room_edges   ij (E,2) rows of the own table, w
    building  id pos label     traj  t_ns pos quat_wxyz agent_id place->places
    mesh      xyz rgb label (uint16, 65535 = none) stamp_ns first_seen_ns faces (F,3)
    stats     layer_{2,3,4}_last = last row of backend/layer_N_statistics.csv (cumulative churn counters), NaN when absent
meta: counts, graph_counts, edge_counts by category pair, edges_unrepresented, mesh_flags, label_names, traj_source, warnings,
      build_seconds, src (size and mtime of the json and of the side files = cache key)
env: LA_RUNS_ROOT, LA_SNAPSHOT_CACHE override the default roots (needed from Windows python over the UNC path).

usage (WSL, via scripts/wsl_py.sh):
    python -m layer_attribution.snapshot build RUN [--rebuild] [--cache-dir DIR]   RUN = run name under ~/runs/uh2 or a run dir
    python -m layer_attribution.snapshot check RUN [RUN ...] [--cache-dir DIR]     structural invariants of the (cached or built) snapshot
"""

import argparse
import json
import os
import pathlib
import sys
import time
import warnings
import zipfile

import numpy as np

SCHEMA = 3
RUNS_ROOT = pathlib.Path(os.environ.get("LA_RUNS_ROOT", "~/runs/uh2")).expanduser()  # Windows: \\wsl.localhost\Ubuntu-24.04\home\hydra\runs\uh2
CACHE_ROOT = pathlib.Path(os.environ.get("LA_SNAPSHOT_CACHE", "~/runs/cache")).expanduser()
TABLES = ("objects", "places", "place_edges", "mplaces", "mplace_edges", "rooms", "room_edges", "building", "traj", "mesh", "stats")
CAT = {"O": "objects", "p": "places", "P": "mplaces", "R": "rooms", "B": "building", "a": "agents"}
NO_LABEL = -1  # DSG stores unset labels as uint32 4294967295
MESH_NO_LABEL = 65535
ID_MASK = (1 << 56) - 1
JSON_NAME = "dsg_with_mesh.json"
STATS_LAYERS = (2, 3, 4)
STATS_COLUMNS = ("nodes_active", "nodes_removed", "nodes_merged", "nodes_w_parents", "nodes_w_children", "edges")
SIDE_FILES = ("trajectory.csv",) + tuple(f"layer_{k}_statistics.csv" for k in STATS_LAYERS)
# category pairs the tables represent; any other edge is only counted in meta["edges_unrepresented"]
REPRESENTED_EDGES = {("O", "p"), ("R", "p"), ("a", "p"), ("a", "a"), ("p", "p"), ("P", "P"), ("R", "R")}

ROW_FIELDS = {
    "objects": ("id", "pos", "label", "bbox_type", "bbox_center", "bbox_dim", "bbox_rot", "last_update_ns", "is_active", "n_mesh_conn",
                "n_parents", "parent_place", "room"),
    "places": ("id", "pos", "dist", "n_basis", "last_update_ns", "is_active", "room", "n_room_parents", "deg", "n_obj", "n_agent"),
    "mplaces": ("id", "pos", "label", "ellipse_centroid", "ellipse_expand", "is_active", "deg"),
    "rooms": ("id", "pos", "n_children", "ext_min", "ext_max", "has_parent", "deg"),
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
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with open(tmp, "wb") as f:
            (np.savez_compressed if compress else np.savez)(f, **self.flat())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path):
        npz = np.load(path, allow_pickle=False)
        meta = json.loads(str(npz["meta"]))
        return cls(meta, {tn: Table(tn, npz=npz) for tn in TABLES}, npz)

    @classmethod
    def build(cls, run_dir):
        """Snapshot straight from the json (needs spark_dsg), no cache involved."""
        return build_snapshot(run_dir)

    @classmethod
    def open(cls, run, cache_root=CACHE_ROOT, rebuild=False, cache=True):
        """Cache hit when schema and source stat (json and side files) match, else build from the json and write the cache."""
        run_dir = resolve_run(run)
        src = run_dir / "backend" / JSON_NAME
        if not src.is_file():
            raise SnapshotError(f"{src} missing (run incomplete?)")
        cp = cache_path(run_dir, cache_root)
        if cache and cp.exists() and not rebuild:
            try:
                snap = cls.load(cp)
                if snap.meta["schema"] == SCHEMA and source_fresh(snap.meta["src"], source_key(run_dir)):
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


# ------------------------------------------------------------------------------------------------ cache key
def _stat(path):
    try:
        st = pathlib.Path(path).stat()
    except OSError:
        return None
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def source_key(run_dir):
    """Size and mtime of the json and of every side file the snapshot reads."""
    b = pathlib.Path(run_dir) / "backend"
    top = _stat(b / JSON_NAME) or {"size": -1, "mtime_ns": -1}
    return {"path": str(b / JSON_NAME), **top, "side": {f: _stat(b / f) for f in SIDE_FILES}}


def _same_stat(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a["size"] == b["size"] and abs(a["mtime_ns"] - b["mtime_ns"]) < 1000  # 1 us: mtime over the Windows UNC path is rounded to 100 ns


def source_fresh(cached, now):
    """True when a cached `meta["src"]` still describes the files on disk."""
    return _same_stat(cached, now) and all(_same_stat(cached.get("side", {}).get(f), now["side"][f]) for f in SIDE_FILES)


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
    for k in STATS_LAYERS:
        a = arr("stats", f"layer_{k}_last")
        if a is not None and a.shape != (len(STATS_COLUMNS),):
            bad.append(f"stats.layer_{k}_last shape {a.shape}")
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


def _layer_stats(run_dir, warns):
    """Last row of backend/layer_{2,3,4}_statistics.csv per layer (cumulative churn counters); NaN vector when unusable."""
    out = {}
    for k in STATS_LAYERS:
        row = np.full(len(STATS_COLUMNS), np.nan)
        csv = run_dir / "backend" / f"layer_{k}_statistics.csv"
        if csv.exists():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    d = np.loadtxt(csv, delimiter=",", skiprows=1, ndmin=2)
                if d.ndim == 2 and d.shape[0] and d.shape[1] >= len(STATS_COLUMNS):
                    row = d[-1, :len(STATS_COLUMNS)].astype(np.float64)
                    with open(csv, encoding="utf-8") as f:
                        head = tuple(f.readline().strip().split(","))[:len(STATS_COLUMNS)]
                    if head != STATS_COLUMNS:
                        warns.append(f"{csv.name} columns {head} differ from {STATS_COLUMNS}")
            except (ValueError, OSError) as e:
                warns.append(f"{csv.name} unreadable: {e}")
        out[f"layer_{k}_last"] = row
    return out


def build_snapshot(run_dir):
    try:
        import spark_dsg as dsg  # lazy: cached snapshots load without it
    except ImportError as e:
        raise SnapshotError("spark_dsg not importable (build needs the WSL Hydra venv)") from e

    t0 = time.time()
    run_dir = pathlib.Path(run_dir)
    src = run_dir / "backend" / JSON_NAME
    if not src.is_file():
        raise SnapshotError(f"{src} missing")
    src_key = source_key(run_dir)  # taken before the load so a file changed meanwhile fails the next cache check
    try:
        g = dsg.DynamicSceneGraph.load(str(src))
    except Exception as e:  # spark_dsg reports bad json as assorted RuntimeError / ValueError types
        raise SnapshotError(f"cannot load {src}: {type(e).__name__}: {e}") from e
    t_load = time.time() - t0
    warns = []
    with np.errstate(all="ignore"):  # default-constructed attributes can hold huge doubles
        return _extract(g, dsg, run_dir, src_key, t0, t_load, warns)


def _extract(g, dsg, run_dir, src_key, t0, t_load, warns):
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
                  "ext_min": ext_min, "ext_max": ext_max, "has_parent": np.array([n.has_parent() for n in R], dtype=bool),
                  "deg": _degree(re_["ij"], len(R))}
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
                    "bbox_rot": np.array([b.world_R_center if v else np.full((3, 3), np.nan) for b, v in zip(bb, ok)], dtype=np.float32).reshape(len(O), 3, 3),
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
    T["stats"] = _layer_stats(run_dir, warns)
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
    meta = {"schema": SCHEMA, "run": run_dir.name, "src": src_key,
            "counts": {"objects": len(O), "places": len(P), "mplaces": len(M), "rooms": len(R), "agents": len(A), "mesh_vertices": nv,
                       "mesh_faces": len(T["mesh"]["faces"]), "place_edges": len(pe["w"]), "mplace_edges": len(me["w"]),
                       "room_edges": len(re_["w"])},
            "graph_counts": {"nodes": int(g.num_nodes()), "edges": int(g.num_edges())},
            "edge_counts": edge_counts, "edges_unrepresented": unrepresented, "mesh_flags": mesh_flags,
            "label_names": names, "bbox_type_names": bbox_names, "no_label": NO_LABEL, "mesh_no_label": MESH_NO_LABEL,
            "traj_source": traj_source, "traj_matches_agents": traj_match, "warnings": warns,
            "build_seconds": {"dsg_load": round(t_load, 2), "total": round(time.time() - t0, 2)}}
    return Snapshot(meta, {k: Table(k, v) for k, v in T.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("run")
    b.add_argument("--rebuild", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("runs", nargs="+")
    for sp in (b, c):
        sp.add_argument("--cache-dir", type=pathlib.Path, default=CACHE_ROOT)
    args = ap.parse_args(argv)
    if args.cmd == "build":
        t = time.time()
        with Snapshot.open(args.run, cache_root=args.cache_dir, rebuild=args.rebuild) as s:
            print(s.summary(), f"| {time.time() - t:.2f}s | cache {cache_path(resolve_run(args.run), args.cache_dir)}")
        return 0
    status = 0
    for run in args.runs:
        with Snapshot.open(run, cache_root=args.cache_dir) as s:
            bad = s.check()
            print(f"{s.meta['run']}: {'OK' if not bad else 'VIOLATIONS ' + '; '.join(bad)}")
            status |= bool(bad)
    return status


if __name__ == "__main__":
    sys.exit(main())
