"""Query-level failure of a scene graph, scored against a clean-input reference graph.

One query per reference object: "the object of class c in room R at the reference position x". Rooms of the reference
are mapped to rooms of a run by overlap of child-place footprints (many-to-one allowed), the answer is the nearest
same-class run node under a representing room, and the query succeeds when it lies within a position tolerance.
Pure numpy/scipy on a SceneView, so it runs without spark_dsg (builders: layer_attribution.query_views).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

SCHEMA = 1
ID_MASK = (1 << 56) - 1

OK = "ok"
ROOM_MISSING = "room_missing"
WRONG_ROOM = "wrong_room"
WRONG_LABEL = "wrong_label"
WRONG_POSITION = "wrong_position"
TARGET_MISSING = "target_missing"
ROOM_SPLIT = "room_split"
ROOM_MERGED = "room_merged"
DUPLICATE = "duplicate"
FAILURE_MODES = (ROOM_MISSING, WRONG_ROOM, WRONG_LABEL, WRONG_POSITION, TARGET_MISSING)
STRICT_ONLY_MODES = (ROOM_SPLIT, ROOM_MERGED, DUPLICATE)
ALL_MODES = FAILURE_MODES + STRICT_ONLY_MODES


def node_name(raw_id) -> str:
    """Raw 64-bit node id -> 'O(12)' (category char in the top byte, index below)."""
    raw = int(raw_id)
    return f"{chr(raw >> 56)}({raw & ID_MASK})"


# ------------------------------------------------------------------------------------------------ scene view
@dataclass
class SceneView:
    """Plain-numpy view of the layers a query touches. Rows are sorted by node id; relations are row indices."""

    name: str
    obj_id: np.ndarray  # (No,) int64
    obj_pos: np.ndarray  # (No, 3) float64
    obj_label: np.ndarray  # (No,) int32
    obj_par_indptr: np.ndarray  # (No+1,) int64, CSR of parent places
    obj_par_idx: np.ndarray  # (K,) int32, rows of the place arrays
    place_id: np.ndarray  # (Np,) int64
    place_pos: np.ndarray  # (Np, 3) float64
    place_radius: np.ndarray  # (Np,) float32, GVD clearance
    place_room: np.ndarray  # (Np,) int32, row of the room arrays or -1
    room_id: np.ndarray  # (Nr,) int64
    room_pos: np.ndarray  # (Nr, 3) float64
    label_names: dict = field(default_factory=dict)
    obj_bbox_dim: np.ndarray | None = None  # (No, 3) float32, nan where the box is invalid

    ARRAYS = ("obj_id", "obj_pos", "obj_label", "obj_par_indptr", "obj_par_idx", "place_id", "place_pos",
              "place_radius", "place_room", "room_id", "room_pos")

    @property
    def n_obj(self) -> int:
        return len(self.obj_id)

    @property
    def n_place(self) -> int:
        return len(self.place_id)

    @property
    def n_room(self) -> int:
        return len(self.room_id)

    def parents(self, i: int) -> np.ndarray:
        """Rows of the parent places of object row i (usually one, sometimes two, possibly none)."""
        return self.obj_par_idx[self.obj_par_indptr[i]:self.obj_par_indptr[i + 1]]

    def rooms_of_object(self, i: int) -> np.ndarray:
        """Sorted distinct rooms of the parent places of object i; places without a room are ignored."""
        r = self.place_room[self.parents(i)]
        return np.unique(r[r >= 0])

    def label_name(self, label: int) -> str:
        return self.label_names.get(int(label), str(int(label)))

    def save(self, path) -> None:
        d = {k: getattr(self, k) for k in self.ARRAYS}
        if self.obj_bbox_dim is not None:
            d["obj_bbox_dim"] = self.obj_bbox_dim
        d["meta"] = np.array(json.dumps({"schema": SCHEMA, "name": self.name,
                                         "label_names": {str(k): v for k, v in self.label_names.items()}}))
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path) -> "SceneView":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta["schema"] != SCHEMA:
                raise ValueError(f"unsupported SceneView schema {meta['schema']}")
            arrays = {k: z[k] for k in cls.ARRAYS}
            bbox = z["obj_bbox_dim"] if "obj_bbox_dim" in z.files else None
        return cls(name=meta["name"], label_names={int(k): v for k, v in meta["label_names"].items()},
                   obj_bbox_dim=bbox, **arrays)


# ------------------------------------------------------------------------------------------------ config
@dataclass(frozen=True)
class QueryConfig:
    tol_m: float = 1.0  # 3D distance within which a run node counts as the target
    room_method: str = "footprint"  # "footprint" (many-to-one overlap) or "centroid" (1-1 Hungarian, baseline)
    rep_overlap: float = 0.3  # a run room represents a reference room at this overlap coefficient
    min_ref_share: float = 0.05  # and only when the shared area is this share of the reference room (no slivers)
    cell_m: float = 0.5  # footprint raster cell
    gate_m: float | None = 1.0  # cell is covered when this close to a place; None = place clearance radius
    min_radius_m: float = 0.5  # floor for the clearance radius gate
    centroid_gate_m: float = 3.0
    exclusive: bool = True  # a run node cannot answer for a target when a same-class rival is closer to it


# ------------------------------------------------------------------------------------------------ room matching
@dataclass
class RoomMatch:
    method: str
    oc: np.ndarray  # (Nr_ref, Nr_run) overlap coefficient: shared cells / smaller footprint (centroid: 1 on matches)
    share: np.ndarray  # (Nr_ref, Nr_run) shared cells / reference footprint (centroid: 1 on matches)
    rep: np.ndarray  # (Nr_ref, Nr_run) bool, run room represents reference room
    ref_cells: np.ndarray  # (Nr_ref,) reference footprint in cells (0 for the centroid method)
    run_cells: np.ndarray  # (Nr_run,)
    coverage: np.ndarray  # (Nr_ref,) share of the reference footprint with a run place nearby (centroid: 1/0)

    def rep_rooms(self, i: int) -> np.ndarray:
        return np.flatnonzero(self.rep[i])

    def structure(self, i: int) -> str:
        """missing / one_to_one / merged / split / merged_split for reference room i."""
        s = self.rep_rooms(i)
        if len(s) == 0:
            return "missing"
        split = len(s) >= 2
        merged = bool((self.rep[:, s].sum(axis=0) >= 2).any())
        return "merged_split" if split and merged else "split" if split else "merged" if merged else "one_to_one"

    def structures(self) -> list:
        return [self.structure(i) for i in range(self.rep.shape[0])]


def _grid(a: SceneView, b: SceneView, cfg: QueryConfig) -> np.ndarray:
    xy = np.concatenate([v.place_pos[:, :2] for v in (a, b) if v.n_place] or [np.zeros((1, 2))])
    pad = (cfg.gate_m if cfg.gate_m is not None else 3.1) + cfg.cell_m
    lo, hi = xy.min(axis=0) - pad, xy.max(axis=0) + pad
    gx, gy = np.arange(lo[0], hi[0] + cfg.cell_m, cfg.cell_m), np.arange(lo[1], hi[1] + cfg.cell_m, cfg.cell_m)
    return np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1).reshape(-1, 2)


def room_label_map(view: SceneView, cells: np.ndarray, cfg: QueryConfig) -> np.ndarray:
    """Room row of the nearest place for every raster cell, -1 where no place is within the gate."""
    if view.n_place == 0:
        return np.full(len(cells), -1, dtype=np.int64)
    dist, idx = cKDTree(view.place_pos[:, :2]).query(cells)
    gate = (np.full(len(cells), cfg.gate_m) if cfg.gate_m is not None
            else np.maximum(view.place_radius[idx].astype(float), cfg.min_radius_m))
    return np.where(dist <= gate, view.place_room[idx], -1).astype(np.int64)


def match_rooms(ref: SceneView, run: SceneView, cfg: QueryConfig = QueryConfig()) -> RoomMatch:
    nr, nn = ref.n_room, run.n_room
    if cfg.room_method == "centroid":
        oc = np.zeros((nr, nn))
        if nr and nn:
            d = np.linalg.norm(ref.room_pos[:, None, :] - run.room_pos[None, :, :], axis=2)
            r, c = linear_sum_assignment(np.where(d <= cfg.centroid_gate_m, d, 1e6))
            for i, j in zip(r, c):
                oc[i, j] = float(d[i, j] <= cfg.centroid_gate_m)
        zeros = np.zeros(nr, dtype=np.int64)
        return RoomMatch("centroid", oc, oc.copy(), oc > 0, zeros, np.zeros(nn, dtype=np.int64), oc.sum(axis=1))
    if cfg.room_method != "footprint":
        raise ValueError(f"unknown room_method {cfg.room_method!r}")
    cells = _grid(ref, run, cfg)
    lref, lrun = room_label_map(ref, cells, cfg), room_label_map(run, cells, cfg)
    ref_cells = np.bincount(lref[lref >= 0], minlength=nr)[:nr].astype(np.int64)
    run_cells = np.bincount(lrun[lrun >= 0], minlength=nn)[:nn].astype(np.int64)
    both = (lref >= 0) & (lrun >= 0)
    shared = np.bincount(lref[both] * max(nn, 1) + lrun[both], minlength=nr * max(nn, 1))[:nr * nn].reshape(nr, nn)
    smaller = np.minimum(ref_cells[:, None], run_cells[None, :])
    oc = np.where(smaller > 0, shared / np.maximum(smaller, 1), 0.0)
    share = shared / np.maximum(ref_cells, 1)[:, None]
    coverage = np.bincount(lref[both], minlength=nr)[:nr] / np.maximum(ref_cells, 1)
    rep = (oc >= cfg.rep_overlap) & (share >= cfg.min_ref_share)
    return RoomMatch("footprint", oc, share, rep, ref_cells, run_cells, coverage)


# ------------------------------------------------------------------------------------------------ queries
@dataclass
class Query:
    """One reference object as a query, with the reference side of its support."""

    qid: int
    ref_row: int  # object row in the reference view
    ref_id: int  # raw node id of the reference object
    label: int
    label_name: str
    pos: np.ndarray  # (3,) target position
    ref_place_rows: tuple  # parent places of the target
    ref_place_ids: tuple
    ref_room_row: int  # room of the parent place nearest to the target
    ref_room_id: int
    ref_room_rows: tuple  # all distinct rooms of the parent places (more than one = ambiguous reference)
    n_same_class_in_room: int  # size of the reference answer set; 1 = the definite description is unique
    nbr_radius_m: float  # radius of the neighbour lists below
    nbr_row: np.ndarray  # (K,) other reference objects near the target, any class
    nbr_label: np.ndarray  # (K,)
    nbr_pos: np.ndarray  # (K, 3)


def objects_without_room(ref: SceneView) -> list:
    """Reference object rows that cannot be queried: no parent place, or parent places without a room."""
    return [i for i in range(ref.n_obj) if len(ref.rooms_of_object(i)) == 0]


def build_queries(ref: SceneView, cfg: QueryConfig = QueryConfig()) -> list:
    """One query per reference object that has an ancestor room, in node-id order."""
    rooms = [ref.rooms_of_object(i) for i in range(ref.n_obj)]
    count = Counter((int(ref.obj_label[i]), int(r)) for i in range(ref.n_obj) for r in rooms[i])
    kd = cKDTree(ref.obj_pos) if ref.n_obj else None
    radius = 2 * cfg.tol_m  # a node within tol of the target can only be closer to a rival within 2*tol of it
    out = []
    for i in range(ref.n_obj):
        if len(rooms[i]) == 0:
            continue
        par = ref.parents(i)
        par = par[ref.place_room[par] >= 0]
        d = np.linalg.norm(ref.place_pos[par] - ref.obj_pos[i], axis=1)
        primary = int(par[int(np.argmin(d))])  # nearest parent place wins, ties by row
        room = int(ref.place_room[primary])
        nb = np.sort(np.array([j for j in kd.query_ball_point(ref.obj_pos[i], radius) if j != i], dtype=np.int64))
        out.append(Query(
            qid=len(out), ref_row=i, ref_id=int(ref.obj_id[i]), label=int(ref.obj_label[i]),
            label_name=ref.label_name(ref.obj_label[i]), pos=ref.obj_pos[i].copy(),
            ref_place_rows=tuple(int(p) for p in ref.parents(i)),
            ref_place_ids=tuple(int(ref.place_id[p]) for p in ref.parents(i)),
            ref_room_row=room, ref_room_id=int(ref.room_id[room]), ref_room_rows=tuple(int(r) for r in rooms[i]),
            n_same_class_in_room=count[(int(ref.obj_label[i]), room)], nbr_radius_m=radius,
            nbr_row=nb, nbr_label=ref.obj_label[nb].astype(np.int64), nbr_pos=ref.obj_pos[nb].reshape(-1, 3)))
    return out


def pair_groups(queries) -> dict:
    """(class, reference room row) -> qids; more than one means 'the c in room R' is ambiguous in the reference."""
    g = defaultdict(list)
    for q in queries:
        g[(q.label, q.ref_room_row)].append(q.qid)
    return dict(g)


@dataclass
class QueryResult:
    """Outcome of one query on one run, with the run side of its support."""

    qid: int
    ok: bool
    mode: str  # ok or one of FAILURE_MODES
    mode_strict: str  # ok or one of ALL_MODES
    room_structure: str  # missing / one_to_one / merged / split / merged_split
    rep_room_rows: tuple  # run rooms representing the reference room
    rep_room_ids: tuple
    hit_row: int  # answering run object (nearest unclaimed same-class node under a representing room), -1 if none
    hit_id: int
    hit_err_m: float  # distance of the answer to the target (nan without answer)
    hit_place_ids: tuple  # parent places of the answer
    hit_room_ids: tuple  # rooms of the answer
    room_coverage: float  # share of the reference room's footprint with a run place nearby
    room_best_oc: float  # largest overlap coefficient of the reference room with any run room (0 if none)
    n_near: int  # unclaimed same-class nodes under representing rooms within tol
    n_same_class_in_rooms: int  # same-class nodes under representing rooms
    err_in_rooms_m: float  # nearest same-class node under representing rooms (inf if none)
    err_any_m: float  # nearest same-class node anywhere (inf if none)
    nearest_any_id: int  # that node, -1 if the run has no node of the class
    elsewhere_row: int  # same-class node within tol under a non-representing room (wrong_room evidence), -1 if none
    elsewhere_id: int
    elsewhere_room_ids: tuple
    other_class_row: int  # other-class node within tol that no reference object explains (wrong_label evidence)
    other_class_id: int
    other_class_label: int

    def row(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        for k in ("rep_room_rows", "rep_room_ids", "hit_place_ids", "hit_room_ids", "elsewhere_room_ids"):
            d[k] = ";".join(str(v) for v in d[k])
        return d


def _claimed(q: Query, pos: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """True where a same-class rival of the target is strictly closer to the node than the target is."""
    claimed = np.zeros(len(pos), dtype=bool)
    for r in q.nbr_pos[q.nbr_label == q.label]:
        claimed |= np.linalg.norm(pos - r, axis=1) < dist
    return claimed


def _room_ids(run: SceneView, i: int) -> tuple:
    return tuple(int(run.room_id[r]) for r in run.rooms_of_object(i))


def resolve(queries, run: SceneView, cfg: QueryConfig = QueryConfig(), match: RoomMatch | None = None, ref=None):
    """Resolve every query on the run; returns (results, RoomMatch). `ref` is needed unless `match` is given."""
    if match is None:
        match = match_rooms(ref, run, cfg)
    if any(2 * cfg.tol_m > q.nbr_radius_m + 1e-9 for q in queries):
        raise ValueError("queries were built with a smaller tolerance than the config")
    obj_rooms = [set(int(r) for r in run.rooms_of_object(i)) for i in range(run.n_obj)]
    structures = match.structures()
    results = []
    for q in queries:
        s = match.rep_rooms(q.ref_room_row)
        sset = set(int(x) for x in s)
        d = np.linalg.norm(run.obj_pos - q.pos, axis=1) if run.n_obj else np.zeros(0)
        same = run.obj_label == q.label
        in_room = np.array([bool(obj_rooms[n] & sset) for n in range(run.n_obj)], dtype=bool)
        claimed = _claimed(q, run.obj_pos, d) if cfg.exclusive and len(q.nbr_pos) else np.zeros(run.n_obj, dtype=bool)
        near = same & in_room & (d <= cfg.tol_m) & ~claimed
        cand = same & in_room
        elsewhere = same & ~in_room & (d <= cfg.tol_m) & ~claimed
        # an other-class node near the target is a label error unless a reference object of its own class is as close
        unexplained = [n for n in np.flatnonzero(~same & (d <= cfg.tol_m))
                       if not ((q.nbr_label == run.obj_label[n])
                               & (np.linalg.norm(q.nbr_pos - run.obj_pos[n], axis=1) <= d[n])).any()]
        other = int(min(unexplained, key=lambda n: (d[n], n))) if unexplained else -1
        hit = int(np.flatnonzero(near)[np.argmin(d[near])]) if near.any() else -1
        els = int(np.flatnonzero(elsewhere)[np.argmin(d[elsewhere])]) if elsewhere.any() else -1

        if len(s) == 0:
            mode = ROOM_MISSING
        elif hit >= 0:
            mode = OK
        elif els >= 0:
            mode = WRONG_ROOM
        elif other >= 0:
            mode = WRONG_LABEL
        elif cand.any():
            mode = WRONG_POSITION
        else:
            mode = TARGET_MISSING
        structure = structures[q.ref_room_row]
        mode_strict = mode
        if mode == OK:
            if structure in ("split", "merged_split"):
                mode_strict = ROOM_SPLIT
            elif structure == "merged":
                mode_strict = ROOM_MERGED
            elif int(near.sum()) >= 2:
                mode_strict = DUPLICATE
        results.append(QueryResult(
            qid=q.qid, ok=mode == OK, mode=mode, mode_strict=mode_strict, room_structure=structure,
            rep_room_rows=tuple(int(x) for x in s), rep_room_ids=tuple(int(run.room_id[x]) for x in s),
            hit_row=hit, hit_id=int(run.obj_id[hit]) if hit >= 0 else -1,
            hit_err_m=float(d[hit]) if hit >= 0 else float("nan"),
            hit_place_ids=tuple(int(run.place_id[p]) for p in run.parents(hit)) if hit >= 0 else (),
            hit_room_ids=_room_ids(run, hit) if hit >= 0 else (),
            room_coverage=float(match.coverage[q.ref_room_row]),
            room_best_oc=float(match.oc[q.ref_room_row].max()) if match.oc.shape[1] else 0.0,
            n_near=int(near.sum()), n_same_class_in_rooms=int(cand.sum()),
            err_in_rooms_m=float(d[cand].min()) if cand.any() else float("inf"),
            err_any_m=float(d[same].min()) if same.any() else float("inf"),
            nearest_any_id=int(run.obj_id[np.flatnonzero(same)[np.argmin(d[same])]]) if same.any() else -1,
            elsewhere_row=els, elsewhere_id=int(run.obj_id[els]) if els >= 0 else -1,
            elsewhere_room_ids=_room_ids(run, els) if els >= 0 else (),
            other_class_row=other, other_class_id=int(run.obj_id[other]) if other >= 0 else -1,
            other_class_label=int(run.obj_label[other]) if other >= 0 else -1))
    return results, match


def summarize(results, strict: bool = False) -> dict:
    """Counts and failure rate of one run; strict also counts room_merged / room_split / duplicate as failures."""
    modes = [r.mode_strict if strict else r.mode for r in results]
    c = Counter(modes)
    n = len(modes)
    n_fail = n - c.get(OK, 0)
    return {"n_queries": n, "n_ok": c.get(OK, 0), "n_fail": n_fail, "failure_rate": n_fail / n if n else float("nan"),
            "modes": {m: c.get(m, 0) for m in (ALL_MODES if strict else FAILURE_MODES)}}


@dataclass
class RunEvaluation:
    run: str
    queries: list
    results: list
    match: RoomMatch
    cfg: QueryConfig

    def summary(self, strict: bool = False) -> dict:
        return summarize(self.results, strict)

    def rows(self) -> list:
        """One flat dict per query: reference support next to the result."""
        out = []
        for q, r in zip(self.queries, self.results):
            out.append({"run": self.run, "qid": q.qid, "ref_node": node_name(q.ref_id), "label": q.label,
                        "label_name": q.label_name, "ref_room": node_name(q.ref_room_id),
                        "ref_places": ";".join(str(p) for p in q.ref_place_ids),
                        "n_same_class_in_room": q.n_same_class_in_room, **r.row()})
        return out


def evaluate(ref: SceneView, run: SceneView, cfg: QueryConfig = QueryConfig(), queries=None) -> RunEvaluation:
    queries = queries if queries is not None else build_queries(ref, cfg)
    results, match = resolve(queries, run, cfg, ref=ref)
    return RunEvaluation(run.name, queries, results, match, cfg)


# ------------------------------------------------------------------------------------------------ benign band
@dataclass(frozen=True)
class BenignBand:
    """Failure rates of runs with only a negligible fault. A fault run inside [0, hi_rate] is masked."""

    n_runs: int
    hi_rate: float  # largest benign failure rate; a fresh benign run exceeds it with probability <= 1/(n_runs+1)
    rates: tuple


def benign_band(rates) -> BenignBand:
    rates = tuple(sorted(float(r) for r in rates))
    if not rates:
        raise ValueError("no benign runs")
    return BenignBand(len(rates), rates[-1], rates)


def is_masked(rate: float, band: BenignBand) -> bool:
    """Fault present but the failure rate is indistinguishable from benign runs (within the benign maximum)."""
    return float(rate) <= band.hi_rate + 1e-12
