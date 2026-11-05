"""Per-layer feature battery for layer-wise failure attribution.

One feature = one function f(run, ref) -> float, the docstring is its definition. `run` and `ref` are FeatureView objects
(prototype-style arrays derived lazily from a canonical Snapshot, snapshot.py); the injection manifest is not reachable from them.
  kind "reference_based": compares `run` with the clean reference `ref`; ideal value = `ideal` (no discrepancy).
  kind "reference_free":  uses `run` only (ref is None); its clean value is whatever the scene gives.
layer tag: mesh_geom | mesh_sem | objects | places | rooms | edges | input_pose | process; the evidence layer of a tag is
dag.FEATURE_LAYER (edges, input_pose and process are not layer evidence).
  mesh_geom = vertex positions/faces, mesh_sem = per-vertex semantic labels (both are "mesh" in the DAG).
Registry entries carry tol_abs/tol_rel (smallest change worth calling a move) and count_based (feature mostly
tracks a node/vertex count, not geometry or meaning).

Conventions kept from the prototype (research/signal_probe/features_proto.py) so that its numbers reproduce:
  - a node with exactly one parent has it as `*_parent`; a node with two or more parents counts as unattached (-1)
  - mesh vertex subsampling, PCA sampling and purity sampling use rng seeds 0, 1, 2 over the unchanged vertex order
  - label histograms count labels 0..20 only (unset = -1 and out-of-range labels are not counted)
  - an INVALID object box has NaN centre and dimensions, which makes the box-based features NaN

  python -m layer_attribution.features_extract ...   runs the battery over run directories
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import math
import sys
import warnings
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Callable, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import cdist

from layer_attribution import dag
from layer_attribution.snapshot import MESH_NO_LABEL, STATS_COLUMNS, Snapshot

OBJECT_LABELS = (5, 6, 7, 10, 12, 13, 18)  # chair bench couch computer painting plant trashcan
N_LABELS = 21
LABEL_VOID, LABEL_FLOOR, LABEL_WALL = 0, 3, 19
MESH_SUB = 100_000  # query points for mesh distances
PCA_SAMPLES = 12_000  # vertices used for local-neighbourhood statistics
PCA_K = 24  # ~0.1-0.15 m neighbourhood at 0.1 m voxels
GATE_OBJ, GATE_PLACE, GATE_ROOM, GATE_QUERY = 1.0, 0.5, 3.0, 1.0  # same gates as analyze_runs.py


# --------------------------------------------------------------------------- snapshot access

class FeatureView:
    """Arrays of one run under the prototype's attribute names, derived lazily from a Snapshot.

    Keep the Snapshot open (or materialized) while features run. Nothing here reads the injection manifest.
    """

    def __init__(self, snapshot: Snapshot):
        self.snapshot = snapshot
        self.name = snapshot.meta.get("run", "")

    # mesh
    @cached_property
    def mesh_xyz(self):
        return np.asarray(self.snapshot.mesh.xyz, dtype=np.float32)

    @cached_property
    def mesh_label(self):
        lab = np.asarray(self.snapshot.mesh.label).astype(np.int32)
        lab[lab == MESH_NO_LABEL] = -1
        return lab

    @cached_property
    def mesh_faces(self):
        return np.asarray(self.snapshot.mesh.faces, dtype=np.int32)

    @cached_property
    def mesh_t_first(self):
        return np.asarray(self.snapshot.mesh.first_seen_ns, dtype=np.int64)

    @cached_property
    def mesh_t_last(self):
        return np.asarray(self.snapshot.mesh.stamp_ns, dtype=np.int64)

    # objects
    @cached_property
    def obj_pos(self):
        return np.asarray(self.snapshot.objects.pos, dtype=np.float64).reshape(-1, 3)

    @cached_property
    def obj_label(self):
        return np.asarray(self.snapshot.objects.label, dtype=np.int32)

    @cached_property
    def obj_bbox_center(self):
        return np.asarray(self.snapshot.objects.bbox_center, dtype=np.float64).reshape(-1, 3)

    @cached_property
    def obj_bbox_dim(self):
        return np.asarray(self.snapshot.objects.bbox_dim, dtype=np.float64).reshape(-1, 3)

    @cached_property
    def obj_bbox_R(self):
        return np.asarray(self.snapshot.objects.bbox_rot, dtype=np.float64).reshape(-1, 3, 3)

    @cached_property
    def obj_parent(self):
        """Place row of an object with exactly one parent when that parent is a place, else -1."""
        o = self.snapshot.objects
        return np.where((o.n_parents == 1) & (o.parent_place >= 0), o.parent_place, -1).astype(np.int32)

    @cached_property
    def obj_conn_offsets(self):
        return np.asarray(self.snapshot.objects.mesh_conn_indptr, dtype=np.int64)

    @cached_property
    def obj_conn_vertices(self):
        return np.asarray(self.snapshot.objects.mesh_conn_idx, dtype=np.int64)

    @cached_property
    def obj_last_update_ns(self):
        return np.asarray(self.snapshot.objects.last_update_ns, dtype=np.int64)

    @cached_property
    def obj_room(self):
        """Room index of each object via object -> place -> room, -1 if the chain breaks."""
        out = np.full(len(self.obj_pos), -1, dtype=int)
        ok = self.obj_parent >= 0
        out[ok] = self.place_parent[self.obj_parent[ok]]
        return out

    # places
    @cached_property
    def place_pos(self):
        return np.asarray(self.snapshot.places.pos, dtype=np.float64).reshape(-1, 3)

    @cached_property
    def place_radius(self):
        return np.asarray(self.snapshot.places.dist, dtype=np.float64)

    @cached_property
    def place_basis(self):
        return np.asarray(self.snapshot.places.n_basis, dtype=np.int32)

    @cached_property
    def place_parent(self):
        """Room row of a place with exactly one room parent, else -1."""
        p = self.snapshot.places
        return np.where(p.n_room_parents == 1, p.room, -1).astype(np.int32)

    @cached_property
    def place_edges(self):
        return np.asarray(self.snapshot.place_edges.ij, dtype=np.int32).reshape(-1, 2)

    @cached_property
    def place_last_update_ns(self):
        return np.asarray(self.snapshot.places.last_update_ns, dtype=np.int64)

    @cached_property
    def place_n_obj_children(self):
        return np.asarray(self.snapshot.places.n_obj, dtype=np.int32)

    @cached_property
    def meshplace_pos(self):
        return np.asarray(self.snapshot.mplaces.pos, dtype=np.float64).reshape(-1, 3)

    # rooms and building
    @cached_property
    def room_pos(self):
        return np.asarray(self.snapshot.rooms.pos, dtype=np.float64).reshape(-1, 3)

    @cached_property
    def room_has_parent(self):
        return np.asarray(self.snapshot.rooms.has_parent, dtype=bool)

    @cached_property
    def room_edges(self):
        return np.asarray(self.snapshot.room_edges.ij, dtype=np.int32).reshape(-1, 2)

    @cached_property
    def building_pos(self):
        return np.asarray(self.snapshot.building.pos, dtype=np.float64).reshape(-1, 3)

    # trajectory, graph totals, Hydra layer statistics
    @cached_property
    def traj(self):
        """(N, 8) rows t_ns, x, y, z, qw, qx, qy, qz."""
        t = self.snapshot.traj
        return np.column_stack([np.asarray(t.t_ns, dtype=np.float64), t.pos, t.quat_wxyz]).reshape(-1, 8)

    @cached_property
    def graph_counts(self):
        g = self.snapshot.meta.get("graph_counts", {})
        return np.array([g.get("nodes", -1), g.get("edges", -1)], dtype=np.int64)

    def stats_last(self, layer):
        """Last row of layer_<layer>_statistics.csv (nodes_active, nodes_removed, nodes_merged, ...); NaN vector when absent."""
        try:
            return np.asarray(getattr(self.snapshot.stats, f"layer_{layer}_last"), dtype=np.float64)
        except AttributeError:
            return np.full(len(STATS_COLUMNS), np.nan)

    # derived structures
    @cached_property
    def mesh_tree(self):
        return cKDTree(self.mesh_xyz.astype(np.float64))

    @cached_property
    def sub_idx(self):
        n = len(self.mesh_xyz)
        if n <= MESH_SUB:
            return np.arange(n)
        return np.sort(np.random.default_rng(0).choice(n, MESH_SUB, replace=False))

    @cached_property
    def sub_xyz(self):
        return self.mesh_xyz[self.sub_idx].astype(np.float64)

    @cached_property
    def t_first_s(self):
        return self.mesh_t_first / 1e9

    def pca_k(self, k):
        """Local PCA on a fixed vertex sample for k nearest neighbours: normals, eigenvalues (ascending), neighbour ids."""
        cache = self.__dict__.setdefault("_pca", {})
        if k not in cache:
            n = len(self.mesh_xyz)
            s = np.sort(np.random.default_rng(1).choice(n, min(PCA_SAMPLES, n), replace=False))
            xyz = self.mesh_xyz.astype(np.float64)
            _, nb = self.mesh_tree.query(xyz[s], k=k, workers=1)
            p = xyz[nb]  # (s, k, 3)
            c = p.mean(1, keepdims=True)
            q = p - c
            cov = np.einsum("ski,skj->sij", q, q) / k
            w, v = np.linalg.eigh(cov)
            cache[k] = dict(sample=s, nb=nb, center=c[:, 0], w=w, normal=v[:, :, 0], q=q)
        return cache[k]

    @property
    def pca(self):
        return self.pca_k(PCA_K)

    @cached_property
    def place_tree(self):
        return cKDTree(self.place_pos)


# --------------------------------------------------------------------------- registry

@dataclass
class Spec:
    name: str
    kind: str
    layer: str
    fn: Callable
    ideal: Optional[float]
    tol_abs: float
    tol_rel: float
    count_based: bool
    unit: str

    @property
    def definition(self):
        return " ".join((self.fn.__doc__ or "").split())


REGISTRY: dict[str, Spec] = {}


def feat(kind, layer, *, ideal=None, tol_abs=0.0, tol_rel=0.0, count_based=False, unit=""):
    assert kind in ("reference_based", "reference_free")

    def deco(fn):
        REGISTRY[fn.__name__] = Spec(fn.__name__, kind, layer, fn, ideal, tol_abs, tol_rel, count_based, unit)
        return fn

    return deco


# --------------------------------------------------------------------------- helpers

def match_points(a, b, gate):
    """Hungarian assignment of two point sets on Euclidean distance; only pairs within gate are kept."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((0, 2), dtype=int)
    d = cdist(a, b)
    r, c = linear_sum_assignment(np.where(d <= gate, d, 1e6))
    ok = d[r, c] <= gate
    return np.stack([r[ok], c[ok]], axis=1)


def js_divergence(p, q):
    """Jensen-Shannon divergence in bits between two non-negative vectors (normalised inside); range 0..1."""
    p = np.asarray(p, float) / max(np.sum(p), 1e-12)
    q = np.asarray(q, float) / max(np.sum(q), 1e-12)
    m = 0.5 * (p + q)

    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _label_hist(labels):
    """Counts per label 0..N_LABELS-1; unset (-1) and out-of-range labels are not counted."""
    lab = np.asarray(labels)
    return np.bincount(lab[(lab >= 0) & (lab < N_LABELS)], minlength=N_LABELS)


def aabb_iou(c1, d1, c2, d2, pad=0.05):
    """Axis-aligned 3D IoU of boxes (centre, dimensions); dimensions padded to >= pad so flat boxes keep a volume."""
    d1, d2 = np.maximum(d1, pad), np.maximum(d2, pad)
    lo = np.maximum(c1 - d1 / 2, c2 - d2 / 2)
    hi = np.minimum(c1 + d1 / 2, c2 + d2 / 2)
    inter = np.prod(np.clip(hi - lo, 0, None), axis=-1)
    return inter / np.maximum(np.prod(d1, -1) + np.prod(d2, -1) - inter, 1e-12)


def _unique_edges(run):
    """Undirected mesh edges as sorted vertex-pair keys with the number of faces using each."""
    f = run.mesh_faces.astype(np.int64)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e.sort(axis=1)
    n = len(run.mesh_xyz)
    key, cnt = np.unique(e[:, 0] * n + e[:, 1], return_counts=True)
    return key // n, key % n, cnt


def _ancestor_room_map(src_rooms, dst_rooms):
    """Hungarian map from src room index to dst room index on centroid distance (gate GATE_ROOM)."""
    pairs = match_points(src_rooms, dst_rooms, GATE_ROOM)
    return {int(i): int(j) for i, j in pairs}


def _ball_voxel_keys(pos, rad, vox=0.3):
    """Voxel keys (int64) covered by the union of balls (pos, rad); a free-space occupancy proxy."""
    keys = []
    base = np.floor(pos / vox).astype(np.int64)
    for r in np.unique(np.round(rad, 1)):
        m = np.round(rad, 1) == r
        k = int(math.ceil(r / vox))
        g = np.arange(-k, k + 1)
        off = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        off = off[np.linalg.norm(off * vox, axis=1) <= r]
        keys.append((base[m][:, None, :] + off[None]).reshape(-1, 3))
    allk = np.concatenate(keys) if keys else np.zeros((0, 3), dtype=np.int64)
    allk = allk + 4096
    return np.unique((allk[:, 0] << 42) | (allk[:, 1] << 21) | allk[:, 2])


def _floor_levels(run):
    """Floor-labelled vertices split into levels (floors): repeatedly take the 0.1 m histogram mode and the
    vertices within 0.6 m of it, until < 5 % of the floor vertices remain. Returns list of (z, t_first_s)."""
    m = run.mesh_label == LABEL_FLOOR
    z, t = run.mesh_xyz[m, 2].astype(float), run.t_first_s[m]
    left = np.ones(len(z), bool)
    out = []
    for _ in range(4):
        if left.sum() < 0.05 * len(z) or left.sum() < 500:
            break
        h, e = np.histogram(z[left], bins=np.arange(z[left].min(), z[left].max() + 0.2, 0.1))
        mode = e[np.argmax(h)] + 0.05
        sel = left & (np.abs(z - mode) < 0.6)
        out.append((z[sel], t[sel]))
        left &= ~sel
    return out


def _robust_sd(x):
    return 1.4826 * float(np.median(np.abs(x - np.median(x))))


# =========================================================================== MESH GEOMETRY, reference-based

@feat("reference_based", "mesh_geom", ideal=1.0, tol_abs=0.05, count_based=True)
def mesh_vertex_count_ratio(run, ref):
    """Number of mesh vertices of the run divided by the reference's."""
    return len(run.mesh_xyz) / len(ref.mesh_xyz)


@feat("reference_based", "mesh_geom", ideal=1.0, tol_abs=0.05, count_based=True)
def mesh_area_ratio(run, ref):
    """Total triangle area of the run mesh divided by the reference's."""

    def area(s):
        v = s.mesh_xyz.astype(np.float64)[s.mesh_faces]
        return float(np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1).sum() / 2)

    return area(run) / area(ref)


@feat("reference_based", "mesh_geom", ideal=0.0, tol_abs=0.02, unit="m")
def mesh_extra_mean(run, ref):
    """Mean over sampled run vertices of the distance to the nearest reference vertex (surface present only in the run)."""
    return float(ref.mesh_tree.query(run.sub_xyz, workers=1)[0].mean())


@feat("reference_based", "mesh_geom", ideal=0.0, tol_abs=0.02, unit="m")
def mesh_missing_mean(run, ref):
    """Mean over sampled reference vertices of the distance to the nearest run vertex (surface lost in the run)."""
    return float(run.mesh_tree.query(ref.sub_xyz, workers=1)[0].mean())


@feat("reference_based", "mesh_geom", ideal=0.0, tol_abs=0.02)
def mesh_extra_frac_gt10cm(run, ref):
    """Fraction of sampled run vertices farther than 0.10 m from every reference vertex (ghost surface)."""
    return float((ref.mesh_tree.query(run.sub_xyz, workers=1)[0] > 0.10).mean())


@feat("reference_based", "mesh_geom", ideal=0.0, tol_abs=0.02)
def mesh_missing_frac_gt10cm(run, ref):
    """Fraction of sampled reference vertices farther than 0.10 m from every run vertex (lost coverage)."""
    return float((run.mesh_tree.query(ref.sub_xyz, workers=1)[0] > 0.10).mean())


@feat("reference_based", "mesh_geom", ideal=0.0, tol_abs=0.05, unit="m")
def mesh_centroid_shift(run, ref):
    """Euclidean distance between the mean vertex position of the run and of the reference."""
    return float(np.linalg.norm(run.mesh_xyz.mean(0) - ref.mesh_xyz.mean(0)))


# =========================================================================== MESH SEMANTICS, reference-based

@feat("reference_based", "mesh_sem", ideal=0.0, tol_abs=0.003)
def mesh_label_hist_js(run, ref):
    """Jensen-Shannon divergence (bits) between the per-vertex label histograms (21 labels) of run and reference."""
    return js_divergence(_label_hist(run.mesh_label), _label_hist(ref.mesh_label))


@feat("reference_based", "mesh_sem", ideal=0.0, tol_abs=0.01)
def mesh_label_disagree_matched(run, ref):
    """Among sampled run vertices with a reference vertex within 0.05 m, fraction whose label differs from that vertex's label."""
    d, i = ref.mesh_tree.query(run.sub_xyz, workers=1)
    m = d < 0.05
    return float(np.mean(run.mesh_label[run.sub_idx][m] != ref.mesh_label[i[m]])) if m.any() else float("nan")


@feat("reference_based", "mesh_sem", ideal=1.0, tol_abs=0.05, count_based=True)
def mesh_objclass_vertex_ratio(run, ref):
    """Number of vertices carrying an object-class label (chair, bench, couch, computer, painting, plant, trashcan), run / reference."""
    f = lambda s: np.isin(s.mesh_label, OBJECT_LABELS).sum()
    return float(f(run) / max(f(ref), 1))


# =========================================================================== MESH, reference-free

@feat("reference_free", "mesh_geom", tol_rel=0.5)
def mesh_roughness_median(run, ref=None):
    """Median surface variation lambda_min / (lambda_0+lambda_1+lambda_2) of the 24-nearest-vertex covariance, over 12k sampled vertices."""
    w = run.pca["w"]
    return float(np.median(w[:, 0] / np.maximum(w.sum(1), 1e-18)))


@feat("reference_free", "mesh_geom", tol_rel=0.5)
def mesh_roughness_p90(run, ref=None):
    """90th percentile of the same surface variation (rough, edgy or fuzzy regions)."""
    w = run.pca["w"]
    return float(np.percentile(w[:, 0] / np.maximum(w.sum(1), 1e-18), 90))


@feat("reference_free", "mesh_geom", tol_rel=0.5, unit="m")
def wall_sheet_sd_median(run, ref=None):
    """Median sqrt(lambda_min) (thickness along the local normal, m) of the 24-vertex neighbourhoods centred on wall-labelled vertices."""
    p = run.pca
    m = run.mesh_label[p["sample"]] == LABEL_WALL
    return float(np.median(np.sqrt(np.maximum(p["w"][m, 0], 0)))) if m.sum() > 50 else float("nan")


@feat("reference_free", "mesh_geom", tol_abs=0.002, tol_rel=0.5, unit="m")
def floor_z_robust_sd(run, ref=None):
    """Floor-labelled vertices are split into floor levels; size-weighted mean over levels of 1.4826*MAD of z (m). A flat floor has ~1 mm."""
    lv = _floor_levels(run)
    if not lv:
        return float("nan")
    n = np.array([len(z) for z, _ in lv], float)
    return float(np.average([_robust_sd(z) for z, _ in lv], weights=n))


@feat("reference_free", "mesh_geom", tol_abs=0.01, unit="m")
def floor_z_time_range(run, ref=None):
    """Per floor level, vertices are cut into 8 equal-count bins by first-seen time (ties in vertex order); range (max-min) of the per-bin median z (m), size-weighted over levels. Odometry drift in z makes the same flat floor sit at different heights over time."""
    lv = _floor_levels(run)
    rng, wts = [], []
    for z, t in lv:
        o = np.argsort(t, kind="stable")  # ~180 vertices share one stamp; the default sort breaks ties by numpy version and CPU
        parts = [p for p in np.array_split(o, 8) if len(p) >= 400]
        if len(parts) >= 4:
            med = [np.median(z[p]) for p in parts]
            rng.append(max(med) - min(med))
            wts.append(len(z))
    return float(np.average(rng, weights=wts)) if rng else float("nan")


@feat("reference_free", "mesh_geom", tol_abs=0.5, tol_rel=0.0)
def floor_time_structure(run, ref=None):
    """floor_z_time_range / floor_z_robust_sd (dimensionless). About 0.55 when the floor scatter is noise-limited; above 1.5 when it is ordered in first-seen time (odometry drift in z). Fault signature, not a magnitude."""
    return floor_z_time_range(run) / floor_z_robust_sd(run)


def _revisit_offset(run, k, min_group):
    p, t = run.pca_k(k), run.t_first_s
    tt = t[p["nb"]]
    o = np.argsort(tt, axis=1)
    ts = np.take_along_axis(tt, o, 1)
    gap = np.diff(ts, axis=1)
    cut = gap.argmax(1) + 1
    ok = (gap.max(1) >= 30.0) & (cut >= min_group) & (k - cut >= min_group)
    if ok.sum() < 30:
        return float("nan")
    q = np.take_along_axis(p["q"], o[:, :, None], 1)
    proj = np.einsum("ski,si->sk", q, p["normal"])
    early = np.arange(k)[None] < cut[:, None]
    m1 = (proj * early).sum(1) / np.maximum(early.sum(1), 1)
    m2 = (proj * ~early).sum(1) / np.maximum((~early).sum(1), 1)
    return float(np.median(np.abs(m2 - m1)[ok]))


@feat("reference_free", "mesh_geom", tol_abs=0.01, unit="m")
def mesh_revisit_offset_median(run, ref=None):
    """For sampled vertices whose 24 nearest neighbours contain two groups first seen >= 30 s apart (>= 6 vertices each), distance along the local normal between the group means; median over such vertices (NaN if < 30 vertices qualify). A re-observed surface should merge (small); odometry drift leaves an offset copy. Sees offsets below the neighbourhood radius only and is biased low."""
    return _revisit_offset(run, PCA_K, 6)


@feat("reference_free", "mesh_geom", tol_abs=0.02, unit="m")
def mesh_revisit_offset_median_k96(run, ref=None):
    """Same as mesh_revisit_offset_median with 96 nearest neighbours and >= 12 vertices per time group (larger neighbourhood, sees larger offset copies)."""
    return _revisit_offset(run, 96, 12)


@feat("reference_free", "mesh_geom", tol_abs=0.002, tol_rel=0.5)
def mesh_open_edge_frac(run, ref=None):
    """Fraction of unique mesh edges used by exactly one triangle (boundary edges: holes, torn surfaces)."""
    _, _, cnt = _unique_edges(run)
    return float((cnt == 1).mean())


@feat("reference_free", "mesh_sem", tol_abs=0.005, tol_rel=0.25)
def mesh_label_edge_disagree_frac(run, ref=None):
    """Fraction of unique mesh edges whose two end vertices carry different labels (label boundary density; speckle raises it)."""
    a, b, _ = _unique_edges(run)
    return float((run.mesh_label[a] != run.mesh_label[b]).mean())


@feat("reference_free", "mesh_sem", tol_abs=0.02, unit="")
def mesh_objclass_knn_purity(run, ref=None):
    """Over sampled object-class vertices: mean fraction of their 12 nearest other vertices that have the same label. Local purity of the labelled mesh."""
    lab = run.mesh_label
    idx = np.flatnonzero(np.isin(lab, OBJECT_LABELS))
    if len(idx) < 200:
        return float("nan")
    idx = np.sort(np.random.default_rng(2).choice(idx, min(len(idx), 8000), replace=False))
    _, nb = run.mesh_tree.query(run.mesh_xyz[idx].astype(np.float64), k=13, workers=1)
    return float((lab[nb[:, 1:]] == lab[idx][:, None]).mean())


@feat("reference_free", "mesh_sem", tol_abs=0.005, tol_rel=0.25)
def mesh_void_frac(run, ref=None):
    """Fraction of mesh vertices with the void/unknown label (0)."""
    return float((run.mesh_label == LABEL_VOID).mean())


# =========================================================================== OBJECTS, reference-based

def _obj_pairs(run, ref):
    return match_points(ref.obj_pos, run.obj_pos, GATE_OBJ)  # (ref idx, run idx), position only


@feat("reference_based", "objects", ideal=1.0, tol_abs=0.1, count_based=True)
def obj_count_ratio(run, ref):
    """Number of object nodes, run / reference."""
    return len(run.obj_pos) / max(len(ref.obj_pos), 1)


@feat("reference_based", "objects", ideal=1.0, tol_abs=0.05)
def obj_recall_pos(run, ref):
    """Fraction of reference objects matched (Hungarian on centre distance, gate 1.0 m) to some run object, labels ignored."""
    return len(_obj_pairs(run, ref)) / max(len(ref.obj_pos), 1)


@feat("reference_based", "objects", ideal=1.0, tol_abs=0.05)
def obj_precision_pos(run, ref):
    """Fraction of run objects matched to a reference object (same matching), labels ignored."""
    return len(_obj_pairs(run, ref)) / max(len(run.obj_pos), 1)


@feat("reference_based", "objects", ideal=1.0, tol_abs=0.05)
def obj_label_agree_matched(run, ref):
    """Among position-matched object pairs, fraction with identical semantic label."""
    p = _obj_pairs(run, ref)
    return float(np.mean(ref.obj_label[p[:, 0]] == run.obj_label[p[:, 1]])) if len(p) else float("nan")


@feat("reference_based", "objects", ideal=1.0, tol_abs=0.05)
def obj_recall_same_label(run, ref):
    """Fraction of reference objects matched within 1.0 m to a run object of the same label (matching restricted to equal labels)."""
    n = 0
    for lab in np.unique(ref.obj_label):
        a, b = ref.obj_pos[ref.obj_label == lab], run.obj_pos[run.obj_label == lab]
        n += len(match_points(a, b, GATE_OBJ))
    return n / max(len(ref.obj_pos), 1)


@feat("reference_based", "objects", ideal=0.0, tol_abs=0.1, unit="m")
def obj_pos_err_median(run, ref):
    """Median centre distance over position-matched object pairs."""
    p = _obj_pairs(run, ref)
    return float(np.median(np.linalg.norm(ref.obj_pos[p[:, 0]] - run.obj_pos[p[:, 1]], axis=1))) if len(p) else float("nan")


@feat("reference_based", "objects", ideal=1.0, tol_abs=0.1)
def obj_bbox_iou_median(run, ref):
    """Median axis-aligned 3D box IoU (dimensions padded to 5 cm) over position-matched object pairs."""
    p = _obj_pairs(run, ref)
    if not len(p):
        return float("nan")
    return float(np.median(aabb_iou(ref.obj_bbox_center[p[:, 0]], ref.obj_bbox_dim[p[:, 0]],
                                    run.obj_bbox_center[p[:, 1]], run.obj_bbox_dim[p[:, 1]])))


@feat("reference_based", "objects", ideal=0.0, tol_abs=0.1)
def obj_bbox_size_err_median(run, ref):
    """Median over position-matched object pairs of the mean relative error of the three box dimensions, |d_run - d_ref| / max(d_ref, 0.1 m)."""
    p = _obj_pairs(run, ref)
    if not len(p):
        return float("nan")
    a, b = ref.obj_bbox_dim[p[:, 0]], run.obj_bbox_dim[p[:, 1]]
    return float(np.median((np.abs(b - a) / np.maximum(a, 0.1)).mean(1)))


@feat("reference_based", "objects", ideal=0.0, tol_abs=0.05)
def obj_extra_near_ref_frac(run, ref):
    """Fraction of run objects that are left over by the position matching (gate 1.0 m) yet lie within 1.5 m of a reference object: duplicates and fragments of existing objects rather than new ones."""
    if not len(run.obj_pos) or not len(ref.obj_pos):
        return float("nan")
    matched = np.zeros(len(run.obj_pos), bool)
    matched[_obj_pairs(run, ref)[:, 1]] = True
    d = cdist(run.obj_pos, ref.obj_pos).min(1)
    return float(((~matched) & (d <= 1.5)).sum() / len(run.obj_pos))


@feat("reference_based", "objects", ideal=0.0, tol_abs=0.02)
def obj_class_hist_js(run, ref):
    """Jensen-Shannon divergence (bits) between the object-count-per-label histograms of run and reference."""
    return js_divergence(_label_hist(run.obj_label), _label_hist(ref.obj_label))


# =========================================================================== OBJECTS, reference-free

def _same_class_nn(run):
    d = cdist(run.obj_pos, run.obj_pos)
    np.fill_diagonal(d, np.inf)
    same = run.obj_label[:, None] == run.obj_label[None, :]
    return np.where(same, d, np.inf)


@feat("reference_free", "objects", tol_abs=0.03)
def obj_dup_same_class_frac_050(run, ref=None):
    """Fraction of objects having another object of the same label with centre within 0.5 m (two nodes for one thing)."""
    return float((_same_class_nn(run).min(1) < 0.5).mean()) if len(run.obj_pos) > 1 else float("nan")


@feat("reference_free", "objects", tol_abs=0.03)
def obj_dup_same_class_frac_100(run, ref=None):
    """Fraction of objects having another object of the same label with centre within 1.0 m."""
    return float((_same_class_nn(run).min(1) < 1.0).mean()) if len(run.obj_pos) > 1 else float("nan")


@feat("reference_free", "objects", tol_abs=0.03)
def obj_overlap_diffclass_frac(run, ref=None):
    """Fraction of objects whose axis-aligned box has IoU > 0.1 with the box of an object of a different label (one thing split into differently labelled nodes)."""
    n = len(run.obj_pos)
    if n < 2:
        return float("nan")
    iou = aabb_iou(run.obj_bbox_center[:, None], run.obj_bbox_dim[:, None], run.obj_bbox_center[None], run.obj_bbox_dim[None])
    diff = run.obj_label[:, None] != run.obj_label[None, :]
    return float(((iou > 0.1) & diff).any(1).mean())


@feat("reference_free", "objects", tol_abs=0.03)
def obj_overlap_sameclass_frac(run, ref=None):
    """Fraction of objects whose axis-aligned box has IoU > 0.1 with the box of another object of the same label (duplicates or fragments)."""
    n = len(run.obj_pos)
    if n < 2:
        return float("nan")
    iou = aabb_iou(run.obj_bbox_center[:, None], run.obj_bbox_dim[:, None], run.obj_bbox_center[None], run.obj_bbox_dim[None])
    np.fill_diagonal(iou, 0)
    same = run.obj_label[:, None] == run.obj_label[None, :]
    return float(((iou > 0.1) & same).any(1).mean())


# generic plausible largest/smallest box dimension per class, metres; knowledge-based, not fitted to the scene
SIZE_PRIOR = {5: (0.3, 1.5), 6: (0.5, 3.0), 7: (0.6, 3.5), 10: (0.2, 2.0), 12: (0.3, 3.0), 13: (0.2, 2.2), 18: (0.2, 1.5)}


@feat("reference_free", "objects", tol_abs=0.05)
def obj_size_prior_violation_frac(run, ref=None):
    """Fraction of objects whose largest box dimension lies outside a generic per-class range (chair 0.3-1.5 m, bench 0.5-3, couch 0.6-3.5, computer 0.2-2, painting 0.3-3, plant 0.2-2.2, trashcan 0.2-1.5)."""
    if not len(run.obj_pos):
        return float("nan")
    big = run.obj_bbox_dim.max(1)
    bad = [not (SIZE_PRIOR[int(l)][0] <= b <= SIZE_PRIOR[int(l)][1]) if int(l) in SIZE_PRIOR else True
           for l, b in zip(run.obj_label, big)]
    return float(np.mean(bad))


@feat("reference_free", "objects", tol_abs=0.05)
def obj_tiny_frac(run, ref=None):
    """Fraction of objects whose largest box dimension is < 0.4 m (fragments)."""
    return float((run.obj_bbox_dim.max(1) < 0.4).mean()) if len(run.obj_pos) else float("nan")


@feat("reference_free", "objects", tol_abs=0.05, unit="")
def obj_logvol_mad_within_class(run, ref=None):
    """Median over labels with >= 4 objects of the median absolute deviation of log box volume (dimensions padded to >= 5 cm) inside the label: size consistency within one class."""
    v = np.log(np.prod(np.maximum(run.obj_bbox_dim, 0.05), axis=1))
    vals = []
    for lab in np.unique(run.obj_label):
        x = v[run.obj_label == lab]
        if len(x) >= 4:
            vals.append(np.median(np.abs(x - np.median(x))))
    return float(np.median(vals)) if vals else float("nan")


@feat("reference_free", "objects", tol_abs=0.05)
def obj_mesh_label_purity_median(run, ref=None):
    """Median over objects of the fraction of its connected mesh vertices (mesh_connections) carrying the object's own label."""
    off, ver = run.obj_conn_offsets, run.obj_conn_vertices
    if not len(run.obj_pos):
        return float("nan")
    pur = [np.mean(run.mesh_label[ver[off[i]:off[i + 1]]] == run.obj_label[i]) for i in range(len(run.obj_pos)) if off[i + 1] > off[i]]
    return float(np.median(pur)) if pur else float("nan")


@feat("reference_free", "objects", tol_abs=0.05)
def obj_mesh_label_purity_low_frac(run, ref=None):
    """Fraction of objects for which < 50 % of the connected mesh vertices carry the object's own label."""
    off, ver = run.obj_conn_offsets, run.obj_conn_vertices
    pur = [np.mean(run.mesh_label[ver[off[i]:off[i + 1]]] == run.obj_label[i]) for i in range(len(run.obj_pos)) if off[i + 1] > off[i]]
    return float(np.mean(np.asarray(pur) < 0.5)) if pur else float("nan")


@feat("reference_free", "objects", tol_abs=0.05, unit="vertices")
def obj_support_median(run, ref=None):
    """Median number of mesh vertices connected to an object."""
    return float(np.median(np.diff(run.obj_conn_offsets))) if len(run.obj_pos) else float("nan")


@feat("reference_free", "objects", tol_abs=0.2, unit="m")
def obj_nearest_place_dist_p90(run, ref=None):
    """90th percentile over objects of the distance to the nearest place node (objects outside the explored free space)."""
    if not len(run.obj_pos):
        return float("nan")
    return float(np.percentile(run.place_tree.query(run.obj_pos, workers=1)[0], 90))


@feat("reference_free", "objects", tol_abs=0.03)
def obj_orphan_frac(run, ref=None):
    """Fraction of objects without a parent place."""
    return float((run.obj_parent < 0).mean()) if len(run.obj_pos) else float("nan")


@feat("reference_free", "objects", tol_abs=0.03)
def obj_place_intrusion_frac(run, ref=None):
    """Fraction of objects whose centre lies inside the free-space ball (radius - 0.05 m) of its parent place: occupied and free at once."""
    ok = run.obj_parent >= 0
    if not ok.any():
        return float("nan")
    d = np.linalg.norm(run.obj_pos[ok] - run.place_pos[run.obj_parent[ok]], axis=1)
    return float((d < run.place_radius[run.obj_parent[ok]] - 0.05).mean())


FLOOR_STANDING = (5, 6, 7, 13, 18)  # chair bench couch plant trashcan rest on the floor


@feat("reference_free", "objects", tol_abs=0.1, unit="m")
def obj_floor_gap_p90(run, ref=None):
    """For floor-standing labels (chair, bench, couch, plant, trashcan): 90th percentile of the distance from the box bottom-centre to the nearest floor-labelled vertex (m)."""
    m = np.isin(run.obj_label, FLOOR_STANDING)
    fl = run.mesh_xyz[run.mesh_label == LABEL_FLOOR]
    if m.sum() < 5 or len(fl) < 100:
        return float("nan")
    bottom = run.obj_bbox_center[m] - np.array([0, 0, 0.5]) * run.obj_bbox_dim[m] * np.array([0, 0, 1])
    d = cKDTree(fl.astype(np.float64)).query(bottom, workers=1)[0]
    return float(np.percentile(d, 90))


# =========================================================================== PLACES, reference-based

@feat("reference_based", "places", ideal=1.0, tol_abs=0.05, count_based=True)
def place_count_ratio(run, ref):
    """Number of GVD place nodes, run / reference."""
    return len(run.place_pos) / max(len(ref.place_pos), 1)


@feat("reference_based", "places", ideal=1.0, tol_abs=0.05)
def place_coverage_050(run, ref):
    """Fraction of reference places with a run place within 0.5 m."""
    return float((run.place_tree.query(ref.place_pos, workers=1)[0] <= GATE_PLACE).mean())


@feat("reference_based", "places", ideal=0.0, tol_abs=0.05)
def place_spurious_050(run, ref):
    """Fraction of run places with no reference place within 0.5 m."""
    return float((ref.place_tree.query(run.place_pos, workers=1)[0] > GATE_PLACE).mean())


@feat("reference_based", "places", ideal=0.0, tol_abs=0.1, unit="m")
def place_chamfer_mean(run, ref):
    """Mean of the two mean nearest-neighbour distances between the place position sets (run->ref and ref->run)."""
    a = ref.place_tree.query(run.place_pos, workers=1)[0].mean()
    b = run.place_tree.query(ref.place_pos, workers=1)[0].mean()
    return float((a + b) / 2)


@feat("reference_based", "places", ideal=1.0, tol_abs=0.05)
def place_freespace_iou(run, ref):
    """IoU of the free-space occupancy implied by the places: union of balls (position, radius) rasterised at 0.3 m, run vs reference."""
    a, b = _ball_voxel_keys(ref.place_pos, ref.place_radius), _ball_voxel_keys(run.place_pos, run.place_radius)
    inter = len(np.intersect1d(a, b, assume_unique=True))
    return inter / max(len(a) + len(b) - inter, 1)


@feat("reference_based", "places", ideal=1.0, tol_abs=0.05)
def place_freevol_ratio(run, ref):
    """Free-space volume implied by the places (union of balls (position, radius) rasterised at 0.3 m), run / reference; an aggregate that does not depend on where individual place nodes sit."""
    return len(_ball_voxel_keys(run.place_pos, run.place_radius)) / max(len(_ball_voxel_keys(ref.place_pos, ref.place_radius)), 1)


@feat("reference_free", "places", tol_abs=20.0, unit="m3")
def place_freevol_m3(run, ref=None):
    """Free-space volume implied by the places (union of balls rasterised at 0.3 m voxels), in m^3."""
    return len(_ball_voxel_keys(run.place_pos, run.place_radius)) * 0.027


@feat("reference_based", "places", ideal=0.0, tol_abs=0.05, unit="m")
def place_radius_err_median(run, ref):
    """Median |radius_run - radius_ref| over reference places that have a run place within 0.5 m (nearest run place)."""
    d, j = run.place_tree.query(ref.place_pos, workers=1)
    ok = d <= GATE_PLACE
    return float(np.median(np.abs(run.place_radius[j[ok]] - ref.place_radius[ok]))) if ok.any() else float("nan")


@feat("reference_based", "places", ideal=1.0, tol_abs=0.05)
def place_radius_mean_ratio(run, ref):
    """Mean place radius (clearance), run / reference."""
    return float(run.place_radius.mean() / ref.place_radius.mean())


@feat("reference_based", "places", ideal=1.0, tol_abs=0.1, count_based=True)
def place_edge_count_ratio(run, ref):
    """Number of place-place edges, run / reference."""
    return len(run.place_edges) / max(len(ref.place_edges), 1)


@feat("reference_based", "places", ideal=1.0, tol_abs=0.1, count_based=True)
def meshplace_count_ratio(run, ref):
    """Number of surface (2D mesh-place) nodes, run / reference."""
    return len(run.meshplace_pos) / max(len(ref.meshplace_pos), 1)


# =========================================================================== PLACES, reference-free

@feat("reference_free", "places", tol_abs=0.03)
def place_mesh_intrusion_frac(run, ref=None):
    """Fraction of places for which the nearest mesh vertex is closer than 0.7 x the place radius (the free-space ball contains surface)."""
    d = run.mesh_tree.query(run.place_pos, workers=1)[0]
    return float((d < 0.7 * run.place_radius).mean())


@feat("reference_free", "places", tol_abs=0.05, unit="")
def place_radius_mesh_ratio_median(run, ref=None):
    """Median over places of place radius / distance to the nearest mesh vertex (1 when the clearance matches the mesh)."""
    d = run.mesh_tree.query(run.place_pos, workers=1)[0]
    return float(np.median(run.place_radius / np.maximum(d, 1e-3)))


@feat("reference_free", "places", tol_abs=0.02)
def place_edge_gap_frac(run, ref=None):
    """Fraction of place-place edges whose end balls do not touch: length > r_u + r_v + 0.1 m."""
    e = run.place_edges
    if not len(e):
        return float("nan")
    L = np.linalg.norm(run.place_pos[e[:, 0]] - run.place_pos[e[:, 1]], axis=1)
    return float((L > run.place_radius[e[:, 0]] + run.place_radius[e[:, 1]] + 0.1).mean())


@feat("reference_free", "places", tol_abs=0.03)
def place_edge_wall_cross_frac(run, ref=None):
    """Fraction of place-place edges for which one of 5 equally spaced points on the segment is closer than 0.15 m to a mesh vertex (traversable edge passes through surface)."""
    e = run.place_edges
    if not len(e):
        return float("nan")
    a, b = run.place_pos[e[:, 0]], run.place_pos[e[:, 1]]
    t = np.linspace(0.1, 0.9, 5)[None, :, None]
    pts = (a[:, None] * (1 - t) + b[:, None] * t).reshape(-1, 3)
    d = run.mesh_tree.query(pts, workers=1)[0].reshape(len(e), 5)
    return float((d.min(1) < 0.15).mean())


def _place_components(run):
    n, e = len(run.place_pos), run.place_edges
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    return connected_components(g, directed=False)[1]


@feat("reference_free", "places", tol_abs=0.5, unit="components")
def place_components(run, ref=None):
    """Number of connected components of the place graph."""
    return float(len(np.unique(_place_components(run))))


@feat("reference_free", "places", tol_abs=0.03)
def place_isolated_frac(run, ref=None):
    """Fraction of places with no place-place edge."""
    deg = np.bincount(run.place_edges.ravel(), minlength=len(run.place_pos))
    return float((deg == 0).mean())


@feat("reference_free", "places", tol_abs=0.2)
def place_degree_mean(run, ref=None):
    """Mean number of place-place edges per place."""
    return float(2 * len(run.place_edges) / max(len(run.place_pos), 1))


@feat("reference_free", "places", tol_abs=0.03)
def place_orphan_frac(run, ref=None):
    """Fraction of places without a parent room."""
    return float((run.place_parent < 0).mean())


@feat("reference_free", "places", tol_abs=0.05, unit="m")
def place_radius_mean(run, ref=None):
    """Mean place radius (distance to the nearest obstacle) in metres."""
    return float(run.place_radius.mean())


@feat("reference_free", "places", tol_abs=0.03)
def place_small_radius_frac(run, ref=None):
    """Fraction of places with radius < 0.5 m (narrow clearance)."""
    return float((run.place_radius < 0.5).mean())


# =========================================================================== ROOMS, reference-based

@feat("reference_based", "rooms", ideal=1.0, tol_abs=0.1, count_based=True)
def room_count_ratio(run, ref):
    """Number of room nodes, run / reference."""
    return len(run.room_pos) / max(len(ref.room_pos), 1)


@feat("reference_based", "rooms", ideal=1.0, tol_abs=0.1)
def room_match_frac(run, ref):
    """Fraction of reference rooms matched (Hungarian on centroid, gate 3 m) to a run room."""
    return len(match_points(ref.room_pos, run.room_pos, GATE_ROOM)) / max(len(ref.room_pos), 1)


@feat("reference_based", "rooms", ideal=0.0, tol_abs=1.0, unit="m")
def room_centroid_err_median(run, ref):
    """Median centroid distance over matched rooms."""
    p = match_points(ref.room_pos, run.room_pos, GATE_ROOM)
    return float(np.median(np.linalg.norm(ref.room_pos[p[:, 0]] - run.room_pos[p[:, 1]], axis=1))) if len(p) else float("nan")


def _place_correspondence(run, ref):
    d, j = run.place_tree.query(ref.place_pos, workers=1)
    ok = (d <= GATE_PLACE) & (ref.place_parent >= 0) & (run.place_parent[j] >= 0)
    return np.flatnonzero(ok), j[ok]


@feat("reference_based", "rooms", ideal=1.0, tol_abs=0.1)
def room_assignment_agreement(run, ref):
    """For reference places with a run place within 0.5 m, fraction whose run room equals the Hungarian-mapped reference room."""
    rm = _ancestor_room_map(ref.room_pos, run.room_pos)
    i, j = _place_correspondence(run, ref)
    if not len(i):
        return float("nan")
    return float(np.mean([rm.get(int(ref.place_parent[a]), -2) == int(run.place_parent[b]) for a, b in zip(i, j)]))


@feat("reference_based", "rooms", ideal=1.0, tol_abs=0.1)
def room_partition_ari(run, ref):
    """Adjusted Rand index between the room partitions of corresponding places (reference place -> nearest run place within 0.5 m); needs no room matching."""
    i, j = _place_correspondence(run, ref)
    if len(i) < 3:
        return float("nan")
    a, b = ref.place_parent[i], run.place_parent[j]
    _, a = np.unique(a, return_inverse=True)
    _, b = np.unique(b, return_inverse=True)
    ct = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(ct, (a, b), 1)
    c2 = lambda x: x * (x - 1) / 2
    sum_ij, sum_a, sum_b, tot = c2(ct).sum(), c2(ct.sum(1)).sum(), c2(ct.sum(0)).sum(), c2(len(a))
    exp = sum_a * sum_b / tot
    den = 0.5 * (sum_a + sum_b) - exp
    return float((sum_ij - exp) / den) if den > 0 else 1.0


# =========================================================================== ROOMS, reference-free

@feat("reference_free", "rooms", tol_abs=0.5, unit="rooms")
def room_empty_count(run, ref=None):
    """Number of rooms with no child place."""
    return float(len(run.room_pos) - len(np.unique(run.place_parent[run.place_parent >= 0])))


@feat("reference_free", "rooms", tol_abs=0.03)
def room_min_share(run, ref=None):
    """Smallest room's share of all room-assigned places (tiny fragments pull it down)."""
    c = np.bincount(run.place_parent[run.place_parent >= 0], minlength=len(run.room_pos))
    return float(c.min() / max(c.sum(), 1))


def _room_hull_areas(run):
    out = []
    for r in range(len(run.room_pos)):
        pts = run.place_pos[run.place_parent == r][:, :2]
        try:
            out.append(ConvexHull(pts).volume) if len(pts) >= 3 else out.append(0.0)
        except Exception:
            out.append(0.0)
    return np.asarray(out)


@feat("reference_free", "rooms", tol_abs=2.0, unit="m2")
def room_footprint_min(run, ref=None):
    """Smallest convex-hull area (xy, m^2) of the child places of a room."""
    a = _room_hull_areas(run)
    return float(a.min()) if len(a) else float("nan")


@feat("reference_free", "rooms", tol_abs=20.0, unit="m2")
def room_footprint_max(run, ref=None):
    """Largest convex-hull area (xy, m^2) of the child places of a room."""
    a = _room_hull_areas(run)
    return float(a.max()) if len(a) else float("nan")


@feat("reference_free", "rooms", tol_abs=0.2)
def room_components_mean(run, ref=None):
    """Mean over rooms of the number of connected pieces of the room's places in the place graph (edges inside the room only); 1 = contiguous."""
    e = run.place_edges
    same = run.place_parent[e[:, 0]] == run.place_parent[e[:, 1]]
    n = len(run.place_pos)
    g = coo_matrix((np.ones(same.sum()), (e[same, 0], e[same, 1])), shape=(n, n))
    comp = connected_components(g, directed=False)[1]
    k = [len(np.unique(comp[run.place_parent == r])) for r in range(len(run.room_pos)) if (run.place_parent == r).any()]
    return float(np.mean(k)) if k else float("nan")


@feat("reference_free", "rooms", tol_abs=0.03)
def room_boundary_mixing(run, ref=None):
    """Mean over places of the fraction of their 6 nearest places (xyz) that belong to a different room: interleaving of the partition."""
    ok = run.place_parent >= 0
    pos, par = run.place_pos[ok], run.place_parent[ok]
    if len(pos) < 8:
        return float("nan")
    _, nb = cKDTree(pos).query(pos, k=7, workers=1)
    return float((par[nb[:, 1:]] != par[:, None]).mean())


@feat("reference_free", "rooms", tol_abs=0.3)
def room_degree_mean(run, ref=None):
    """Mean number of room-room edges per room."""
    return float(2 * len(run.room_edges) / max(len(run.room_pos), 1))


@feat("reference_free", "rooms", tol_abs=0.5)
def room_elongation_max(run, ref=None):
    """Largest sqrt(lambda_1 / lambda_2) of the xy covariance of a room's child places (rooms with >= 5 places): corridor-like or merged rooms score high."""
    out = []
    for r in range(len(run.room_pos)):
        pts = run.place_pos[run.place_parent == r][:, :2]
        if len(pts) >= 5:
            w = np.linalg.eigvalsh(np.cov(pts.T))
            out.append(math.sqrt(max(w[1], 1e-9) / max(w[0], 1e-9)))
    return float(max(out)) if out else float("nan")


@feat("reference_free", "rooms", tol_abs=0.03)
def room_without_objects_frac(run, ref=None):
    """Fraction of rooms with no object below them (object -> place -> room)."""
    if not len(run.room_pos):
        return float("nan")
    return float(1 - len(np.unique(run.obj_room[run.obj_room >= 0])) / len(run.room_pos))


# =========================================================================== CROSS-LAYER / QUERY

def _query_status(run, ref):
    """Test query 'object of label c in room R' per reference object: ok | wrong_object | target_missing | room_missing."""
    rm = _ancestor_room_map(ref.room_pos, run.room_pos)
    cnt = {"ok": 0, "wrong_object": 0, "target_missing": 0, "room_missing": 0}
    for i in range(len(ref.obj_pos)):
        r = ref.obj_room[i]
        if r < 0:
            continue
        rr = rm.get(int(r))
        if rr is None:
            cnt["room_missing"] += 1
            continue
        cand = np.flatnonzero((run.obj_label == ref.obj_label[i]) & (run.obj_room == rr))
        if not len(cand):
            cnt["target_missing"] += 1
        elif np.linalg.norm(run.obj_pos[cand] - ref.obj_pos[i], axis=1).min() <= GATE_QUERY:
            cnt["ok"] += 1
        else:
            cnt["wrong_object"] += 1
    return cnt


@feat("reference_based", "edges", ideal=1.0, tol_abs=0.05)
def query_success_rate(run, ref):
    """Fraction of reference objects (class c, room R) whose query 'class c in room R' resolves in the run to an object of the same label within 1 m of the reference object, after mapping R to the run's room."""
    c = _query_status(run, ref)
    return c["ok"] / max(sum(c.values()), 1)


@feat("reference_based", "edges", ideal=0.0, tol_abs=0.03)
def query_fail_room_missing(run, ref):
    """Fraction of test queries that fail because the target's room has no run room within 3 m."""
    c = _query_status(run, ref)
    return c["room_missing"] / max(sum(c.values()), 1)


@feat("reference_based", "edges", ideal=0.0, tol_abs=0.03)
def query_fail_target_missing(run, ref):
    """Fraction of test queries that fail because the room exists but holds no object of that label."""
    c = _query_status(run, ref)
    return c["target_missing"] / max(sum(c.values()), 1)


@feat("reference_based", "edges", ideal=0.0, tol_abs=0.03)
def query_fail_wrong_object(run, ref):
    """Fraction of test queries that fail because every same-label object in the room is > 1 m from the reference object."""
    c = _query_status(run, ref)
    return c["wrong_object"] / max(sum(c.values()), 1)


@feat("reference_based", "edges", ideal=1.0, tol_abs=0.1)
def edge_obj_room_agreement(run, ref):
    """Parent-chain consistency: among reference objects matched (Hungarian, gate 1 m, equal labels) to a run object, fraction whose run room (object -> place -> room) is the Hungarian-mapped reference room; objects with a broken chain count as disagreeing."""
    rm = _ancestor_room_map(ref.room_pos, run.room_pos)
    ok = tot = 0
    for lab in np.unique(ref.obj_label):
        ia, ib = np.flatnonzero(ref.obj_label == lab), np.flatnonzero(run.obj_label == lab)
        for a, b in match_points(ref.obj_pos[ia], run.obj_pos[ib], GATE_OBJ):
            ra = ref.obj_room[ia[a]]
            if ra < 0:
                continue
            tot += 1
            ok += rm.get(int(ra), -2) == run.obj_room[ib[b]]
    return ok / tot if tot else float("nan")


@feat("reference_free", "edges", tol_abs=0.03)
def query_ambiguity_frac(run, ref=None):
    """Among (label, room) pairs that contain at least one object, fraction that contain two or more (query 'label in room' is ambiguous)."""
    ok = run.obj_room >= 0
    keys = run.obj_label[ok] * 1000 + run.obj_room[ok]
    if not len(keys):
        return float("nan")
    _, c = np.unique(keys, return_counts=True)
    return float((c >= 2).mean())


@feat("reference_free", "edges", tol_abs=0.5, unit="m")
def obj_parent_dist_median(run, ref=None):
    """Median distance between an object and the position of its parent place."""
    ok = run.obj_parent >= 0
    return float(np.median(np.linalg.norm(run.obj_pos[ok] - run.place_pos[run.obj_parent[ok]], axis=1))) if ok.any() else float("nan")


# =========================================================================== INPUT POSE (trajectory), both kinds

def _traj_xyz_t(s):
    return s.traj[:, 1:4], s.traj[:, 0] / 1e9


@feat("reference_based", "input_pose", ideal=0.0, tol_abs=0.1, unit="m")
def traj_ate_rmse(run, ref):
    """RMS translation error of the run trajectory against the reference trajectory linearly interpolated at the run's keyframe times (no alignment)."""
    p, t = _traj_xyz_t(run)
    q, u = _traj_xyz_t(ref)
    if len(p) < 2 or len(q) < 2:
        return float("nan")
    qi = np.stack([np.interp(t, u, q[:, k]) for k in range(3)], axis=1)
    return float(np.sqrt(np.mean(np.sum((p - qi) ** 2, axis=1))))


@feat("reference_based", "input_pose", ideal=1.0, tol_abs=0.1, count_based=True)
def traj_keyframe_ratio(run, ref):
    """Number of trajectory keyframes, run / reference."""
    return len(run.traj) / max(len(ref.traj), 1)


@feat("reference_free", "input_pose", tol_abs=0.5)
def traj_dt_p95_over_median(run, ref=None):
    """95th percentile of keyframe time gaps divided by their median (dropped frames stretch gaps)."""
    _, t = _traj_xyz_t(run)
    dt = np.diff(t)
    return float(np.percentile(dt, 95) / np.median(dt)) if len(dt) > 5 else float("nan")


@feat("reference_free", "input_pose", tol_abs=0.5)
def traj_step_max_over_median(run, ref=None):
    """Largest distance between consecutive keyframes divided by the median distance (jumps)."""
    p, _ = _traj_xyz_t(run)
    s = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return float(s.max() / np.median(s)) if len(s) > 5 and np.median(s) > 0 else float("nan")


@feat("reference_free", "input_pose", tol_abs=0.1, unit="m/s^2")
def traj_accel_rms(run, ref=None):
    """RMS of the change of keyframe velocity between consecutive keyframe intervals (m/s^2): path roughness."""
    p, t = _traj_xyz_t(run)
    if len(p) < 4:
        return float("nan")
    v = np.diff(p, axis=0) / np.diff(t)[:, None]
    tm = (t[1:] + t[:-1]) / 2
    a = np.diff(v, axis=0) / np.diff(tm)[:, None]
    return float(np.sqrt(np.mean(np.sum(a ** 2, axis=1))))


@feat("reference_free", "input_pose", tol_abs=2.0, unit="deg/s")
def traj_yaw_rate_p95(run, ref=None):
    """95th percentile of the absolute heading change rate (deg/s) between consecutive keyframes."""
    w, x, y, z = run.traj[:, 4], run.traj[:, 5], run.traj[:, 6], run.traj[:, 7]
    # camera optical frame: forward = +z axis of the camera; heading = angle of that axis in the xy plane
    fx = 2 * (x * z + w * y)
    fy = 2 * (y * z - w * x)
    yaw = np.unwrap(np.arctan2(fy, fx))
    t = run.traj[:, 0] / 1e9
    if len(yaw) < 5:
        return float("nan")
    return float(np.percentile(np.abs(np.degrees(np.diff(yaw)) / np.diff(t)), 95))


@feat("reference_free", "input_pose", tol_abs=0.02)
def traj_mesh_collision_frac(run, ref=None):
    """Fraction of trajectory keyframes whose position is closer than 0.5 m to a mesh vertex (the robot cannot be inside the surfaces it mapped)."""
    if len(run.traj) < 5:
        return float("nan")
    return float((run.mesh_tree.query(run.traj[:, 1:4], workers=1)[0] < 0.5).mean())


# =========================================================================== PROCESS TRACE (Hydra layer statistics)

def _churn(run, layer, col):
    s = run.stats_last(layer)
    return float(s[col] / max(s[0], 1)) if np.isfinite(s).all() else float("nan")


@feat("reference_free", "process", tol_abs=0.5, tol_rel=0.3)
def obj_removed_per_final(run, ref=None):
    """Cumulative number of object nodes removed during the run divided by the final object count (layer_2_statistics.csv)."""
    return _churn(run, 2, 1)


@feat("reference_free", "process", tol_abs=0.5, tol_rel=0.3)
def obj_merged_per_final(run, ref=None):
    """Cumulative number of object merges divided by the final object count."""
    return _churn(run, 2, 2)


@feat("reference_free", "process", tol_abs=0.5, tol_rel=0.3)
def place_removed_per_final(run, ref=None):
    """Cumulative number of place nodes removed divided by the final place count (layer_3_statistics.csv)."""
    return _churn(run, 3, 1)


@feat("reference_free", "process", tol_abs=0.2, tol_rel=0.3)
def place_merged_per_final(run, ref=None):
    """Cumulative number of place merges divided by the final place count."""
    return _churn(run, 3, 2)

# --------------------------------------------------------------------------- driver

def as_view(x) -> FeatureView:
    """FeatureView of a Snapshot; a FeatureView passes through."""
    return x if isinstance(x, FeatureView) else FeatureView(x)


def meta() -> list[dict]:
    """One dict per registered feature, in registry order.

    keys: name, kind, layer (tag), evidence_layer (dag.FEATURE_LAYER, None for edges / input_pose / process), ideal, tol_abs,
    tol_rel, count_based, unit, definition (the docstring).
    """
    return [dict(name=s.name, kind=s.kind, layer=s.layer, evidence_layer=dag.layer_of_feature(s.layer), ideal=s.ideal,
                 tol_abs=s.tol_abs, tol_rel=s.tol_rel, count_based=s.count_based, unit=s.unit, definition=s.definition)
            for s in REGISTRY.values()]


def compute(view, ref_view=None, names=None) -> tuple[dict, dict]:
    """(values, errors) of the registered features for one run, in registry order.

    A feature that raises, or returns +-inf, gets NaN and its error text in `errors`; a feature that returns NaN by its own
    definition (too little data) has no error entry. reference_based features are skipped when ref_view is None.
    `view` / `ref_view` may be Snapshot or FeatureView. Unknown names raise KeyError.
    """
    view = as_view(view)
    ref_view = None if ref_view is None else as_view(ref_view)
    if names is not None:
        names = set(names)
        unknown = names - set(REGISTRY)
        if unknown:
            raise KeyError(f"unknown features: {sorted(unknown)}")
    values, errors = {}, {}
    for name, spec in REGISTRY.items():
        if names is not None and name not in names:
            continue
        if spec.kind == "reference_based" and ref_view is None:
            continue
        try:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                v = float(spec.fn(view, ref_view if spec.kind == "reference_based" else None))
        except Exception as e:  # one failing feature must not stop the battery
            values[name], errors[name] = float("nan"), f"{type(e).__name__}: {e}"
            continue
        if math.isinf(v):
            values[name], errors[name] = float("nan"), "non-finite result (inf)"
        else:
            values[name] = v
    return values, errors


def _called_features(code, acc=None):
    """Registered feature names referenced by a code object, nested functions included."""
    acc = set() if acc is None else acc
    acc |= {n for n in code.co_names if n in REGISTRY}
    for c in code.co_consts:
        if hasattr(c, "co_names"):
            _called_features(c, acc)
    return acc


@lru_cache(maxsize=1)
def source_hashes() -> dict:
    """feature name -> short sha1 of its source, the source of the features it calls and all non-feature code of this module.

    Keys the cached feature values: any change that can alter a value changes the hash.
    """
    src = inspect.getsource(sys.modules[__name__])
    lines = src.splitlines(keepends=True)
    own, spans = {}, []
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in REGISTRY and node.decorator_list:
            a, b = node.decorator_list[0].lineno - 1, node.end_lineno
            own[node.name] = "".join(lines[a:b])
            spans.append((a, b))
    rest = "".join(ln for i, ln in enumerate(lines) if not any(a <= i < b for a, b in spans))
    base = hashlib.sha1(rest.encode()).hexdigest()
    out = {}
    for n in own:
        seen, todo = {n}, [n]
        while todo:
            for m in _called_features(REGISTRY[todo.pop()].fn.__code__) - seen:
                seen.add(m)
                todo.append(m)
        text = base + "".join(own[m] for m in sorted(seen))
        out[n] = hashlib.sha1(text.encode()).hexdigest()[:12]
    return out
