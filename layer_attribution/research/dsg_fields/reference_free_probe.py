#!/usr/bin/env python3
"""Cheap reference-free quantities from ONE Snapshot: feasibility and cost only, no fault evaluation.

Each quantity is a function of Snapshot fields (declared in `needs`), registered with its layer.
Per quantity the script reports the value, wall-clock cost without shared structures, and whether the needed fields are
populated. Shared structures (kd-trees, mesh edge list, place components) are built once and costed separately.
A quantity whose inputs are empty (e.g. no rooms) is reported as missing, never as an error; an exception is recorded
per quantity and does not stop the run.

usage (WSL, via research/dsg_fields/wsl_py.sh):
    reference_free_probe.py RUN [RUN ...] [--out-dir DIR]     values + costs for the named runs
    reference_free_probe.py --dev-cost [--out-dir DIR]        cost / feasibility over all finished development runs (no values kept)
    reference_free_probe.py --degenerate N                    N random degenerate graphs: every quantity gives a value or n/a, no errors
"""

import argparse
import json
import pathlib
import sys
import time
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from snapshot_prototype import TABLES, Snapshot, finished_dev_runs, resolve_run  # noqa: E402

REGISTRY = []
FLOOR, WALL, UNKNOWN = 3, 19, 0


def quantity(layer, name, needs):
    def deco(fn):
        REGISTRY.append((layer, name, needs, fn))
        return fn

    return deco


class Ctx:
    """Shared, lazily built structures; their build time is kept apart from the quantity that triggers them."""

    def __init__(self, snap, run_dir):
        self.s, self.run_dir, self._c, self.shared_ms, self._pending = snap, run_dir, {}, {}, 0.0

    def get(self, key, build):
        if key not in self._c:
            t = time.perf_counter()
            self._c[key] = build()
            ms = (time.perf_counter() - t) * 1000
            self.shared_ms[key] = round(ms, 1)
            self._pending += ms
        return self._c[key]

    def take_pending_ms(self):
        ms, self._pending = self._pending, 0.0
        return ms

    @property
    def mesh_kd(self):
        return self.get("mesh_kd", lambda: cKDTree(self.s.mesh.xyz))

    @property
    def mesh_edges(self):
        def build():
            f = self.s.mesh.faces.astype(np.int64)
            e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]]), axis=1)
            nv = len(self.s.mesh.xyz)
            uniq, counts = np.unique(e[:, 0] * nv + e[:, 1], return_counts=True)
            return np.stack([uniq // nv, uniq % nv], axis=1), counts

        return self.get("mesh_edges", build)

    @property
    def floor(self):
        """kd-tree on floor-labelled vertices in xy plus their heights; None when the mesh has no floor label."""
        def build():
            m = self.s.mesh
            sel = m.label == FLOOR
            return (cKDTree(m.xyz[sel][:, :2]), m.xyz[sel][:, 2]) if sel.any() else None

        return self.get("floor_kd", build)

    def floor_below(self, xyz, radius=0.75, tol=0.2):
        """Height of the highest floor-labelled vertex at or below z + tol within `radius` in xy (nan if none).
        The scene labels floor and ceiling both 'floor', so the nearest vertex in xy alone can be the ceiling."""
        xyz = np.asarray(xyz, dtype=np.float64)
        out = np.full(len(xyz), np.nan)
        fl = self.floor
        if fl is None:
            return out
        kd, z = fl
        rows = np.flatnonzero(np.isfinite(xyz).all(axis=1))  # invalid boxes have nan centres
        for i, hit in zip(rows, kd.query_ball_point(xyz[rows, :2], r=radius)):
            if hit:
                zz = z[hit]
                zz = zz[zz <= xyz[i, 2] + tol]
                if len(zz):
                    out[i] = zz.max()
        return out

    @property
    def place_cc(self):
        def build():
            pe = self.s.place_edges.ij
            n = len(self.s.places)
            a = sparse.coo_matrix((np.ones(len(pe)), (pe[:, 0], pe[:, 1])), shape=(n, n))
            return csgraph.connected_components(a, directed=False)

        return self.get("place_cc", build)


def pct(x, qs=(5, 50, 95)):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return {f"p{q}": float(np.percentile(x, q)) for q in qs} if len(x) else {}


# ------------------------------------------------------------------------------------------------ mesh
@quantity("mesh", "label_unknown_frac", ["mesh.label"])
def _(c):
    return {"frac": float(np.mean(c.s.mesh.label == UNKNOWN))}


@quantity("mesh", "label_histogram", ["mesh.label"])
def _(c):
    lab = c.s.mesh.label
    cnt = np.bincount(lab[lab < 256], minlength=1)
    return {"frac_by_label": {int(k): round(float(v) / len(lab), 5) for k, v in enumerate(cnt) if v}}


@quantity("mesh", "edge_label_flip_rate", ["mesh.faces", "mesh.label"])
def _(c):
    e, _ = c.mesh_edges
    lab = c.s.mesh.label
    return {"rate": float(np.mean(lab[e[:, 0]] != lab[e[:, 1]])), "n_edges": int(len(e))}


@quantity("mesh", "label_normal_consistency", ["mesh.faces", "mesh.xyz", "mesh.label"])
def _(c):
    """floor faces should have vertical normals, wall faces horizontal ones: tests labels against geometry."""
    x, f, lab = c.s.mesh.xyz.astype(np.float64), c.s.mesh.faces, c.s.mesh.label
    n = np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]])
    norm = np.linalg.norm(n, axis=1)
    nz = np.abs(n[:, 2]) / np.maximum(norm, 1e-12)
    pure = (lab[f[:, 0]] == lab[f[:, 1]]) & (lab[f[:, 1]] == lab[f[:, 2]]) & (norm > 1e-12)
    fl, wl = pure & (lab[f[:, 0]] == FLOOR), pure & (lab[f[:, 0]] == WALL)
    return {"floor_faces": int(fl.sum()), "floor_vertical_normal_frac": float(np.mean(nz[fl] > np.cos(np.radians(20)))) if fl.any() else None,
            "wall_faces": int(wl.sum()), "wall_horizontal_normal_frac": float(np.mean(nz[wl] < np.sin(np.radians(20)))) if wl.any() else None}


@quantity("mesh", "boundary_edge_frac", ["mesh.faces"])
def _(c):
    _, cnt = c.mesh_edges
    return {"boundary_frac": float(np.mean(cnt == 1)), "nonmanifold_frac": float(np.mean(cnt > 2))}


@quantity("mesh", "connected_components", ["mesh.faces"])
def _(c):
    e, _ = c.mesh_edges
    nv = len(c.s.mesh.xyz)
    a = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nv, nv))
    n, lab = csgraph.connected_components(a, directed=False)
    sizes = np.bincount(lab)
    return {"n": int(n), "largest_frac": float(sizes.max() / nv), "n_ge_100_vertices": int((sizes >= 100).sum())}


@quantity("mesh", "edge_length_and_area", ["mesh.xyz", "mesh.faces"])
def _(c):
    x, f = c.s.mesh.xyz, c.s.mesh.faces
    e = np.linalg.norm(x[f[:, 1]] - x[f[:, 0]], axis=1)
    area = 0.5 * np.linalg.norm(np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]]), axis=1)
    return {"edge_p50": float(np.median(e)), "edge_p99": float(np.percentile(e, 99)), "area_p99": float(np.percentile(area, 99)),
            "sliver_frac(area<1e-4)": float(np.mean(area < 1e-4))}


@quantity("mesh", "surface_thickness", ["mesh.xyz", "mesh.label"])
def _(c):
    """sqrt of the smallest covariance eigenvalue per 0.5 m cell: planar surfaces ~0, double walls / noise larger."""
    m, out = c.s.mesh, {}
    for nm, lab in (("floor", FLOOR), ("wall", WALL)):
        x = m.xyz[m.label == lab].astype(np.float64)
        if len(x) < 30:
            continue
        cell = np.floor(x / 0.5).astype(np.int64)
        _, inv = np.unique(cell, axis=0, return_inverse=True)
        inv = inv.ravel()
        n = np.bincount(inv)
        mean = np.stack([np.bincount(inv, x[:, k]) for k in range(3)], 1) / n[:, None]
        xc = x - mean[inv]
        cov = np.stack([np.stack([np.bincount(inv, xc[:, i] * xc[:, j]) for j in range(3)], 1) for i in range(3)], 1) / n[:, None, None]
        ok = n >= 30
        if ok.any():
            out[f"{nm}_thickness"] = pct(np.sqrt(np.maximum(np.linalg.eigvalsh(cov[ok])[:, 0], 0)), (50, 95)) | {"n_cells": int(ok.sum())}
    return out


@quantity("mesh", "floor_label_height_modes", ["mesh.xyz", "mesh.label"])
def _(c):
    """peaks of the z histogram of floor-labelled vertices (floor and ceiling in this scene): level structure and drift."""
    m = c.s.mesh
    z = m.xyz[m.label == FLOOR][:, 2]
    if len(z) == 0:
        return {}
    h, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.05, 0.05))
    share = h / len(z)
    peaks = [i for i in range(len(h)) if share[i] >= 0.03 and (i == 0 or h[i] >= h[i - 1]) and (i == len(h) - 1 or h[i] > h[i + 1])]
    return {"peaks_z_share": [[round(float(edges[i] + 0.025), 3), round(float(share[i]), 3)] for i in peaks]}


# ------------------------------------------------------------------------------------------------ objects
@quantity("objects", "coincident_same_label", ["objects.pos", "objects.label"])
def _(c):
    o = c.s.objects
    kd = cKDTree(o.pos)
    out = {}
    for r in (0.5, 1.0):
        pairs = kd.query_pairs(r, output_type="ndarray")
        same = pairs[o.label[pairs[:, 0]] == o.label[pairs[:, 1]]] if len(pairs) else pairs
        out[f"pairs_r{r}"] = int(len(same))
    out["n_objects"] = int(len(o))
    return out


@quantity("objects", "bbox_size_extremes", ["objects.bbox_dim"])
def _(c):
    d = c.s.objects.bbox_dim
    vol = d.prod(1)
    return {"min_dim<0.05": int((d.min(1) < 0.05).sum()), "max_dim>3": int((d.max(1) > 3).sum()), "vol_median": float(np.nanmedian(vol)),
            "vol_p95": float(np.nanpercentile(vol, 95))}


@quantity("objects", "mesh_label_purity", ["objects.mesh_conn_idx", "objects.label", "mesh.label"])
def _(c):
    o, lab = c.s.objects, c.s.mesh.label
    pur = np.array([np.mean(lab[o.csr("mesh_conn_indptr", "mesh_conn_idx", i)] == o.label[i]) if o.n_mesh_conn[i] else np.nan for i in range(len(o))])
    return {"mean": float(np.nanmean(pur)), "min": float(np.nanmin(pur)), "n_lt_0.8": int((pur < 0.8).sum()), "n_lt_0.5": int((pur < 0.5).sum()),
            "n_without_vertices": int(np.isnan(pur).sum())}


@quantity("objects", "bbox_vs_connected_vertices", ["objects.bbox_center", "objects.bbox_dim", "objects.mesh_conn_idx", "mesh.xyz"])
def _(c):
    o, x = c.s.objects, c.s.mesh.xyz
    err = []
    for i in range(len(o)):
        if not o.n_mesh_conn[i]:
            continue
        v = x[o.csr("mesh_conn_indptr", "mesh_conn_idx", i)]
        lo, hi = o.bbox_center[i] - o.bbox_dim[i] / 2, o.bbox_center[i] + o.bbox_dim[i] / 2
        err.append(max(np.abs(lo - v.min(0)).max(), np.abs(hi - v.max(0)).max()))
    return {"max_edge_err_m": pct(err, (50, 95)) | {"max": float(np.max(err))}} if err else {}


@quantity("objects", "orphans_and_multi_parent", ["objects.n_parents"])
def _(c):
    o = c.s.objects
    return {"no_parent": int((o.n_parents == 0).sum()), "multi_parent": int((o.n_parents > 1).sum()),
            "parents_in_different_rooms": int(sum(len(set(c.s.places.room[o.csr("parent_indptr", "parent_idx", i)])) > 1 for i in range(len(o))))}


@quantity("objects", "height_above_floor", ["objects.bbox_center", "objects.bbox_dim", "mesh.label", "mesh.xyz"])
def _(c):
    o = c.s.objects
    ctr = o.bbox_center.astype(np.float64)
    bottom = ctr[:, 2] - o.bbox_dim[:, 2] / 2
    h = bottom - c.floor_below(np.column_stack([ctr[:, :2], ctr[:, 2]]))
    return {"bottom_minus_floor": pct(h, (5, 50, 95)), "n_below_-0.2": int((h < -0.2).sum()), "n_above_1.5": int((h > 1.5).sum()), "n_no_floor": int(np.isnan(h).sum())}


@quantity("objects", "parent_place_distance", ["objects.parent_place", "places.pos", "places.dist"])
def _(c):
    o, p = c.s.objects, c.s.places
    ok = o.parent_place >= 0
    d = np.linalg.norm(o.pos[ok] - p.pos[o.parent_place[ok]], axis=1)
    return {"dist_m": pct(d, (50, 95)), "dist_over_radius": pct(d / p.dist[o.parent_place[ok]], (50, 95))}


@quantity("objects", "nearest_place_room_agreement", ["objects.pos", "objects.room", "places.pos", "places.room"])
def _(c):
    """cross-layer: the room of the place nearest to an object should be the object's own room."""
    o, p = c.s.objects, c.s.places
    d, i = cKDTree(p.pos).query(o.pos)
    ok = o.room >= 0
    return {"agree_frac": float(np.mean(p.room[i[ok]] == o.room[ok])) if ok.any() else None, "n_disagree": int((p.room[i[ok]] != o.room[ok]).sum()),
            "nearest_place_dist_m": pct(d, (50, 95))}


# ------------------------------------------------------------------------------------------------ places
@quantity("places", "clearance_vs_mesh", ["places.pos", "places.dist", "mesh.xyz"])
def _(c):
    p = c.s.places
    d, _ = c.mesh_kd.query(p.pos)
    gap = d - p.dist
    return {"mesh_dist_minus_radius": pct(gap, (1, 5, 50)), "frac_gap<-0.1": float(np.mean(gap < -0.1)), "frac_gap<-0.3": float(np.mean(gap < -0.3))}


@quantity("places", "radius_vs_basis_points", ["places.dist", "places.basis_pos", "places.basis_indptr"])
def _(c):
    p = c.s.places
    owner = np.repeat(np.arange(len(p)), np.diff(p.basis_indptr))
    d = np.linalg.norm(p.basis_pos - p.pos[owner], axis=1)
    mn = np.full(len(p), np.inf)
    np.minimum.at(mn, owner, d)
    err = np.abs(mn - p.dist)
    err = err[np.isfinite(err)]
    return {"frac_|radius-min_basis_dist|>1e-3": float(np.mean(err > 1e-3)), "frac_>0.1": float(np.mean(err > 0.1)), "n_basis_mean": float(p.n_basis.mean())}


@quantity("places", "basis_points_on_mesh", ["places.basis_pos", "mesh.xyz"])
def _(c):
    d, _ = c.mesh_kd.query(c.s.places.basis_pos)
    return {"dist_to_mesh": pct(d, (50, 95, 99))}


@quantity("places", "graph_components_and_degree", ["place_edges.ij", "places.deg"])
def _(c):
    n, lab = c.place_cc
    sizes = np.bincount(lab)
    deg = c.s.places.deg
    return {"n_components": int(n), "largest_frac": float(sizes.max() / len(lab)), "isolated": int((deg == 0).sum()), "deg_min_med_max": [int(deg.min()), float(np.median(deg)), int(deg.max())]}


@quantity("places", "edge_weight_vs_endpoint_radius", ["place_edges.ij", "place_edges.w", "places.dist"])
def _(c):
    p, pe = c.s.places, c.s.place_edges
    r = np.minimum(p.dist[pe.ij[:, 0]], p.dist[pe.ij[:, 1]])
    return {"weight<=min_endpoint_radius_frac": float(np.mean(pe.w <= r + 1e-6)), "weight_p50": float(np.median(pe.w))}


@quantity("places", "edges_through_geometry", ["place_edges.ij", "places.pos", "places.dist", "mesh.xyz"])
def _(c):
    p, ij = c.s.places, c.s.place_edges.ij
    a, b = p.pos[ij[:, 0]], p.pos[ij[:, 1]]
    r = np.minimum(p.dist[ij[:, 0]], p.dist[ij[:, 1]])
    worst = np.full(len(ij), np.inf)
    for t in (0.25, 0.5, 0.75):
        d, _ = c.mesh_kd.query(a + t * (b - a))
        worst = np.minimum(worst, d / r)
    return {"worst_mesh_dist_over_min_radius": pct(worst, (1, 5, 50)), "frac<0.5": float(np.mean(worst < 0.5)), "frac<0.25": float(np.mean(worst < 0.25))}


@quantity("places", "near_duplicates", ["places.pos"])
def _(c):
    pairs = cKDTree(c.s.places.pos).query_pairs(0.15, output_type="ndarray")
    return {"pairs_r0.15": int(len(pairs))}


@quantity("places", "centers_inside_object_bbox", ["places.pos", "objects.bbox_center", "objects.bbox_dim"])
def _(c):
    """cross-layer: free-space place centres should not lie inside an object bounding box."""
    p, o = c.s.places.pos, c.s.objects
    inside = np.zeros(len(p), dtype=bool)
    for i in range(len(o)):
        if np.isfinite(o.bbox_dim[i]).all():
            inside |= np.all(np.abs(p - o.bbox_center[i]) <= o.bbox_dim[i] / 2, axis=1)
    return {"n_inside": int(inside.sum()), "frac": float(inside.mean())}


@quantity("places", "trajectory_coverage", ["traj.pos", "places.pos", "places.dist", "traj.place"])
def _(c):
    t, p = c.s.traj, c.s.places
    d, _ = cKDTree(p.pos).query(t.pos)
    ok = t.place >= 0
    dp = np.linalg.norm(t.pos[ok] - p.pos[t.place[ok]], axis=1)
    return {"pose_to_nearest_place_m": pct(d, (50, 95)) | {"max": float(d.max())},
            "pose_outside_parent_sphere_frac": float(np.mean(dp > p.dist[t.place[ok]])) if ok.any() else None, "poses_without_parent": int((~ok).sum())}


# ------------------------------------------------------------------------------------------------ mesh places
@quantity("mplaces", "mesh_label_purity_and_coverage", ["mplaces.mesh_conn_idx", "mplaces.label", "mesh.label"])
def _(c):
    mp, lab = c.s.mplaces, c.s.mesh.label
    owner = np.repeat(np.arange(len(mp)), np.diff(mp.mesh_conn_indptr))
    same = lab[mp.mesh_conn_idx] == mp.label[owner]
    floor_idx = np.flatnonzero(lab == FLOOR)
    cov = np.isin(floor_idx, mp.mesh_conn_idx).mean() if len(floor_idx) else None
    return {"vertex_label_matches_node_label": float(same.mean()), "floor_vertices_covered": None if cov is None else float(cov),
            "vertices_claimed_by_>1_node": float(1 - len(np.unique(mp.mesh_conn_idx)) / max(1, len(mp.mesh_conn_idx)))}


@quantity("mplaces", "graph_components_and_height", ["mplace_edges.ij", "mplaces.deg", "mplaces.pos"])
def _(c):
    mp = c.s.mplaces
    ij, n = c.s.mplace_edges.ij, len(mp)
    a = sparse.coo_matrix((np.ones(len(ij)), (ij[:, 0], ij[:, 1])), shape=(n, n))
    k, lab = csgraph.connected_components(a, directed=False)
    return {"n_components": int(k), "largest_frac": float(np.bincount(lab).max() / n), "isolated": int((mp.deg == 0).sum()), "z_std": float(mp.pos[:, 2].std())}


# ------------------------------------------------------------------------------------------------ rooms
@quantity("rooms", "contiguity", ["rooms.child_idx", "place_edges.ij", "places.room"])
def _(c):
    p, ij = c.s.places, c.s.place_edges.ij
    intra = ij[p.room[ij[:, 0]] == p.room[ij[:, 1]]]
    n = len(p)
    a = sparse.coo_matrix((np.ones(len(intra)), (intra[:, 0], intra[:, 1])), shape=(n, n))
    _, lab = csgraph.connected_components(a, directed=False)
    ncomp = [len(np.unique(lab[c.s.rooms.csr("child_indptr", "child_idx", r)])) for r in range(len(c.s.rooms))]
    return {"components_per_room": ncomp, "rooms_with_gt1": int(sum(k > 1 for k in ncomp)), "cross_room_place_edges": int(len(ij) - len(intra))}


@quantity("rooms", "orphan_places_and_objects", ["places.room"])
def _(c):
    """places without a room parent exist in some runs (every place has one in the clean run)."""
    p, o = c.s.places, c.s.objects
    return {"places_without_room": int((p.room < 0).sum()), "frac": float(np.mean(p.room < 0)), "places_with_several_rooms": int((p.n_room_parents > 1).sum()),
            "objects_without_room": int((o.room < 0).sum())}


@quantity("rooms", "adjacency_vs_place_edges", ["room_edges.ij", "place_edges.ij", "places.room"])
def _(c):
    p, ij = c.s.places, c.s.place_edges.ij
    ra, rb = p.room[ij[:, 0]], p.room[ij[:, 1]]
    cross = {tuple(sorted(x)) for x in zip(ra[ra != rb].tolist(), rb[ra != rb].tolist())}
    rooms = {tuple(sorted(x)) for x in c.s.room_edges.ij.tolist()}
    return {"room_edges": len(rooms), "room_edges_without_place_edge": len(rooms - cross), "place_edge_room_pairs_without_room_edge": len(cross - rooms)}


@quantity("rooms", "size_and_overlap", ["rooms.n_children", "rooms.ext_min", "rooms.ext_max"])
def _(c):
    r = c.s.rooms
    lo, hi = r.ext_min[:, :2], r.ext_max[:, :2]
    area = np.prod(hi - lo, axis=1)
    inter = np.zeros((len(r), len(r)))
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            w = np.minimum(hi[i], hi[j]) - np.maximum(lo[i], lo[j])
            inter[i, j] = np.prod(np.maximum(w, 0))
    return {"n_children_min_med_max": [int(r.n_children.min()), float(np.median(r.n_children)), int(r.n_children.max())], "tiny_lt10": int((r.n_children < 10).sum()),
            "extent_area_m2_min_med_max": [float(np.nanmin(area)), float(np.nanmedian(area)), float(np.nanmax(area))],
            "max_pair_overlap_over_smaller_area": float(max((inter[i, j] / min(area[i], area[j]) for i in range(len(r)) for j in range(i + 1, len(r))
                                                            if min(area[i], area[j]) > 0), default=0.0))}


@quantity("rooms", "pos_vs_children_centroid", ["rooms.pos", "rooms.child_idx", "places.pos"])
def _(c):
    r, p = c.s.rooms, c.s.places
    d = [np.linalg.norm(r.pos[i] - p.pos[r.csr("child_indptr", "child_idx", i)].mean(0)) for i in range(len(r)) if r.n_children[i]]
    return {"dist_m": pct(d, (50, 95)) | {"max": float(np.max(d))}} if d else {}


@quantity("rooms", "objectless_rooms", ["objects.room", "rooms.id"])
def _(c):
    cnt = np.bincount(c.s.objects.room[c.s.objects.room >= 0], minlength=len(c.s.rooms))
    return {"objects_per_room": cnt.tolist(), "rooms_without_objects": int((cnt == 0).sum())}


@quantity("building", "pos_vs_rooms", ["building.pos", "rooms.pos"])
def _(c):
    b, r = c.s.building, c.s.rooms
    return {"dist_to_mean_room_pos_m": float(np.linalg.norm(b.pos[0] - r.pos.mean(0))), "n_building_nodes": int(len(b))}


# ------------------------------------------------------------------------------------------------ trajectory
@quantity("traj", "timing", ["traj.t_ns"])
def _(c):
    dt = np.diff(c.s.traj.t_ns) / 1e9
    if len(dt) == 0:
        return {}
    med = np.median(dt)
    return {"dt_median_s": float(med), "dt_max_s": float(dt.max()), "n_dt_gt_2x_median": int((dt > 2 * med).sum()), "monotonic": bool((dt > 0).all())}


@quantity("traj", "step_and_smoothness", ["traj.pos", "traj.quat_wxyz", "traj.t_ns"])
def _(c):
    t = c.s.traj
    if len(t.t_ns) < 3:
        return {}
    step = np.linalg.norm(np.diff(t.pos, axis=0), axis=1)
    dt = np.maximum(np.diff(t.t_ns) / 1e9, 1e-9)
    acc = np.linalg.norm(np.diff(t.pos, 2, axis=0), axis=1)
    q = t.quat_wxyz
    dang = 2 * np.degrees(np.arccos(np.clip(np.abs(np.sum(q[1:] * q[:-1], axis=1)), 0, 1)))
    return {"step_m": pct(step, (50, 99)) | {"max": float(step.max())}, "speed_mps_p99": float(np.percentile(step / dt, 99)),
            "second_diff_m": pct(acc, (50, 99)), "rot_step_deg": pct(dang, (50, 99)) | {"max": float(dang.max())}}


@quantity("traj", "height_above_floor", ["traj.pos", "mesh.label", "mesh.xyz"])
def _(c):
    h = c.s.traj.pos[:, 2] - c.floor_below(c.s.traj.pos, radius=1.0)
    return {"pose_z_minus_floor": pct(h, (5, 50, 95)) | ({"std": float(np.nanstd(h))} if np.isfinite(h).any() else {}), "n_no_floor": int(np.isnan(h).sum())}


@quantity("traj", "clearance_vs_mesh", ["traj.pos", "mesh.xyz"])
def _(c):
    d, _ = c.mesh_kd.query(c.s.traj.pos)
    return {"dist_to_mesh_m": pct(d, (1, 5, 50)), "frac<0.3": float(np.mean(d < 0.3))}


# ------------------------------------------------------------------------------------------------ run files outside the DSG
@quantity("files", "layer_statistics_final_rows", ["layer_*_statistics.csv"])
def _(c):
    out = {}
    for side in ("frontend", "backend"):
        for layer, nm in ((2, "objects"), (3, "places")):
            f = c.run_dir / side / f"layer_{layer}_statistics.csv"
            if not f.exists():
                return {"missing_file": str(f.relative_to(c.run_dir))}
            rows = f.read_text().strip().splitlines()
            if len(rows) < 2:
                return {"empty_file": str(f.relative_to(c.run_dir))}
            out[f"{side}_{nm}"] = dict(zip(rows[0].split(","), [int(x) for x in rows[-1].split(",")]))
    fo, bo = out["frontend_objects"]["nodes_active"], out["backend_objects"]["nodes_active"]
    fp, bp = out["frontend_places"]["nodes_active"], out["backend_places"]["nodes_active"]
    out["backend_over_frontend"] = {"objects": bo / fo if fo else None, "places": bp / fp if fp else None}
    return out


# ------------------------------------------------------------------------------------------------ driver
def has_fields(snap, needs):
    missing = []
    for need in needs:
        t, _, f = need.partition(".")
        if t not in TABLES:
            continue  # run file, not a snapshot field
        try:
            arr = getattr(snap._tables[t], f)
            if arr.size == 0 or (arr.dtype.kind == "f" and not np.isfinite(arr).any()):
                missing.append(need)
        except (AttributeError, KeyError):
            missing.append(need)
    return missing


def run_probe(run, verbose=True, cache_root=None):
    run_dir = resolve_run(run)
    t0 = time.perf_counter()
    with Snapshot.open(run, **({"cache_root": cache_root} if cache_root else {})) as snap:
        t_open = time.perf_counter() - t0
        snap.materialize()
    ctx = Ctx(snap, run_dir)
    res = {"_open_seconds": round(t_open, 3)}
    if verbose:
        print(f"== {run}: Snapshot open {t_open:.2f}s | {snap.summary()}")
        print(f"{'layer':9s} {'quantity':36s} {'ms':>8s}  value")
    for layer, name, needs, fn in REGISTRY:
        key = f"{layer}.{name}"
        miss = has_fields(snap, needs)
        if miss:
            res[key] = {"ms": None, "missing": miss}
            if verbose:
                print(f"{layer:9s} {name:36s} {'n/a':>8s}  missing/unpopulated: {miss}")
            continue
        t = time.perf_counter()
        try:
            val = fn(ctx)
        except Exception as e:  # recorded, the remaining quantities still run
            res[key] = {"ms": None, "error": f"{type(e).__name__}: {str(e)[:160]}"}
            ctx.take_pending_ms()
            if verbose:
                print(f"{layer:9s} {name:36s} {'ERROR':>8s}  {res[key]['error']}")
            continue
        ms = (time.perf_counter() - t) * 1000 - ctx.take_pending_ms()
        res[key] = {"ms": round(ms, 1), "value": val}
        if verbose:
            print(f"{layer:9s} {name:36s} {ms:8.1f}  {json.dumps(val, default=float)}")
    res["_shared_structures_ms"] = ctx.shared_ms
    res["_total_ms"] = round(sum(v["ms"] for v in res.values() if isinstance(v, dict) and v.get("ms")) + sum(ctx.shared_ms.values()), 1)
    if verbose:
        print(f"shared structures (ms): {ctx.shared_ms}")
        print(f"total {res['_total_ms']:.0f} ms (quantities + shared structures, snapshot already loaded)")
    return res


def dev_cost(out_dir):
    """Feasibility and cost over every finished development run; values are not kept."""
    runs, skipped = finished_dev_runs()
    per = {}
    shared = {}
    for r in runs:
        try:
            res = run_probe(r.name, verbose=False)
        except Exception as e:  # snapshot could not be built
            per.setdefault("_snapshot_errors", []).append(f"{r.name}: {type(e).__name__}: {str(e)[:100]}")
            continue
        for k, v in res.items():
            if k.startswith("_") or not isinstance(v, dict):
                continue
            d = per.setdefault(k, {"ok": 0, "missing": 0, "error": 0, "ms": [], "errors": []})
            if "value" in v:
                d["ok"] += 1
                d["ms"].append(v["ms"])
            elif "missing" in v:
                d["missing"] += 1
            else:
                d["error"] += 1
                d["errors"].append(f"{r.name}: {v.get('error')}")
        for k, ms in res["_shared_structures_ms"].items():
            shared.setdefault(k, []).append(ms)
    summary = {"n_runs": len(runs), "skipped": skipped, "quantities": {}, "shared_structures_ms": {k: {"median": float(np.median(v)), "max": float(np.max(v))} for k, v in shared.items()}}
    print(f"{len(runs)} development runs; skipped (counts): {skipped}")
    print(f"{'quantity':52s} {'ok':>3s} {'n/a':>4s} {'err':>4s} {'median ms':>10s} {'max ms':>8s}")
    for k, d in per.items():
        if k == "_snapshot_errors":
            continue
        summary["quantities"][k] = {"ok": d["ok"], "missing": d["missing"], "error": d["error"], "ms_median": float(np.median(d["ms"])) if d["ms"] else None,
                                    "ms_max": float(np.max(d["ms"])) if d["ms"] else None, "errors": d["errors"][:3]}
        print(f"{k:52s} {d['ok']:3d} {d['missing']:4d} {d['error']:4d} {np.median(d['ms']) if d['ms'] else float('nan'):10.1f} {np.max(d['ms']) if d['ms'] else float('nan'):8.1f}")
    print("shared structures median/max ms:", {k: (round(v['median']), round(v['max'])) for k, v in summary["shared_structures_ms"].items()})
    if per.get("_snapshot_errors"):
        print("snapshot errors:", per["_snapshot_errors"])
        summary["snapshot_errors"] = per["_snapshot_errors"]
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "reference_free_cost_dev_runs.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf8")
    return summary


def degenerate(n_graphs, seed=1):
    """Every quantity on random degenerate graphs (empty layers, orphans, tiny meshes): must report values or n/a, never an error."""
    import tempfile

    import spark_dsg as dsg
    from snapshot_prototype import _random_graph, _write_run

    rng = np.random.default_rng(seed)
    outcome, errors = {}, {}
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        for k in range(n_graphs):
            g, _, _ = _random_graph(dsg, rng)
            _write_run(td / f"g{k}", g, None)
            for key, v in run_probe(td / f"g{k}", verbose=False, cache_root=td / "cache").items():
                if key.startswith("_") or not isinstance(v, dict):
                    continue
                state = "ok" if "value" in v else "missing" if "missing" in v else "error"
                outcome.setdefault(key, {"ok": 0, "missing": 0, "error": 0})[state] += 1
                if state == "error":
                    errors.setdefault(key, set()).add(v["error"])
    bad = {k: sorted(v) for k, v in errors.items()}
    print(f"{n_graphs} degenerate graphs: quantity evaluations ok {sum(v['ok'] for v in outcome.values())}, "
          f"n/a {sum(v['missing'] for v in outcome.values())}, errors {sum(v['error'] for v in outcome.values())}")
    for k, v in bad.items():
        print(f"  ERROR {k}: {v}")
    return outcome, bad


def main():
    warnings.simplefilter("ignore", RuntimeWarning)  # empty-slice nan reductions are reported as missing values instead
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--dev-cost", action="store_true")
    ap.add_argument("--degenerate", type=int, default=0, metavar="N", help="run on N random degenerate graphs")
    ap.add_argument("--out-dir", type=pathlib.Path, default=None)
    args = ap.parse_args()
    if args.degenerate:
        degenerate(args.degenerate)
        return
    if args.dev_cost:
        dev_cost(args.out_dir)
        return
    results = {r: run_probe(r) for r in args.runs}
    if len(args.runs) == 2:
        a, b = (results[r] for r in args.runs)
        same = {k: a[k].get("value") == b[k].get("value") for k in a if isinstance(a[k], dict) and "value" in a[k]}
        print(f"\nidentical values between {args.runs[0]} and {args.runs[1]}: {sum(same.values())}/{len(same)}", [k for k, v in same.items() if not v])
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "reference_free_probe.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf8")


if __name__ == "__main__":
    sys.exit(main())
