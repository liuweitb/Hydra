#!/usr/bin/env python3
"""Known-answer checks for features_proto on synthetic snapshots. Run: python selftest_features.py"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import features_proto as fp  # noqa: E402


def fake(**arrays):
    s = fp.Snap.__new__(fp.Snap)
    s.a = arrays
    s.name = "syn"
    s.manifest = {}
    return s


def plane(n=40000, noise=0.0, z0=1.0, t=None, seed=0, label=fp.LABEL_FLOOR, size=10.0):
    r = np.random.default_rng(seed)
    xyz = np.c_[r.uniform(0, size, n), r.uniform(0, size, n), z0 + r.normal(0, noise, n) if noise else np.full(n, z0)]
    return xyz.astype(np.float32), np.full(n, label, np.int16), (np.zeros(n, np.int64) if t is None else t)


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name, detail)
    return ok


def main():
    ok = True
    ok &= check("js identical", fp.js_divergence([1, 2, 3], [2, 4, 6]) < 1e-12)
    ok &= check("js disjoint = 1 bit", abs(fp.js_divergence([1, 0], [0, 1]) - 1) < 1e-12)
    c = np.zeros((1, 3))
    d = np.ones((1, 3))
    ok &= check("iou identical", abs(fp.aabb_iou(c, d, c, d)[0] - 1) < 1e-12)
    ok &= check("iou half overlap = 1/3", abs(fp.aabb_iou(c, d, c + [0.5, 0, 0], d)[0] - 1 / 3) < 1e-12)

    # partition agreement: identical -> 1, crossed 2x2 -> -0.5
    pos = np.array([[0, 0, 0], [0.1, 0, 0], [5, 0, 0], [5.1, 0, 0]], float)

    def snap_p(par):
        s = fake(place_pos=pos, place_parent=np.array(par))
        s.place_tree = fp.cKDTree(pos)
        return s

    ok &= check("ari identical", abs(fp.room_partition_ari(snap_p([0, 0, 1, 1]), snap_p([0, 0, 1, 1])) - 1) < 1e-12)
    ok &= check("ari crossed = -0.5", abs(fp.room_partition_ari(snap_p([0, 1, 0, 1]), snap_p([0, 0, 1, 1])) + 0.5) < 1e-12)

    # free-space iou
    pp = np.array([[0, 0, 0], [10, 0, 0]], float)
    rr = np.array([1.0, 1.0])
    a = fake(place_pos=pp, place_radius=rr)
    b = fake(place_pos=pp + [100, 0, 0], place_radius=rr)
    ok &= check("freespace iou identical", abs(fp.place_freespace_iou(a, a) - 1) < 1e-12)
    ok &= check("freespace iou disjoint", fp.place_freespace_iou(a, b) == 0)

    # roughness: flat plane ~ 0, noisy plane much larger
    for noise, lo, hi in ((0.0, 0, 1e-6), (0.02, 1e-3, 1)):
        xyz, lab, t = plane(noise=noise)
        s = fake(mesh_xyz=xyz, mesh_label=lab, mesh_t_first=t)
        v = fp.mesh_roughness_median(s)
        ok &= check(f"roughness noise={noise}", lo <= v <= hi, f"{v:.2e}")

    # floor drift: z grows linearly with first-seen time -> time range ~ total drift; no drift -> ~0
    n = 40000
    t = np.random.default_rng(3).uniform(0, 500, n)
    for drift, lo, hi in ((0.0, 0, 0.005), (0.3, 0.2, 0.35)):
        xyz, lab, _ = plane(n=n)
        xyz[:, 2] += (drift * t / 500).astype(np.float32)
        s = fake(mesh_xyz=xyz, mesh_label=lab, mesh_t_first=(t * 1e9).astype(np.int64))
        v = fp.floor_z_time_range(s)
        ok &= check(f"floor_z_time_range drift={drift}", lo <= v <= hi, f"{v:.3f}")

    # revisit offset: two sheets first seen 100 s apart, interleaved so each neighbourhood holds both.
    # The PCA normal tilts to absorb part of the offset, so the estimate is biased low (0.5-1.0 x truth).
    n = 40000
    r = np.random.default_rng(4)
    base_xyz = np.c_[r.uniform(0, 4, n), r.uniform(0, 4, n), np.zeros(n)]
    second = r.random(n) < 0.5
    tt = np.where(second, 200e9, 100e9).astype(np.int64)
    est, est96 = [], []
    for off in (0.0, 0.05, 0.2):
        xyz = base_xyz.copy()
        xyz[second, 2] = off
        s = fake(mesh_xyz=xyz.astype(np.float32), mesh_label=np.full(n, 3, np.int16), mesh_t_first=tt)
        est.append(fp.mesh_revisit_offset_median(s))
        est96.append(fp.mesh_revisit_offset_median_k96(s))
    ok &= check("revisit offset: zero sheets apart", est[0] < 0.01, f"{est[0]:.3f}")
    ok &= check("revisit offset: 0.05 m", 0.025 < est[1] < 0.055, f"{est[1]:.3f}")
    ok &= check("revisit offset k24: 0.20 m is beyond the neighbourhood (nan)", np.isnan(est[2]))
    # coarser sampling like the real mesh (about 0.06 m vertex spacing) so 96 neighbours reach 0.2 m
    n2 = 6000
    r2 = np.random.default_rng(5)
    b2 = np.c_[r2.uniform(0, 4, n2), r2.uniform(0, 4, n2), np.zeros(n2)]
    sec2 = r2.random(n2) < 0.5
    t2 = np.where(sec2, 200e9, 100e9).astype(np.int64)
    est96 = []
    for off in (0.0, 0.05, 0.2):
        xyz = b2.copy()
        xyz[sec2, 2] = off
        est96.append(fp.mesh_revisit_offset_median_k96(fake(mesh_xyz=xyz.astype(np.float32), mesh_label=np.full(n2, 3, np.int16), mesh_t_first=t2)))
    ok &= check("revisit offset k96: zero / 0.05 / 0.20 m", est96[0] < 0.01 and 0.025 < est96[1] < 0.055 and 0.08 < est96[2] < 0.21,
                f"{est96[0]:.3f} {est96[1]:.3f} {est96[2]:.3f}")
    s2 = fake(mesh_xyz=base_xyz.astype(np.float32), mesh_label=np.full(n, 3, np.int16), mesh_t_first=np.full(n, 100e9, np.int64))
    ok &= check("revisit offset nan without time gap", np.isnan(fp.mesh_revisit_offset_median(s2)))

    # floor_time_structure: ordered-in-time scatter scores high, noise-limited floor low
    n = 40000
    t = np.random.default_rng(6).uniform(0, 500, n)
    xyz, lab, _ = plane(n=n, noise=0.002)
    low = fp.floor_time_structure(fake(mesh_xyz=xyz, mesh_label=lab, mesh_t_first=(t * 1e9).astype(np.int64)))
    xyz2 = xyz.copy()
    xyz2[:, 2] += (0.3 * t / 500).astype(np.float32)
    high = fp.floor_time_structure(fake(mesh_xyz=xyz2, mesh_label=lab, mesh_t_first=(t * 1e9).astype(np.int64)))
    ok &= check("floor_time_structure: noise-limited < 0.5, drifting > 1.5", low < 0.5 and high > 1.5, f"{low:.2f} {high:.2f}")

    # free volume: identical = 1, doubling every radius multiplies the union volume by roughly 8
    pp = np.array([[0, 0, 0], [10, 0, 0]], float)
    a = fake(place_pos=pp, place_radius=np.array([1.0, 1.0]))
    b = fake(place_pos=pp, place_radius=np.array([2.0, 2.0]))
    ok &= check("freevol ratio identical", abs(fp.place_freevol_ratio(a, a) - 1) < 1e-12)
    r = fp.place_freevol_ratio(b, a)
    ok &= check("freevol ratio radius x2 about 8", 5 < r < 11, f"{r:.2f}")

    # object duplicate / overlap
    s = fake(obj_pos=np.array([[0, 0, 0], [0.3, 0, 0], [5, 0, 0]], float), obj_label=np.array([5, 5, 7]),
             obj_bbox_center=np.array([[0, 0, 0], [0.3, 0, 0], [5, 0, 0]], float), obj_bbox_dim=np.ones((3, 3)))
    ok &= check("dup frac 2/3", abs(fp.obj_dup_same_class_frac_050(s) - 2 / 3) < 1e-12)
    s.a["obj_label"] = np.array([5, 7, 7])
    ok &= check("diff-class overlap 2/3", abs(fp.obj_overlap_diffclass_frac(s) - 2 / 3) < 1e-12)

    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
