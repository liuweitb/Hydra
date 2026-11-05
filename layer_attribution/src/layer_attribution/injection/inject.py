#!/usr/bin/env python3
"""Inject exactly one fault into a Hydra MP3D-format dataset folder and write a ground-truth manifest."""
import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import tifffile
import yaml
from PIL import Image
from scipy import ndimage
from scipy.spatial.transform import Rotation

LAYER_ORDER = ["mesh", "objects", "places", "rooms", "buildings"]
CHANNELS = {
    "color": "rgb_{:07d}.png",
    "depth": "depth_{:07d}.tiff",
    "labels": "labels_{:07d}.png",
}
# ade20k_mp3d label space (Hydra config/label_spaces/ade20k_mp3d_label_space.yaml)
OBJECT_LABELS = [3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 25,
                 27, 28, 30, 31, 32, 33, 34, 35, 36, 37]
# look-alike pairs inside the object set so a confused object stays an object
CONFUSION_MAP = {3: 19, 5: 34, 6: 21, 7: 30, 8: 36, 10: 32, 11: 10, 12: 31, 13: 7,
                 14: 37, 15: 18, 18: 15, 19: 3, 20: 36, 21: 6, 22: 35, 23: 25,
                 25: 23, 27: 34, 28: 37, 30: 7, 31: 12, 32: 10, 33: 6, 34: 5,
                 35: 22, 36: 8, 37: 14}
LABEL_COMMON = {"object_labels": OBJECT_LABELS, "void_label": 0, "granularity": "blob"}


# ---------------------------------------------------------------- file io

def _read_img(path):
    if path.suffix == ".tiff":
        return tifffile.imread(path)
    with Image.open(path) as im:
        return np.array(im)


def _write_img(path, arr):
    if path.suffix == ".tiff":
        tifffile.imwrite(path, arr)
    else:
        Image.fromarray(arr).save(path)


def _place(src, dst, copy):
    # hard link keeps unchanged channels free; copy fallback covers cross-volume or FAT
    if not copy:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def _read_poses(src):
    with open(src / "poses.csv", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def _write_poses(dst, fields, rows):
    with open(dst / "poses.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _frame_path(root, channel, idx):
    return root / channel / CHANNELS[channel].format(idx)


# ---------------------------------------------------------------- image helpers

def _differs(a, b):
    d = a != b
    if a.dtype.kind == "f":
        d &= ~(np.isnan(a) & np.isnan(b))
    return d


def _valid_depth(a):
    return np.isfinite(a) & (a > 0)


def _depth_scale(a):
    return 1e-3 if a.dtype == np.uint16 else 1.0


def _choose(a, eligible, rng, severity, granularity):
    """Boolean mask over eligible pixels covering ~severity of them; whole blobs unless pixel mode."""
    if granularity == "pixel":
        return eligible & (rng.random(a.shape) < severity)
    # a blob is one 4-connected component of a single label
    labelled, blobs = [], []
    for cls in np.unique(a[eligible]):
        lab, n = ndimage.label(a == cls)
        sizes = np.bincount(lab.ravel())[1:]
        blobs += [(len(labelled), j + 1, int(s)) for j, s in enumerate(sizes)]
        labelled.append(lab)
    target = severity * int(eligible.sum())
    picked = [[] for _ in labelled]
    total = 0
    for k in rng.permutation(len(blobs)):
        if total >= target:
            break
        li, cid, size = blobs[k]
        picked[li].append(cid)
        total += size
    mask = np.zeros(a.shape, bool)
    for lab, ids in zip(labelled, picked):
        if ids:
            mask |= np.isin(lab, ids)
    return mask


def _label_stats(a, objects, eligible, changed, n_blobs):
    n_obj = int(objects.sum())
    n_ch = int(changed.sum())
    return {"n_pixels": int(a.size), "n_object_pixels": n_obj,
            "n_eligible_pixels": int(eligible.sum()), "n_changed": n_ch,
            "frac_changed": n_ch / a.size,
            "frac_object_changed": n_ch / n_obj if n_obj else 0.0,
            "n_blobs_changed": n_blobs}


def _count_blobs(a, changed):
    # blobs are counted on the source label image so the number is comparable across faults
    return int(sum(ndimage.label(changed & (a == c))[1] for c in np.unique(a[changed])))


# ---------------------------------------------------------------- per-frame image faults

def label_confusion(a, rng, severity, p):
    objects = np.isin(a, p["object_labels"])
    cmap = {k: v for k, v in p["confusion_map"].items() if k in set(p["object_labels"])}
    eligible = np.isin(a, list(cmap))
    changed = _choose(a, eligible, rng, severity, p["granularity"])
    out = a.copy()
    for k, v in cmap.items():
        out[changed & (a == k)] = v
    changed = _differs(a, out)
    return out, _label_stats(a, objects, eligible, changed, _count_blobs(a, changed))


def label_dropout(a, rng, severity, p):
    objects = np.isin(a, p["object_labels"])
    changed = _choose(a, objects, rng, severity, p["granularity"])
    out = a.copy()
    out[changed] = p["void_label"]
    changed = _differs(a, out)
    return out, _label_stats(a, objects, objects, changed, _count_blobs(a, changed))


def depth_noise(a, rng, severity, p):
    valid = _valid_depth(a)
    d = a.astype(np.float64)
    noisy = d + rng.standard_normal(a.shape) * severity * p["rel_sigma_max"] * d
    if a.dtype == np.uint16:
        noisy = np.clip(np.rint(noisy), 0, np.iinfo(np.uint16).max)
    # a non-positive draw would turn a valid pixel invalid; keep the original instead
    out = np.where(valid & (noisy > 0), noisy, d).astype(a.dtype)
    changed = _differs(a, out)
    diff = np.abs(out.astype(np.float64) - d)[changed]
    err, rel = diff * _depth_scale(a), diff / d[changed]
    n_valid = int(valid.sum())
    return out, {"n_pixels": int(a.size), "n_valid": n_valid, "n_changed": int(changed.sum()),
                 "frac_valid_changed": int(changed.sum()) / n_valid if n_valid else 0.0,
                 "mean_abs_err_m": float(err.mean()) if err.size else 0.0,
                 "rms_err_m": float(np.sqrt((err ** 2).mean())) if err.size else 0.0,
                 "mean_rel_err": float(rel.mean()) if rel.size else 0.0}


def depth_holes(a, rng, severity, p):
    h, w = a.shape
    dropped = bool(rng.random() < severity * p["max_frame_drop_prob"])
    mask = np.ones(a.shape, bool) if dropped else np.zeros(a.shape, bool)
    n_rects = 0
    target = severity * p["max_hole_area_frac"] * a.size
    while not dropped and mask.sum() < target and n_rects < 1000:
        rh = max(1, int(rng.uniform(p["min_rect_frac"], p["max_rect_frac"]) * h))
        rw = max(1, int(rng.uniform(p["min_rect_frac"], p["max_rect_frac"]) * w))
        y0, x0 = int(rng.integers(0, h - rh + 1)), int(rng.integers(0, w - rw + 1))
        mask[y0:y0 + rh, x0:x0 + rw] = True
        n_rects += 1
    valid = _valid_depth(a)
    out = a.copy()
    out[mask] = np.nan if a.dtype.kind == "f" else 0
    gone = valid & ~_valid_depth(out)
    n_valid = int(valid.sum())
    return out, {"n_pixels": int(a.size), "n_valid_before": n_valid,
                 "n_newly_invalid": int(gone.sum()),
                 "frac_pixels_newly_invalid": int(gone.sum()) / a.size,
                 "frac_valid_removed": int(gone.sum()) / n_valid if n_valid else 0.0,
                 "frame_dropped": dropped, "n_rects": n_rects}


IMAGE_FAULTS = {"label_confusion": ("labels", label_confusion),
                "label_dropout": ("labels", label_dropout),
                "depth_noise": ("depth", depth_noise),
                "depth_holes": ("depth", depth_holes)}


# ---------------------------------------------------------------- whole-dataset faults

def _rot_z(psi):
    c, s = np.cos(psi), np.sin(psi)
    m = np.zeros((len(psi), 3, 3))
    m[:, 0, 0], m[:, 0, 1], m[:, 1, 0], m[:, 1, 1], m[:, 2, 2] = c, -s, s, c, 1.0
    return m


def pose_drift(src, dst, severity, seed, p, copy):
    fields, rows = _read_poses(src)
    pos = np.array([[float(r[k]) for k in ("tx", "ty", "tz")] for r in rows])
    quat = np.array([[float(r[k]) for k in ("qx", "qy", "qz", "qw")] for r in rows])
    n = len(rows)
    rng = np.random.default_rng(seed)
    if p["mode"] == "random_walk":
        d_psi = rng.normal(0.0, np.radians(p["yaw_sigma_deg"]) * severity, n)
        d_pos = rng.normal(0.0, p["trans_sigma_m"] * severity, (n, 3))
    else:
        direction = rng.normal(size=3)
        direction[2] = 0.0
        direction /= np.linalg.norm(direction)
        d_psi = np.full(n, np.radians(p["bias_yaw_deg"]) * severity * rng.choice([-1.0, 1.0]))
        d_pos = np.tile(direction * p["bias_trans_m"] * severity, (n, 1))
    d_psi[0], d_pos[0] = 0.0, 0.0
    psi = np.cumsum(d_psi)
    # odometry-style integration: heading error rotates every later increment, errors never revert
    step = np.diff(pos, axis=0, prepend=pos[:1])
    rot = _rot_z(psi) - np.eye(3)
    offset = np.cumsum(np.einsum("nij,nj->ni", rot, step) + d_pos, axis=0)
    new_pos = pos + offset
    new_quat = (Rotation.from_euler("z", psi) * Rotation.from_quat(quat)).as_quat()
    new_quat *= np.where((new_quat * quat).sum(axis=1) < 0, -1.0, 1.0)[:, None]

    touched = (psi != 0) | (offset != 0).any(axis=1)
    out_rows = []
    for i, r in enumerate(rows):
        r = dict(r)
        if touched[i]:
            for k, v in zip(("tx", "ty", "tz"), new_pos[i]):
                r[k] = repr(float(v))
            for k, v in zip(("qx", "qy", "qz", "qw"), new_quat[i]):
                r[k] = repr(float(v))
        out_rows.append(r)
    if touched.any():
        _write_poses(dst, fields, out_rows)
    else:
        _place(src / "poses.csv", dst / "poses.csv", copy)
    _link_all(src, dst, copy, skip={"poses.csv"})

    # stats are re-measured from the values as written
    wp = np.array([[float(r[k]) for k in ("tx", "ty", "tz")] for r in out_rows])
    wq = np.array([[float(r[k]) for k in ("qx", "qy", "qz", "qw")] for r in out_rows])
    rel = Rotation.from_quat(wq) * Rotation.from_quat(quat).inv()
    yaw = np.where(touched, np.degrees(rel.as_rotvec()[:, 2]), 0.0)
    rot_err = np.where(touched, np.degrees(rel.magnitude()), 0.0)
    err = wp - pos
    frames = [{"frame": i, "src_frame": i, "timestamp_ns": int(rows[i]["timestamp_ns"]),
               "trans_error_m": float(np.linalg.norm(err[i])), "dx_m": float(err[i, 0]),
               "dy_m": float(err[i, 1]), "dz_m": float(err[i, 2]),
               "yaw_error_deg": float(yaw[i]), "rot_error_deg": float(rot_err[i])}
              for i in range(n)]
    path_len = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
    extra = {"path_length_m": path_len,
             "final_trans_error_m": frames[-1]["trans_error_m"],
             "final_drift_pct_of_path": 100 * frames[-1]["trans_error_m"] / path_len if path_len else 0.0,
             "final_yaw_error_deg": frames[-1]["yaw_error_deg"]}
    return frames, extra, n, int(touched.sum())


def frame_drop(src, dst, severity, seed, p, copy):
    fields, rows = _read_poses(src)
    n = len(rows)
    rng = np.random.default_rng(seed)
    n_drop = min(int(round(severity * p["max_drop_frac"] * n)), n - 1)
    k = min(math.ceil(n_drop / p["chunk_len"]), n - n_drop + 1) if n_drop else 0
    dropped = np.zeros(n, bool)
    if k:
        lens = rng.permutation(n_drop // k + (np.arange(k) < n_drop % k))
        # distinct cut points among kept frames keep chunks from merging
        cuts = np.sort(rng.choice(n - n_drop + 1, size=k, replace=False))
        start = cuts + np.concatenate([[0], np.cumsum(lens)[:-1]])
        for s, ln in zip(start, lens):
            dropped[s:s + ln] = True
    kept = np.flatnonzero(~dropped)
    chunks = [[int(s), int(e)] for s, e in zip(np.flatnonzero(dropped & ~np.r_[False, dropped[:-1]]),
                                               np.flatnonzero(dropped & ~np.r_[dropped[1:], False]) + 1)]
    _write_or_link_poses(src, dst, fields, [rows[i] for i in kept], copy, same=not n_drop)
    _link_all(src, dst, copy, skip={"poses.csv"}, index_map={int(s): j for j, s in enumerate(kept)})
    gap = np.diff(kept, prepend=-1) - 1
    frames = [{"frame": j, "src_frame": int(s), "timestamp_ns": int(rows[s]["timestamp_ns"]),
               "dropped_before": int(gap[j])} for j, s in enumerate(kept)]
    extra = {"dropped_chunks_src": chunks, "dropped_src_indices": np.flatnonzero(dropped).tolist(),
             "kept_src_indices": kept.tolist(), "n_dropped": int(dropped.sum())}
    return frames, extra, int(len(kept)), int(dropped.sum())


def _write_or_link_poses(src, dst, fields, rows, copy, same):
    if same:
        _place(src / "poses.csv", dst / "poses.csv", copy)
    else:
        _write_poses(dst, fields, rows)


def _link_all(src, dst, copy, skip=(), index_map=None, only=None):
    """Link top-level files and channel frames, renumbering frames via index_map if given."""
    for f in ("camera_info.yaml", "poses.csv"):
        if f not in skip and (src / f).exists():
            _place(src / f, dst / f, copy)
    for ch in CHANNELS:
        if only is not None and ch not in only or not (src / ch).is_dir():
            continue
        (dst / ch).mkdir(exist_ok=True)
        for i, j in (index_map.items() if index_map else _all_frames(src, ch)):
            sp = _frame_path(src, ch, i)
            if sp.exists():
                _place(sp, _frame_path(dst, ch, j), copy)


def _all_frames(src, ch):
    idx = sorted(int(f.stem.rsplit("_", 1)[1]) for f in (src / ch).iterdir())
    return ((i, i) for i in idx)


# ---------------------------------------------------------------- driver

# name -> (first corrupted layer, other layers hit directly, corrupted input, note)
FAULT_META = {
    "label_confusion": ("objects", [], "labels", "wrong semantic class on object blobs"),
    "label_dropout": ("objects", [], "labels", "object blobs relabelled void"),
    "depth_noise": ("mesh", ["places"], "depth", "depth-proportional gaussian; TSDF feeds mesh and GVD places"),
    "depth_holes": ("mesh", ["places"], "depth", "rectangular invalid regions and whole-frame depth loss"),
    "pose_drift": ("mesh", [], "poses", "cumulative drift; no separate backend layer in the scene graph, mesh is first built from poses"),
    "frame_drop": ("all", [], "all", "contiguous chunks of frames removed, files renumbered, timestamps kept"),
}
DEFAULTS = {
    "label_confusion": {**LABEL_COMMON, "confusion_map": CONFUSION_MAP},
    "label_dropout": dict(LABEL_COMMON),
    "depth_noise": {"rel_sigma_max": 0.05},
    "depth_holes": {"max_hole_area_frac": 0.5, "max_frame_drop_prob": 0.2,
                    "min_rect_frac": 0.05, "max_rect_frac": 0.3},
    "pose_drift": {"mode": "random_walk", "trans_sigma_m": 0.02, "yaw_sigma_deg": 0.3,
                   "bias_trans_m": 0.005, "bias_yaw_deg": 0.1},
    "frame_drop": {"max_drop_frac": 0.5, "chunk_len": 10},
}


def _resolve_params(fault, params):
    out = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
           for k, v in DEFAULTS[fault].items()}
    unknown = set(params or {}) - set(out)
    if unknown:
        raise ValueError(f"unknown params for {fault}: {sorted(unknown)}; valid: {sorted(out)}")
    out.update(params or {})
    if "confusion_map" in out:
        out["confusion_map"] = {int(k): int(v) for k, v in out["confusion_map"].items()}
    if "object_labels" in out:
        out["object_labels"] = [int(x) for x in out["object_labels"]]
    if out.get("granularity", "blob") not in ("blob", "pixel"):
        raise ValueError("granularity must be 'blob' or 'pixel'")
    if out.get("mode", "random_walk") not in ("random_walk", "bias"):
        raise ValueError("mode must be 'random_walk' or 'bias'")
    return out


def _summarize(frames):
    skip = {"frame", "src_frame", "timestamp_ns"}
    keys = [k for k, v in frames[0].items() if k not in skip and isinstance(v, (int, float))] if frames else []
    return {k: {"mean": float(np.mean([f[k] for f in frames])), "max": float(np.max([f[k] for f in frames])),
                "sum": float(np.sum([f[k] for f in frames]))} for k in keys}


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def inject(src, dst, fault, severity, seed=0, params=None, copy=False):
    """Write a copy of dataset src to dst with one fault; return the manifest dict."""
    src, dst = Path(src).resolve(), Path(dst).resolve()
    if fault not in FAULT_META:
        raise ValueError(f"unknown fault {fault!r}; valid: {sorted(FAULT_META)}")
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be in [0, 1]")
    if dst == src or src in dst.parents or dst in src.parents:
        raise ValueError("dst must not overlap src")
    if dst.exists() and any(dst.iterdir()):
        raise ValueError(f"dst not empty: {dst}")
    for f in ("poses.csv", "camera_info.yaml"):
        if not (src / f).exists():
            raise FileNotFoundError(src / f)
    p = _resolve_params(fault, params)
    dst.mkdir(parents=True, exist_ok=True)

    extra, n_mod = {}, 0
    if fault in IMAGE_FAULTS:
        channel, fn = IMAGE_FAULTS[fault]
        if not (src / channel).is_dir():
            raise FileNotFoundError(src / channel)
        _link_all(src, dst, copy, only=set(CHANNELS) - {channel})
        (dst / channel).mkdir(exist_ok=True)
        n_in = len(_read_poses(src)[1])
        frames = []
        for i in range(n_in):
            sp = _frame_path(src, channel, i)
            if not sp.exists():
                continue
            a = _read_img(sp)
            b, st = fn(a, np.random.default_rng([seed, i]), severity, p)
            if _differs(a, b).any():
                _write_img(_frame_path(dst, channel, i), b)
                n_mod += 1
            else:
                _place(sp, _frame_path(dst, channel, i), copy)
            assert b.shape == a.shape and b.dtype == a.dtype
            frames.append({"frame": i, "src_frame": i, **st})
        n_out = n_in
    elif fault == "pose_drift":
        frames, extra, n_out, n_mod = pose_drift(src, dst, severity, seed, p, copy)
        n_in = n_out
    else:
        n_in = len(_read_poses(src)[1])
        frames, extra, n_out, n_mod = frame_drop(src, dst, severity, seed, p, copy)

    layer, also, corrupted, note = FAULT_META[fault]
    manifest = {
        "schema_version": 1, "fault": fault, "severity": severity, "seed": seed, "params": p,
        "expected_first_layer": layer, "also_hits_directly": also,
        "corrupted_input": corrupted, "note": note, "layer_order": LAYER_ORDER,
        "src_dataset": str(src), "n_frames_in": n_in, "n_frames_out": n_out,
        "n_units_modified": n_mod, "summary": _summarize(frames), "extra": extra,
        "frame_stats": frames,
    }
    text = json.dumps(manifest, indent=1, default=_json_default, allow_nan=False)
    (dst / "injection_manifest.json").write_text(text)
    # return the on-disk form (int dict keys become str) so callers see what the file says
    return json.loads(text)


def _parse_param(s):
    k, _, v = s.partition("=")
    try:
        return k, json.loads(v)
    except json.JSONDecodeError:
        return k, v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--fault", required=True, choices=sorted(FAULT_META))
    ap.add_argument("--severity", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                    help="override a fault param; VALUE is JSON, e.g. --param granularity=pixel")
    ap.add_argument("--label-space", help="Hydra label space yaml; sets object_labels and void_label")
    ap.add_argument("--copy", action="store_true", help="copy unchanged files instead of hard-linking")
    a = ap.parse_args(argv)
    params = dict(_parse_param(s) for s in a.param)
    if a.label_space and "object_labels" in DEFAULTS[a.fault]:
        with open(a.label_space) as f:
            ls = yaml.safe_load(f)
        params.setdefault("object_labels", ls["object_labels"])
        params.setdefault("void_label", (ls.get("invalid_labels") or [0])[0])  # uhumans2 spaces list no invalid labels; 0 = unknown
    m = inject(a.src, a.dst, a.fault, a.severity, a.seed, params, a.copy)
    print(f"{m['fault']} severity={m['severity']} seed={m['seed']} first_layer={m['expected_first_layer']} "
          f"frames {m['n_frames_in']}->{m['n_frames_out']} modified={m['n_units_modified']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
