#!/usr/bin/env python3
"""Compute the feature battery on all development snapshots; write feature_values.csv and run_meta.csv.

Development data only: reference, clean repeat, benign runs (s03, seeds 0 and 1) and seed-0 fault runs
(s20/s50/s80). Any other _r1/_r2 name is rejected here and in the extractor; the manifest copy must agree.
Single process; per-run per-feature results are cached by feature source hash.
usage: run_probe.py [--only f1,f2] [--no-cache] [--snapshots DIR] [--out DIR] [--runs name1,name2]
"""

import argparse
import hashlib
import inspect
import json
import pathlib
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import features_proto as fp  # noqa: E402

FAULTS = ["label_confusion", "label_dropout", "depth_noise", "depth_holes", "pose_drift", "frame_drop"]
F = "|".join(FAULTS)
DEV_NAME = re.compile(r"^uh2_(clean|clean_b)$|^uh2_(%s)_s03_r[01]$|^uh2_(%s)_s(20|50|80)(_r0)?$" % (F, F))


def run_info(name, snap):
    """(role, fault, severity, seed) from the manifest copy; names only pick the candidate set."""
    m = snap.manifest
    if not m:
        return ("reference" if name == "uh2_clean" else "clean_repeat"), "none", 0.0, 0
    sev, seed = float(m["severity"]), int(m["seed"])
    if seed != 0 and sev > 0.05:
        raise RuntimeError(f"held-out run refused: {name} seed={seed} severity={sev}")
    return ("benign" if sev <= 0.05 else "fault"), m["fault"], sev, seed


def src_hash(fn):
    return hashlib.sha1(inspect.getsource(fn).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", default="D:/hydra_work/results/signal_probe/snapshots")
    ap.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent))
    ap.add_argument("--cache", default="D:/hydra_work/results/signal_probe/feature_cache")
    ap.add_argument("--only", default=None)
    ap.add_argument("--runs", default=None, help="comma list of run names to (re)compute; default all dev snapshots")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    out, cache = pathlib.Path(args.out), pathlib.Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only else None
    want = set(args.runs.split(",")) if args.runs else None

    paths = sorted(p for p in pathlib.Path(args.snapshots).glob("*.npz") if DEV_NAME.match(p.stem) and (want is None or p.stem in want))
    ref = fp.Snap(pathlib.Path(args.snapshots) / "uh2_clean.npz")
    rows, meta = {}, {}
    for p in paths:
        name = p.stem
        cf = cache / f"{name}.json"
        cached = {} if (args.no_cache or not cf.exists()) else json.loads(cf.read_text())
        todo = [n for n, s in fp.REGISTRY.items()
                if (only is None or n in only) and cached.get(n, {}).get("h") != src_hash(s.fn)]
        snap = ref if name == "uh2_clean" else None
        if todo:
            snap = snap or fp.Snap(p)
            t0 = time.time()
            vals = fp.compute_all(snap, ref, names=set(todo), log=lambda m: print("  !", m, flush=True))
            for n, v in vals.items():
                cached[n] = {"h": src_hash(fp.REGISTRY[n].fn), "v": v}
            cf.write_text(json.dumps(cached))
            print(f"{name}: {len(todo)} features in {time.time() - t0:.1f}s", flush=True)
        snap = snap or fp.Snap(p)
        role, fault, sev, seed = run_info(name, snap)
        meta[name] = dict(run=name, role=role, fault=fault, severity=sev, seed=seed,
                          n_vertices=len(snap.mesh_xyz), n_objects=len(snap.obj_pos), n_places=len(snap.place_pos),
                          n_rooms=len(snap.room_pos), n_keyframes=len(snap.traj))
        rows[name] = {n: cached[n]["v"] for n in fp.REGISTRY if n in cached}
        del snap

    # the reference against itself must give the ideal value; flag any feature that does not
    for n, s in fp.REGISTRY.items():
        if s.kind == "reference_based" and "uh2_clean" in rows:
            v = rows["uh2_clean"].get(n)
            if v is not None and np.isfinite(v) and abs(v - s.ideal) > 1e-6:
                print("  ! self-check", n, v, "ideal", s.ideal)
            rows["uh2_clean"][n] = s.ideal

    df = pd.DataFrame(rows).reindex(list(fp.REGISTRY))
    df.index.name = "feature"
    mt = pd.DataFrame(meta).T
    mt["_o"] = [0 if r == "reference" else 1 if r == "clean_repeat" else 2 if r == "benign" else 3 for r in mt.role]
    mt["_f"] = [FAULTS.index(f) if f in FAULTS else -1 for f in mt.fault]
    mt = mt.sort_values(["_o", "severity", "_f", "seed"]).drop(columns=["_o", "_f"])
    df = df[list(mt.index)]
    df.to_csv(out / "feature_values.csv", float_format="%.8g")
    mt.to_csv(out / "run_meta.csv", index=False)
    print("wrote feature_values.csv", df.shape, "run_meta.csv", mt.shape)
    print("runs:", ", ".join(mt.index))


if __name__ == "__main__":
    main()
