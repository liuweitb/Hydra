#!/usr/bin/env python3
"""Which features tell one injected fault from another over ALL available severities (s03 both seeds, s20, s50, s80)?

Range separation: for fault pair (a, b) and feature f, over the runs of each fault the deviation d = |value - center|
spans [lo, hi]; the pair is separated when one fault's lo exceeds the other's hi.  gap = lo_high / hi_low (>= 1.5 is a
usable margin, inf when the lower fault has d = 0).  Development data only, no thresholds involved: this is about
signatures, not detection.  Also tests two ratio signatures computed from existing features.
usage: lookalike_probe.py [--dir DIR]
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import analyze_probe as ap  # noqa: E402
import evidence_groups as eg  # noqa: E402

PAIRS = [("pose_drift", "depth_holes"), ("pose_drift", "depth_noise"), ("depth_holes", "depth_noise"),
         ("label_confusion", "label_dropout"), ("frame_drop", "pose_drift")]


def fault_runs(M, f):
    return list(M.index[(M.fault == f) & ((M.role == "benign") | (M.seed == 0))])


def gap_table(D, M, info, fa, fb, kind, include_count=False):
    ra, rb = fault_runs(M, fa), fault_runs(M, fb)
    rows = []
    for n in info.index[info.group.isin(eg.GROUPS + ("INPUT_POSE",)) & (info.kind == kind)]:
        if info.at[n, "count_based"] and not include_count:
            continue
        a, b = D.loc[n, ra].astype(float).dropna(), D.loc[n, rb].astype(float).dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        if a.min() >= b.max():
            hi, lo, who = a.min(), b.max(), fa
        elif b.min() >= a.max():
            hi, lo, who = b.min(), a.max(), fb
        else:
            continue
        gap = hi / lo if lo > 0 else np.inf
        floor = info.at[n, "floor"]
        if hi < floor:  # both sides inside the practical tolerance: not a usable signature
            continue
        rows.append(dict(feature=n, group=info.at[n, "group"], larger_for=who, gap=gap, lo_of_larger=hi, hi_of_smaller=lo,
                         floor=floor, n_a=len(a), n_b=len(b)))
    t = pd.DataFrame(rows)
    return t.sort_values("gap", ascending=False) if len(t) else t


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--dir", default=str(pathlib.Path(__file__).resolve().parent))
    d = pathlib.Path(ap_.parse_args().dir)
    V, M, info = ap.load(d)
    D = ap.deviation(V, info)
    # ratio signatures (scene-free ratios of absolute feature values)
    V2 = V.copy()
    V2.loc["sig_floor_time_over_sd"] = V.loc["floor_z_time_range"] / V.loc["floor_z_robust_sd"]
    V2.loc["sig_rough_over_floor_sd"] = V.loc["mesh_roughness_median"] / V.loc["floor_z_robust_sd"]
    print("## runs per fault used for range separation\n")
    print(", ".join(f"{ap.SHORT[f]}: {len(fault_runs(M, f))}" for f in ap.FAULTS))
    for fa, fb in PAIRS:
        for kind in ("reference_free", "reference_based"):
            t = gap_table(D, M, info, fa, fb, kind)
            print(f"\n### {fa} vs {fb}, {kind}: {len(t)} separating non-count features (gap >= 1)\n")
            if len(t):
                t = t.head(8)
                print(ap.md(t.set_index("feature"), "{:.3g}", index=True))
    print("\n## ratio signatures (raw values per run, s03 = both seeds)\n")
    rows = {}
    for f in ("pose_drift", "depth_holes", "depth_noise", "label_confusion", "label_dropout", "frame_drop"):
        for r in sorted(fault_runs(M, f), key=lambda r: (M.at[r, "severity"], M.at[r, "seed"])):
            rows[f"{ap.SHORT[f]} s{int(round(M.at[r, 'severity'] * 100)):02d} r{M.at[r, 'seed']}"] = V2.loc[
                ["sig_floor_time_over_sd", "sig_rough_over_floor_sd", "mesh_vertex_count_ratio", "mesh_void_frac"], r]
    rows["clean"] = V2.loc[["sig_floor_time_over_sd", "sig_rough_over_floor_sd", "mesh_vertex_count_ratio", "mesh_void_frac"], "uh2_clean"]
    print(ap.md(pd.DataFrame(rows).T, "{:.4g}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
