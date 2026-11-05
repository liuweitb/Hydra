#!/usr/bin/env python3
"""Benign-calibrated thresholds, severity scaling, evidence-group rule evaluation from feature_values.csv.

Development data only (reference, clean repeat, benign s03 seeds 0/1, seed-0 s20/s50/s80).
Writes thresholds.csv, feature_table.csv, feature_judgement.csv, rule_eval.csv; prints the tables to stdout.

Deviation d = |value - center|; center = ideal value (reference_based: the run against itself) or the value on
uh2_clean (reference_free).  Threshold h = max(MARGIN * max over benign d, floor), floor = tol_abs + tol_rel*|center|
(smallest change worth calling a move).  Strength x = d / h; a feature "moves" when x >= 1.
Policy A = all 12 benign runs.  Policy N = the mesh-neutral benign runs: those whose reference-based MESH
deviations all stay below their floors (a property of the benign run alone; s03 depth_holes and pose_drift already
move the mesh).  Benign false alarms are measured leave-one-run-out (LOO) and leave-one-family-out.
usage: analyze_probe.py [--dir DIR] [--margin M]
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import evidence_groups as eg  # noqa: E402
import features_proto as fp  # noqa: E402

FAULTS = ["label_confusion", "label_dropout", "depth_noise", "depth_holes", "pose_drift", "frame_drop"]
SHORT = {"label_confusion": "lc", "label_dropout": "ld", "depth_noise": "dn", "depth_holes": "dh", "pose_drift": "pd", "frame_drop": "fd"}
EXPECTED = {"label_confusion": "OBJECTS", "label_dropout": "OBJECTS", "depth_noise": "MESH", "depth_holes": "MESH",
            "pose_drift": "MESH", "frame_drop": "all"}  # expected_first_layer in the manifests
SEVS = (0.2, 0.5, 0.8)
MARGIN = 1.5
ALLOWED_READS = {"MESH": {"mesh_geom", "mesh_sem"}, "OBJECTS": {"mesh_geom", "mesh_sem"}, "PLACES": {"mesh_geom"}, "ROOMS": {"places"}}


# physically motivated subsets (chosen after the seed-0 s50 values were seen, so in-sample on development data)
REC_REF = {
    "MESH": ["mesh_extra_frac_gt10cm", "mesh_missing_frac_gt10cm", "mesh_extra_mean", "mesh_missing_mean", "mesh_centroid_shift"],
    "OBJECTS": ["obj_label_agree_matched", "obj_recall_same_label", "obj_class_hist_js", "mesh_label_hist_js", "obj_precision_pos",
                "obj_extra_near_ref_frac"],
    "PLACES": ["place_coverage_050", "place_spurious_050", "place_chamfer_mean", "place_freespace_iou", "place_freevol_ratio"],
    "ROOMS": ["room_match_frac", "room_assignment_agreement", "room_partition_ari", "room_centroid_err_median"],
}
REC_FREE = {
    "MESH": ["mesh_roughness_median", "floor_z_robust_sd", "floor_z_time_range", "mesh_revisit_offset_median_k96", "wall_sheet_sd_median"],
    "OBJECTS": ["obj_mesh_label_purity_low_frac", "mesh_objclass_knn_purity", "obj_overlap_diffclass_frac", "mesh_void_frac"],
    "PLACES": ["place_components", "place_edge_wall_cross_frac", "place_mesh_intrusion_frac", "place_freevol_m3"],
    "ROOMS": ["room_boundary_mixing", "room_footprint_min", "room_footprint_max", "room_elongation_max"],
}


MIN_REF = {
    "MESH": ["mesh_extra_frac_gt10cm", "mesh_missing_frac_gt10cm"],
    "OBJECTS": ["obj_label_agree_matched", "obj_recall_same_label", "obj_class_hist_js"],
    "PLACES": ["place_coverage_050", "place_freevol_ratio"],
    "ROOMS": ["room_assignment_agreement", "room_partition_ari"],
}
MIN_FREE = {
    "MESH": ["mesh_roughness_median", "floor_z_robust_sd", "floor_z_time_range"],
    "OBJECTS": ["obj_mesh_label_purity_low_frac", "mesh_objclass_knn_purity", "mesh_void_frac"],
    "PLACES": ["place_freevol_m3", "place_edge_wall_cross_frac"],
    "ROOMS": ["room_boundary_mixing", "room_footprint_min"],
}


# --------------------------------------------------------------------------- data
def load(d):
    V = pd.read_csv(d / "feature_values.csv", index_col=0)
    M = pd.read_csv(d / "run_meta.csv", index_col=0)
    M["severity"] = M["severity"].round(2)
    R = fp.REGISTRY
    info = pd.DataFrame({
        "kind": {n: s.kind for n, s in R.items()}, "layer": {n: s.layer for n, s in R.items()},
        "unit": {n: s.unit for n, s in R.items()}, "count_based": {n: s.count_based for n, s in R.items()},
        "ideal": {n: s.ideal for n, s in R.items()}, "tol_abs": {n: s.tol_abs for n, s in R.items()},
        "tol_rel": {n: s.tol_rel for n, s in R.items()}, "definition": {n: s.definition for n, s in R.items()},
    }).reindex(V.index)
    eg.check(R)
    info["group"] = [eg.GROUP[n][0] for n in info.index]
    info["tag"] = [eg.GROUP[n][1] for n in info.index]
    info["reads"] = [eg.GROUP[n][2] for n in info.index]
    info["center"] = [info.at[n, "ideal"] if info.at[n, "kind"] == "reference_based" else V.at[n, "uh2_clean"] for n in info.index]
    info["floor"] = info["tol_abs"] + info["tol_rel"] * info["center"].abs()
    return V, M, info


def deviation(V, info):
    return V.sub(info["center"], axis=0).abs()


def run_of(M, fault, sev, seed=0):
    r = M.index[(M.fault == fault) & (np.isclose(M.severity, sev)) & (M.seed == seed)]
    return r[0] if len(r) else None


def benign_by_family(M):
    return {f: list(M.index[(M.role == "benign") & (M.fault == f)]) for f in FAULTS}


# --------------------------------------------------------------------------- thresholds
def thresholds(D, benign, info, margin):
    mx = D[benign].max(axis=1, skipna=True)
    return pd.concat([margin * mx, info["floor"]], axis=1).max(axis=1), mx


def loo_false_alarm(D, M, benign, info, margin, by_family=False):
    """Per feature and benign run: 1 if the run exceeds the threshold built from the other benign runs (own run, or
    own whole family when by_family), else 0; NaN where the feature is undefined."""
    fam_of = M.fault
    flags = pd.DataFrame(index=D.index, columns=benign, dtype=float)
    for r in benign:
        rest = [b for b in benign if b != r and (not by_family or fam_of[b] != fam_of[r])]
        h, _ = thresholds(D, rest, info, margin)
        flags[r] = (D[r] > h).astype(float).where(D[r].notna())
    return flags


def saturation_index(D, M, info):
    """Per fault family and group: median over responsive reference-based non-count features (d50 > 2*floor) of
    mean over seeds of d at s03 divided by d at s50. About 0.06 = proportional to severity, >= 0.5 = s03 acts like s50."""
    fam = benign_by_family(M)
    si, nn = {}, {}
    for f in FAULTS:
        r50 = run_of(M, f, 0.5)
        for g in eg.GROUPS:
            fs = list(info.index[(info.group == g) & (info.kind == "reference_based") & (~info.count_based)])
            ratios = [D.loc[n, fam[f]].mean() / D.at[n, r50] for n in fs
                      if r50 and np.isfinite(D.at[n, r50]) and D.at[n, r50] > 2 * max(info.at[n, "floor"], 1e-9)]
            si[(SHORT[f], g)], nn[(SHORT[f], g)] = (float(np.median(ratios)) if ratios else np.nan), len(ratios)
    mk = lambda dct: pd.DataFrame({g: {SHORT[f]: dct[(SHORT[f], g)] for f in FAULTS} for g in eg.GROUPS})
    return mk(si), mk(nn)


def mesh_neutral(D, M, info, benign_all):
    """Benign runs whose reference-based MESH (non-count) deviations are all within the floor."""
    fs = list(info.index[(info.group == "MESH") & (info.kind == "reference_based") & (~info.count_based)])
    ratio = D.loc[fs, benign_all].div(info.loc[fs, "floor"], axis=0).max()
    return [r for r in benign_all if ratio[r] <= 1.0], ratio


def policy_thresholds(D, M, info, benign_A, benign_N, margin):
    return {"A": thresholds(D, benign_A, info, margin)[0], "N": thresholds(D, benign_N, info, margin)[0]}


# --------------------------------------------------------------------------- group rule
def detector_sets(info, kinds):
    """Pre-registered per-group sets: non-count features of the group whose reads stay inside the allowed layers."""
    sets = {}
    for g in eg.GROUPS:
        ok = info.index[(info.group == g) & (~info.count_based) & info.kind.isin(kinds)]
        keep = []
        for n in ok:
            reads = {t for t in info.at[n, "reads"].split(";") if t}
            if reads <= ALLOWED_READS[g]:
                keep.append(n)
        sets[g] = keep
    return sets


def predict(counts, k):
    fired = {g: counts[g] >= k for g in eg.GROUPS}
    if fired["MESH"]:
        return "MESH", fired
    if fired["OBJECTS"] and fired["PLACES"]:
        return "OBJECTS|PLACES", fired
    for g in ("OBJECTS", "PLACES", "ROOMS"):
        if fired[g]:
            return g, fired
    return "none", fired


def evaluate_rule(D, M, info, benign_A, benign_N, margin, mode_sets, ks=(1, 2, 3)):
    """Rows: run x policy x mode x k -> number of moving features per group and the DAG prediction.
    Fault runs: thresholds from all benign runs of the policy.  Benign runs: leave-one-run-out ('LOO_run') and
    leave-one-family-out ('LOO_family') thresholds from the policy's benign runs."""
    rows = []
    for r in M.index[M.role.isin(["fault", "benign"])]:
        f, role = M.at[r, "fault"], M.at[r, "role"]
        thr = {}
        for pol, ben in (("A", benign_A), ("N", benign_N)):
            if role == "fault":
                thr[pol] = thresholds(D, ben, info, margin)[0]
            else:
                thr[pol + "_LOO_run"] = thresholds(D, [b for b in ben if b != r], info, margin)[0]
                thr[pol + "_LOO_family"] = thresholds(D, [b for b in ben if M.at[b, "fault"] != f], info, margin)[0]
        for pol, h in thr.items():
            x = D[r] / h
            for mode, sets in mode_sets.items():
                counts = {g: int((x[sets[g]] >= 1).sum()) for g in eg.GROUPS}
                for k in ks:
                    pred, _ = predict(counts, k)
                    rows.append(dict(run=r, role=role, fault=f, severity=M.at[r, "severity"], policy=pol, mode=mode, k=k, pred=pred,
                                     **{f"n_{g}": counts[g] for g in eg.GROUPS}))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- printing
def md(df, fmt="{:.2f}", index=True):
    cols = list(df.columns)
    head = ([df.index.name or ""] if index else []) + [str(c) for c in cols]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for i, row in df.iterrows():
        cells = [fmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else ("" if isinstance(v, float) else str(v)) for v in row]
        lines.append("| " + " | ".join(([str(i)] if index else []) + cells) + " |")
    return "\n".join(lines)


def severity_scaling(D, M, info):
    """Per feature and fault: deviation at s03 (mean of two seeds), s20, s50, s80 and a class of the dose response.
    silent = never above the floor; step = s03 already >= 50 % of the largest later deviation; graded = non-decreasing
    (10 % slack); non_monotone otherwise."""
    rows = []
    for f in FAULTS:
        runs = {s: [r for r in M.index if M.at[r, "fault"] == f and np.isclose(M.at[r, "severity"], s) and (M.at[r, "seed"] == 0 or s <= 0.05)]
                for s in (0.03, 0.2, 0.5, 0.8)}
        for n in info.index:
            d = {s: (float(D.loc[n, rr].astype(float).mean()) if rr else np.nan) for s, rr in runs.items()}
            vals = [d[s] for s in (0.03, 0.2, 0.5, 0.8) if np.isfinite(d[s])]
            later = [d[s] for s in (0.2, 0.5, 0.8) if np.isfinite(d[s])]
            if not vals or not later:
                cls = "na"
            elif max(vals) <= info.at[n, "floor"]:
                cls = "silent"
            elif np.isfinite(d[0.03]) and d[0.03] >= 0.5 * max(later):
                cls = "step"
            elif np.all(np.diff(vals) >= -0.1 * max(vals)):
                cls = "graded"
            else:
                cls = "non_monotone"
            rows.append(dict(feature=n, group=info.at[n, "group"], kind=info.at[n, "kind"], count_based=info.at[n, "count_based"], fault=f,
                             d_s03=d[0.03], d_s20=d[0.2], d_s50=d[0.5], d_s80=d[0.8], floor=info.at[n, "floor"], dose_class=cls))
    return pd.DataFrame(rows)


def dose_table(X, M, feats, faults=FAULTS, sevs=(0.03, 0.2, 0.5, 0.8)):
    """Strength per feature for each fault and severity (s03 = mean of the two benign seeds)."""
    cols = {}
    for f in faults:
        for s in sevs:
            runs = [r for r in M.index if M.at[r, "fault"] == f and np.isclose(M.at[r, "severity"], s) and (M.at[r, "seed"] == 0 or s <= 0.05)]
            if runs:
                cols[f"{SHORT[f]}{int(round(s * 100)):02d}"] = X.loc[feats, runs].mean(axis=1)
    return pd.DataFrame(cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(pathlib.Path(__file__).resolve().parent))
    ap.add_argument("--margin", type=float, default=MARGIN)
    args = ap.parse_args()
    out = pathlib.Path(args.dir)
    margin = args.margin
    V, M, info = load(out)
    D = deviation(V, info)
    benign_A = list(M.index[M.role == "benign"])
    benign_N, nratio = mesh_neutral(D, M, info, benign_A)
    faults = M[M.role == "fault"]
    print(f"runs {len(M)}: benign {len(benign_A)}, fault {len(faults)}; margin {margin}")
    print("fault runs:", ", ".join(f"{SHORT[a]}{int(round(b * 100))}" for a, b in zip(faults.fault, faults.severity)))

    print("\n## mesh-neutral benign runs (largest reference-based MESH deviation / floor; <= 1 is neutral)\n")
    print(md(pd.DataFrame({"max_dev_over_floor": nratio, "neutral": [r in benign_N for r in benign_A]}, index=benign_A), "{:.2f}"))

    # ---------------- thresholds under both policies
    hA, mxA = thresholds(D, benign_A, info, margin)
    hN, mxN = thresholds(D, benign_N, info, margin)
    fam_max = pd.DataFrame({SHORT[f]: D.loc[:, [r for r in benign_A if M.at[r, "fault"] == f]].max(axis=1) for f in FAULTS})
    argfam = fam_max.idxmax(axis=1).where(fam_max.max(axis=1) > 0, "-")
    over_old = (D[benign_A].sub(info["floor"], axis=0) > 0).sum(axis=1)
    fa = {}
    for pol, ben in (("A", benign_A), ("N", benign_N)):
        for by_fam in (False, True):
            fl = loo_false_alarm(D, M, ben, info, margin, by_family=by_fam)
            fa[(pol, by_fam)] = fl.mean(axis=1)
    th = pd.concat([info[["kind", "layer", "group", "tag", "reads", "count_based", "center", "floor"]],
                    fam_max.add_prefix("benign_max_"), mxA.rename("benign_max_A"), argfam.rename("benign_argmax_family"),
                    hA.rename("thr_A"), hN.rename("thr_N"), (hA <= info["floor"] + 1e-12).rename("floor_binds_A"),
                    (hN <= info["floor"] + 1e-12).rename("floor_binds_N"), over_old.rename("n_benign_over_old_thr"),
                    fa[("A", False)].rename("loo_run_fa_A"), fa[("A", True)].rename("loo_family_fa_A"),
                    fa[("N", False)].rename("loo_run_fa_N"), fa[("N", True)].rename("loo_family_fa_N")], axis=1)
    th.rename_axis("feature").to_csv(out / "thresholds.csv", float_format="%.6g")

    print("\n## single-feature leave-one-out false-alarm rate on benign runs (mean over non-count features)\n")
    rows = {}
    for g in eg.GROUPS:
        for kind in ("reference_based", "reference_free"):
            idx = info.index[(info.group == g) & (info.kind == kind) & (~info.count_based)]
            rows[f"{g} {kind[10:]}"] = {"A run": fa[("A", False)][idx].mean(), "A family": fa[("A", True)][idx].mean(),
                                         "N run": fa[("N", False)][idx].mean(), "N family": fa[("N", True)][idx].mean()}
    print(md(pd.DataFrame(rows).T, "{:.3f}"))

    print("\n## saturation index (median d_s03 / d_s50 over responsive reference-based non-count features; ~0.06 proportional, >= 0.5 saturated)\n")
    si, nn = saturation_index(D, M, info)
    print(md(si) + "\n\nfeatures used:\n\n" + md(nn, "{:.0f}"))

    print("\n## which benign family sets the threshold (non-count features: count by group, kind, family)\n")
    t = pd.DataFrame({"group": info.group, "kind": info.kind.str[10:], "argmax": argfam})[~info.count_based]
    print(md(t.groupby(["group", "kind", "argmax"]).size().unstack(fill_value=0)))

    # ---------------- strengths
    XA, XN = D.div(hA, axis=0), D.div(hN, axis=0)
    long = []
    for r in M.index[M.role.isin(["fault", "benign"])]:
        for n in info.index:
            long.append((n, r, M.at[r, "role"], M.at[r, "fault"], M.at[r, "severity"], M.at[r, "seed"], V.at[n, r], info.at[n, "center"],
                         D.at[n, r], hA[n], XA.at[n, r], hN[n], XN.at[n, r]))
    jl = pd.DataFrame(long, columns=["feature", "run", "role", "fault", "severity", "seed", "value", "center", "dev", "thr_A", "x_A", "thr_N", "x_N"])
    jl = jl.merge(info[["kind", "layer", "group", "tag", "count_based"]].rename_axis("feature").reset_index(), on="feature")
    jl["moves_A"] = jl.x_A >= 1
    jl["moves_N"] = jl.x_N >= 1
    mono = {}
    for (n, f), g in jl[jl.role == "fault"].groupby(["feature", "fault"]):
        x = g.sort_values("severity").dev.to_numpy()
        mono[(n, f)] = bool(np.all(np.diff(x) >= -0.1 * np.nanmax(x))) if len(x) >= 3 and np.isfinite(x).all() else np.nan
    jl["dose_monotone"] = [mono.get((a, b), np.nan) for a, b in zip(jl.feature, jl.fault)]
    jl.to_csv(out / "feature_judgement.csv", index=False, float_format="%.6g")

    ft = th.copy()
    ft["definition"], ft["unit"] = info["definition"], info["unit"]
    for pol, X in (("A", XA), ("N", XN)):
        for f in FAULTS:
            for s in SEVS:
                r = run_of(M, f, s)
                if r:
                    ft[f"x{pol}_{SHORT[f]}_s{int(round(s * 100)):02d}"] = X[r]
        for s in SEVS:
            c = [f"x{pol}_{SHORT[f]}_s{int(round(s * 100)):02d}" for f in FAULTS if f"x{pol}_{SHORT[f]}_s{int(round(s * 100)):02d}" in ft.columns]
            ft[f"moves_{pol}_s{int(round(s * 100)):02d}"] = [";".join(a.split("_")[1] for a in c if np.isfinite(ft.at[i, a]) and ft.at[i, a] >= 1) for i in ft.index]
    vals = V.copy()
    vals.columns = [f"v:{c}" for c in vals.columns]
    pd.concat([ft, vals], axis=1).rename_axis("feature").reset_index().to_csv(out / "feature_table.csv", index=False, float_format="%.6g")

    # ---------------- usefulness
    fr = M.index[(M.role == "fault") & (M.severity >= 0.2)]
    best = pd.DataFrame({"best_x_A": XA[fr].max(axis=1), "best_x_N": XN[fr].max(axis=1)})
    useless = th.join(best)
    print("\n## features already above the old heuristic threshold (floor) on benign runs (non-count; n of 12 benign runs; best strength on any fault run >= s20)\n")
    t = useless[(useless.n_benign_over_old_thr >= 1) & (~useless.count_based)][["group", "kind", "n_benign_over_old_thr", "benign_argmax_family", "best_x_A", "best_x_N"]]
    print(md(t.sort_values(["group", "n_benign_over_old_thr"], ascending=[True, False]), "{:.1f}"))
    print("\n## useless: never reaches strength 1 on any seed-0 fault run with severity >= 0.2\n")
    for pol in ("A", "N"):
        u = useless[(useless[f"best_x_{pol}"] < 1) & (useless.group.isin(eg.GROUPS))]
        print(f"policy {pol}: {len(u)} of {int(useless.group.isin(eg.GROUPS).sum())} layer features")
        print(md(u.groupby(["group", "kind"]).size().unstack(fill_value=0), "{:.0f}"))
        print("  " + ", ".join(u.index))

    # ---------------- severity scaling of the strongest features
    top = []
    for g in eg.GROUPS:
        sub = useless[(useless.group == g) & (~useless.count_based)].sort_values("best_x_N", ascending=False).head(6)
        top += list(sub.index)
    top += [n for n in info.index[info.group == "INPUT_POSE"]]
    for pol, X in (("A", XA), ("N", XN)):
        print(f"\n## strength vs severity, policy {pol} (s03 = mean of the two benign seeds; seed 0 otherwise); strongest features per group\n")
        t = dose_table(X, M, top)
        t.insert(0, "grp", info.loc[top, "group"].str[:4])
        print(md(t, "{:.1f}"))

    sc = severity_scaling(D, M, info)
    sc.to_csv(out / "severity_scaling.csv", index=False, float_format="%.6g")
    t = sc[sc.group.isin(eg.GROUPS) & ~sc.count_based]
    print("\n## dose-response class of non-count layer features, by fault (silent = never above floor; step = s03 already >= half of the later maximum)\n")
    print(md(t.groupby(["fault", "dose_class"]).size().unstack(fill_value=0), "{:.0f}"))
    sub = t[(t.dose_class != "silent") & t.group.isin(["MESH", "OBJECTS"])]
    print("\nstep-class features by fault (MESH/OBJECTS):\n")
    for f in FAULTS:
        names = list(sub[(sub.fault == f) & (sub.dose_class == "step")].feature)
        print(f"  {SHORT[f]}: {', '.join(names) if names else '-'}")

    # ---------------- group rule
    ref_sets, free_sets = detector_sets(info, ("reference_based",)), detector_sets(info, ("reference_free",))
    both_sets = {g: ref_sets[g] + free_sets[g] for g in eg.GROUPS}
    for d_ in (REC_REF, REC_FREE, MIN_REF, MIN_FREE):
        for g, names in d_.items():
            assert all(n in info.index and info.at[n, "group"] == g for n in names), (g, names)
    rec_both = {g: REC_REF[g] + REC_FREE[g] for g in eg.GROUPS}
    print("\n## detector set sizes (all = non-count features with reads inside allowed layers; rec = physically motivated subset)\n")
    print(md(pd.DataFrame({"all ref": {g: len(ref_sets[g]) for g in eg.GROUPS}, "all free": {g: len(free_sets[g]) for g in eg.GROUPS},
                           "rec ref": {g: len(REC_REF[g]) for g in eg.GROUPS}, "rec free": {g: len(REC_FREE[g]) for g in eg.GROUPS}}), "{:.0f}"))
    min_both = {g: MIN_REF[g] + MIN_FREE[g] for g in eg.GROUPS}
    modes = {"ref": ref_sets, "free": free_sets, "both": both_sets, "rec_ref": REC_REF, "rec_free": REC_FREE, "rec_both": rec_both,
             "min_ref": MIN_REF, "min_free": MIN_FREE, "min_both": min_both}
    RE = evaluate_rule(D, M, info, benign_A, benign_N, margin, modes)
    RE.to_csv(out / "rule_eval.csv", index=False)
    n_scored = int(sum(EXPECTED[f] != "all" for f in faults.fault))
    rows = {}
    for mode in modes:
        for k in (1, 2, 3):
            sub = RE[(RE["mode"] == mode) & (RE.k == k)]
            fsub, fl = sub[sub.role == "fault"], sub[sub.role == "benign"]
            row = {}
            for pol in ("A", "N"):
                g = fsub[fsub.policy == pol]
                ok = g.pred == g.fault.map(EXPECTED)
                row[f"{pol} correct"] = f"{int(ok.sum())}/{n_scored}"
                row[f"{pol} fd"] = ",".join(g[g.fault == "frame_drop"].sort_values("severity").pred)
                b = fl[fl.policy == f"{pol}_LOO_run"]
                bf = fl[fl.policy == f"{pol}_LOO_family"]
                if pol == "N":
                    inn = b.run.isin(benign_N)
                    row["N alarm on dh/pd s03 (4)"] = f"{(b[~inn].pred != 'none').mean():.2f}"
                    b, bf = b[inn], bf[bf.run.isin(benign_N)]
                row[f"{pol} benign alarm run"] = f"{(b.pred != 'none').mean():.2f}"
                row[f"{pol} benign alarm family"] = f"{(bf.pred != 'none').mean():.2f}"
            rows[f"{mode} k={k}"] = row
    print(f"\n## DAG rule summary: correct top-1 on {n_scored} scored seed-0 runs (frame_drop expected 'all' is listed, not scored); benign alarm = share of benign runs where any group fires (A: 12 runs; N: the 8 mesh-neutral runs, the 4 dh/pd s03 runs listed apart)\n")
    print(md(pd.DataFrame(rows).T))
    for mode, k in (("both", 1), ("both", 2), ("rec_both", 1), ("rec_both", 2), ("rec_free", 2), ("rec_ref", 2), ("min_both", 1), ("min_free", 1), ("min_ref", 1)):
        sub = RE[(RE["mode"] == mode) & (RE.k == k) & (RE.role == "fault")]
        t = sub.pivot_table(index=["fault", "severity"], columns="policy", values="pred", aggfunc="first")
        t["expected"] = [EXPECTED[f] for f, _ in t.index]
        print(f"\n### predictions, mode {mode}, k={k}\n")
        print(md(t))
    print("\n## moving-feature counts per group, mode rec_both (policy A | policy N), plus input-level evidence\n")
    inp = {kind: list(info.index[(info.group == "INPUT_POSE") & (info.kind == kind)]) for kind in ("reference_based", "reference_free")}
    for pol, X in (("A", XA), ("N", XN)):
        rows = {}
        for r in faults.index:
            rows[(M.at[r, "fault"], M.at[r, "severity"])] = {**{f"n_{g}": int((X.loc[rec_both[g], r] >= 1).sum()) for g in eg.GROUPS},
                                                              "n_INPUT_ref": int((X.loc[inp["reference_based"], r] >= 1).sum()),
                                                              "n_INPUT_free": int((X.loc[inp["reference_free"], r] >= 1).sum())}
        t = pd.DataFrame(rows).T
        t.index.names = ["fault", "severity"]
        print(f"policy {pol}\n" + md(t.sort_index(), "{:.0f}"))
    print("\n## benign runs: top-1 of the rule (mode rec_both, k=2) under leave-one-family-out thresholds\n")
    sub = RE[(RE["mode"] == "rec_both") & (RE.k == 2) & (RE.role == "benign") & RE.policy.isin(["A_LOO_family", "N_LOO_family"])]
    print(md(sub.pivot_table(index="run", columns="policy", values="pred", aggfunc="first")))

    # ---------------- calibrated but not DAG-aware: blame the group with the largest strength
    print("\n## calibrated strengths without the DAG order: blame the group whose recommended set has the largest strength (mode rec_both)\n")
    rows = {}
    for r in faults.index:
        row = {}
        for pol, X in (("A", XA), ("N", XN)):
            sc = {g: float(np.nanmax(X.loc[rec_both[g], r])) for g in eg.GROUPS}
            row[f"{pol} pick"] = max(sc, key=sc.get) if max(sc.values()) >= 1 else "none"
            row[f"{pol} scores"] = "/".join(f"{sc[g]:.1f}" for g in eg.GROUPS)
        row["expected"] = EXPECTED[M.at[r, "fault"]]
        rows[(M.at[r, "fault"], M.at[r, "severity"])] = row
    t = pd.DataFrame(rows).T
    t.index.names = ["fault", "severity"]
    print("scores are MESH/OBJECTS/PLACES/ROOMS\n")
    print(md(t.sort_index()))
    okA = int(sum(v["A pick"] == v["expected"] for v in rows.values()))
    okN = int(sum(v["N pick"] == v["expected"] for v in rows.values()))
    print(f"\nlargest-strength correct: A {okA}/{n_scored}, N {okN}/{n_scored} (scored runs only)")

    # ---------------- baselines on raw (uncalibrated) discrepancy
    print("\n## baselines: layer with the largest raw reference-based discrepancy, and layer where the query failure is visible\n")
    raw = {"MESH": ["mesh_extra_frac_gt10cm", "mesh_missing_frac_gt10cm"], "OBJECTS": ["obj_recall_same_label", "obj_precision_pos"],
           "PLACES": ["place_coverage_050", "place_spurious_050"], "ROOMS": ["room_assignment_agreement", "room_match_frac"]}
    rows = {}
    for r in faults.index:
        disc = {g: float(np.nanmax([D.at[n, r] for n in ns])) for g, ns in raw.items()}
        q = {k: V.at[k, r] for k in ("query_fail_room_missing", "query_fail_target_missing", "query_fail_wrong_object")}
        vis = "ROOMS" if q["query_fail_room_missing"] >= max(q["query_fail_target_missing"], q["query_fail_wrong_object"]) else "OBJECTS"
        rows[(M.at[r, "fault"], M.at[r, "severity"])] = {**{g: v for g, v in disc.items()}, "largest_discrepancy": max(disc, key=disc.get),
                                                          "failure_visible": vis, "expected": EXPECTED[M.at[r, "fault"]]}
    t = pd.DataFrame(rows).T
    t.index.names = ["fault", "severity"]
    print(md(t.sort_index(), "{:.2f}"))
    ok1 = int(sum(v["largest_discrepancy"] == v["expected"] for v in rows.values()))
    ok2 = int(sum(v["failure_visible"] == v["expected"] for v in rows.values()))
    print(f"\nlargest-discrepancy correct {ok1}/{len(rows)}, failure-visible correct {ok2}/{len(rows)} (frame_drop expected 'all' counts as wrong)")

    print("\n## feature inventory (layer features by group and kind; n_count = count-flagged)\n")
    inv = info.groupby(["group", "kind"]).agg(n=("kind", "size"), n_count=("count_based", "sum")).unstack(fill_value=0)
    print(md(inv, "{:.0f}"))

    # ---------------- margin sensitivity
    print("\n## margin sensitivity: single-feature LOO false alarm (run | family) and share of layer features moving on seed-0 s50 faults\n")
    rows = {}
    f50 = [r for r in faults.index if np.isclose(M.at[r, "severity"], 0.5)]
    lf = info.index[info.group.isin(eg.GROUPS) & ~info.count_based]
    for m_ in (1.0, 1.25, 1.5, 2.0, 3.0):
        row = {}
        for pol, ben in (("A", benign_A), ("N", benign_N)):
            row[f"{pol} FA run"] = loo_false_alarm(D, M, ben, info, m_).mean(axis=1)[lf].mean()
            row[f"{pol} FA family"] = loo_false_alarm(D, M, ben, info, m_, by_family=True).mean(axis=1)[lf].mean()
            h_ = thresholds(D, ben, info, m_)[0]
            row[f"{pol} moving at s50"] = float(((D[f50].div(h_, axis=0) >= 1).loc[lf]).to_numpy().mean())
        rows[f"margin {m_}"] = row
    print(md(pd.DataFrame(rows).T, "{:.3f}"))

    # ---------------- count tracking
    nodes = {"MESH": "n_vertices", "OBJECTS": "n_objects", "PLACES": "n_places", "ROOMS": "n_rooms"}
    base = M.loc["uh2_clean"]
    rr = M.index[M.role.isin(["benign", "fault"])]
    rho, nsel = {}, {}
    for n in info.index[info.group.isin(eg.GROUPS)]:
        g = info.at[n, "group"]
        ratio = (M.loc[rr, nodes[g]] / base[nodes[g]]).astype(float)
        sel = ((ratio - 1).abs() > 0.03).to_numpy()
        v = V.loc[n, rr].astype(float)
        ok = sel & v.notna().to_numpy()
        nsel[n] = int(ok.sum())
        rho[n] = float(pd.Series(v[ok].to_numpy()).corr(pd.Series(ratio[ok].to_numpy()), method="spearman")) if ok.sum() >= 8 and v[ok].nunique() > 2 else np.nan
    th["rho_count"] = pd.Series(rho)
    th["rho_count_n"] = pd.Series(nsel)
    th.rename_axis("feature").to_csv(out / "thresholds.csv", float_format="%.6g")
    print("\n## features that move AND track their own layer's node count (|Spearman rho| >= 0.8 of the raw value against the count ratio, over runs whose count changed by > 3 %), non-count-flagged only\n")
    t = th.join(best)
    t = t[(t.rho_count.abs() >= 0.8) & (~t.count_based) & t.group.isin(eg.GROUPS) & ((t.best_x_A >= 1) | (t.best_x_N >= 1))][["group", "kind", "rho_count", "rho_count_n", "best_x_N"]]
    print(md(t.sort_values(["group", "rho_count"], ascending=[True, False]), "{:.2f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
