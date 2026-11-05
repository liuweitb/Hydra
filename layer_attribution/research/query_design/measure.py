#!/usr/bin/env python3
"""Measure the query-level failure on DEVELOPMENT runs only (reference, repeat, benign s03 r0/r1, seed-0 faults).

Windows python (numpy/scipy/pandas); reads SceneView files built by build_views.sh and the manifests of the run dirs.
Writes small csv tables to research/query_design/tables and per-query rows to D:/hydra_work/results/query_design.

    PYTHONPATH=D:/Hydra/layer_attribution/src python measure.py
"""

import collections
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from layer_attribution import queries as Q
from layer_attribution.queries import QueryConfig, SceneView

HERE = pathlib.Path(__file__).resolve().parent
VIEWS = pathlib.Path("D:/hydra_work/results/query_design/views")
RUNS = pathlib.Path(r"\\wsl.localhost\Ubuntu-24.04\home\hydra\runs\uh2")
BIG = pathlib.Path("D:/hydra_work/results/query_design")
TABLES = HERE / "tables"
FAULTS = ["label_confusion", "label_dropout", "depth_noise", "depth_holes", "pose_drift", "frame_drop"]
BENIGN_SEVERITY = 0.03
CFG = QueryConfig()


# ------------------------------------------------------------------------------------------------ run inventory
def manifest(name):
    p = RUNS / name / "injection_manifest.json"
    if not p.is_file():
        return None
    m = json.loads(p.read_text())
    return {k: m.get(k) for k in ("fault", "severity", "seed", "expected_first_layer", "also_hits_directly")}


def run_info(name):
    man = manifest(name)
    if man is None:
        if name not in ("uh2_clean", "uh2_clean_b"):
            raise SystemExit(f"{name}: no manifest and not a clean run")
        return {"fault": "none", "severity": 0.0, "seed": -1, "role": "reference" if name == "uh2_clean" else "repeat",
                "expected_first_layer": ""}
    sev, seed = float(man["severity"]), int(man["seed"])
    if not (seed == 0 or abs(sev - BENIGN_SEVERITY) < 1e-9):  # dev/test split guard
        raise SystemExit(f"{name}: seed {seed} severity {sev} is not development data")
    role = "benign" if abs(sev - BENIGN_SEVERITY) < 1e-9 else "main"
    return {"fault": man["fault"], "severity": sev, "seed": seed, "role": role,
            "expected_first_layer": man["expected_first_layer"] or ""}


def load_all():
    names = sorted(p.stem for p in VIEWS.glob("*.npz"))
    views, info = {}, {}
    for n in names:
        info[n] = run_info(n)
        views[n] = SceneView.load(VIEWS / f"{n}.npz")
    return views, info


# ------------------------------------------------------------------------------------------------ helpers
def ghost_count(ref, run, tol):
    """Run objects with no same-class reference object within tol (object-level precision, informational)."""
    n = 0
    for i in range(run.n_obj):
        m = ref.obj_label == run.obj_label[i]
        if not m.any() or np.linalg.norm(ref.obj_pos[m] - run.obj_pos[i], axis=1).min() > tol:
            n += 1
    return n


def adhoc_query_test(ref, run, gate_room=3.0, gate_q=1.0):
    """Re-implementation of the ad-hoc test (1-1 centroid Hungarian, single-parent objects only)."""
    def single_room(v, i):
        p = v.parents(i)
        if len(p) != 1:
            return None
        r = v.place_room[p[0]]
        return int(r) if r >= 0 else None

    room_map = {}
    if ref.n_room and run.n_room:
        d = np.linalg.norm(ref.room_pos[:, None] - run.room_pos[None], axis=2)
        r, c = linear_sum_assignment(np.where(d <= gate_room, d, 1e6))
        room_map = {int(i): int(j) for i, j in zip(r, c) if d[i, j] <= gate_room}
    st = collections.Counter()
    for o in range(ref.n_obj):
        room = single_room(ref, o)
        if room is None:
            continue
        rr = room_map.get(room)
        if rr is None:
            st["room_missing"] += 1
            continue
        c = [n for n in range(run.n_obj) if run.obj_label[n] == ref.obj_label[o] and single_room(run, n) == rr]
        if not c:
            st["target_missing"] += 1
        elif min(np.linalg.norm(run.obj_pos[n] - ref.obj_pos[o]) for n in c) <= gate_q:
            st["ok"] += 1
        else:
            st["wrong_object"] += 1
    return sum(st.values()), st


def evaluate_all(ref, views, cfg, queries=None):
    queries = queries if queries is not None else Q.build_queries(ref, cfg)
    return {n: Q.evaluate(ref, v, cfg, queries) for n, v in views.items() if n != "uh2_clean"}, queries


def run_rows(evs, info, views, ref, cfg):
    rows = []
    for n, ev in evs.items():
        s, st = ev.summary(), ev.summary(strict=True)
        structs = collections.Counter(ev.match.structures())
        qstruct = collections.Counter(r.room_structure for r in ev.results)
        row = {"run": n, **info[n], "n_queries": s["n_queries"], "n_fail": s["n_fail"], "failure_rate": round(s["failure_rate"], 4)}
        row.update({f"n_{m}": c for m, c in s["modes"].items()})
        row.update(strict_n_fail=st["n_fail"], strict_failure_rate=round(st["failure_rate"], 4),
                   n_room_split=st["modes"][Q.ROOM_SPLIT], n_room_merged=st["modes"][Q.ROOM_MERGED], n_duplicate=st["modes"][Q.DUPLICATE])
        row.update({f"rooms_{k}": structs.get(k, 0) for k in ("missing", "one_to_one", "merged", "split", "merged_split")})
        groups = Q.pair_groups(ev.queries)
        ok = {r.qid: r.ok for r in ev.results}
        row.update(pairs=len(groups), pair_fail_all_of=sum(not all(ok[i] for i in g) for g in groups.values()),
                   pair_fail_any_of=sum(not any(ok[i] for i in g) for g in groups.values()))
        row.update(n_run_rooms=views[n].n_room, n_run_objects=views[n].n_obj, n_run_places=views[n].n_place,
                   n_ghost_objects=ghost_count(ref, views[n], cfg.tol_m),
                   queries_in_changed_rooms=sum(v for k, v in qstruct.items() if k != "one_to_one"))
        nq_adhoc, st_adhoc = adhoc_query_test(ref, views[n])
        row.update(adhoc_n_queries=nq_adhoc, adhoc_failure_rate=round(1 - st_adhoc["ok"] / nq_adhoc, 4),
                   adhoc_room_missing=st_adhoc["room_missing"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df["_o"] = df["role"].map({"repeat": 0, "benign": 1, "main": 2})
    df["_f"] = df["fault"].map({f: i for i, f in enumerate(FAULTS)}).fillna(-1)
    return df.sort_values(["_o", "_f", "severity", "seed"]).drop(columns=["_o", "_f"]).reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ analyses
def reference_informativeness(ref, queries, cfg):
    names = ref.label_names
    rooms = [ref.rooms_of_object(i) for i in range(ref.n_obj)]
    groups = Q.pair_groups(queries)
    mult = collections.Counter(len(v) for v in groups.values())
    nn = []
    for q in queries:
        m = (ref.obj_label == q.label)
        m[q.ref_row] = False
        nn.append(float(np.linalg.norm(ref.obj_pos[m] - q.pos, axis=1).min()) if m.any() else float("inf"))
    nn = np.array(nn)
    same_room_close = {}
    for thr in (0.5, 0.75, 1.0, 1.5, 2.0):
        c = 0
        for q in queries:
            sel = [j for j in q.nbr_row if ref.obj_label[j] == q.label and np.linalg.norm(ref.obj_pos[j] - q.pos) < thr]
            c += bool(sel)
        same_room_close[str(thr)] = c
    per_class = {}
    for c in sorted(set(int(x) for x in ref.obj_label)):
        qs = [q for q in queries if q.label == c]
        per_class[names.get(c, str(c))] = {"objects": len(qs), "rooms": len({q.ref_room_row for q in qs}),
                                          "in_ambiguous_pair": sum(q.n_same_class_in_room > 1 for q in qs)}
    per_room = {Q.node_name(ref.room_id[r]): int(sum(q.ref_room_row == r for q in queries)) for r in range(ref.n_room)}
    two_parent = [Q.node_name(ref.obj_id[i]) for i in range(ref.n_obj) if len(ref.parents(i)) > 1]
    return {
        "n_objects": ref.n_obj, "n_places": ref.n_place, "n_rooms": ref.n_room,
        "objects_with_ancestor_room": int(sum(len(r) >= 1 for r in rooms)),
        "objects_without_room": len(Q.objects_without_room(ref)),
        "objects_with_two_parent_places": two_parent,
        "objects_with_two_distinct_rooms": int(sum(len(r) > 1 for r in rooms)),
        "n_queries": len(queries),
        "n_class_room_pairs": len(groups),
        "pair_multiplicity_histogram": {str(k): v for k, v in sorted(mult.items())},
        "unique_pairs": int(sum(1 for v in groups.values() if len(v) == 1)),
        "ambiguous_pairs": int(sum(1 for v in groups.values() if len(v) > 1)),
        "queries_in_ambiguous_pairs": int(sum(len(v) for v in groups.values() if len(v) > 1)),
        "queries_in_unique_pairs": int(sum(len(v) for v in groups.values() if len(v) == 1)),
        "pairs_with_same_class_repeat": int(sum(1 for v in groups.values() if len(v) > 1)),
        "nearest_same_class_distance_m_quantiles": {str(p): round(float(np.quantile(nn[np.isfinite(nn)], p)), 3) for p in (0, .05, .1, .25, .5, .75, 1)},
        "queries_with_same_class_rival_closer_than_m": same_room_close,
        "per_class": per_class, "queries_per_room": per_room,
    }


def position_error_tables(evs, info):
    """Nearest same-class node distance (anywhere) of every reference object in benign runs."""
    rows, allv = [], []
    for n, ev in evs.items():
        if info[n]["role"] != "benign":
            continue
        e = np.array([r.err_any_m for r in ev.results])
        allv.append((info[n]["fault"], e))
    pooled = np.concatenate([e for _, e in allv])
    pooled_f = pooled[np.isfinite(pooled)]
    edges = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf]
    hist = np.histogram(pooled_f, bins=edges)[0]
    rows.append({"group": "all_benign", "n": len(pooled), "n_inf": int((~np.isfinite(pooled)).sum()),
                 **{f"q{int(p * 100):02d}": round(float(np.quantile(pooled_f, p)), 3) for p in (.5, .9, .95, .99)},
                 "max_below_1.5": round(float(pooled_f[pooled_f <= 1.5].max()), 3),
                 "min_above_0.75": round(float(pooled_f[pooled_f > 0.75].min()), 3) if (pooled_f > 0.75).any() else None})
    for f in FAULTS:
        v = np.concatenate([e for ff, e in allv if ff == f])
        vf = v[np.isfinite(v)]
        rows.append({"group": f, "n": len(v), "n_inf": int((~np.isfinite(v)).sum()),
                     **{f"q{int(p * 100):02d}": round(float(np.quantile(vf, p)), 3) for p in (.5, .9, .95, .99)},
                     "max_below_1.5": round(float(vf[vf <= 1.5].max()), 3),
                     "min_above_0.75": round(float(vf[vf > 0.75].min()), 3) if (vf > 0.75).any() else None})
    h = pd.DataFrame({"bin_lo_m": edges[:-1], "bin_hi_m": edges[1:], "n_benign_objects": hist})
    tail = np.sort(pooled_f)[::-1][:25]
    return pd.DataFrame(rows), h, tail


def tolerance_sweep(ref, views, info, taus):
    rows = []
    for tau in taus:
        cfg = QueryConfig(tol_m=tau)
        evs, _ = evaluate_all(ref, views, cfg)
        ben = [ev.summary()["failure_rate"] for n, ev in evs.items() if info[n]["role"] == "benign"]
        for n, ev in evs.items():
            rows.append({"tol_m": tau, "run": n, "fault": info[n]["fault"], "severity": info[n]["severity"], "seed": info[n]["seed"],
                         "role": info[n]["role"], "failure_rate": round(ev.summary()["failure_rate"], 4),
                         "benign_hi": round(max(ben), 4)})
    return pd.DataFrame(rows)


def room_matching_table(ref, views, info, queries, cfg):
    """Centroid Hungarian versus footprint overlap, and agreement with the object-implied correspondence."""
    rows = []
    for n, v in views.items():
        if n == "uh2_clean":
            continue
        foot = Q.match_rooms(ref, v, cfg)
        cent = Q.match_rooms(ref, v, QueryConfig(room_method="centroid"))
        votes = collections.defaultdict(collections.Counter)  # reference room -> run rooms of exactly matched objects
        for q in queries:
            d = np.linalg.norm(v.obj_pos - q.pos, axis=1)
            cand = np.flatnonzero((v.obj_label == q.label) & (d <= 0.5))
            if len(cand) == 1:
                for r in v.rooms_of_object(cand[0]):
                    votes[q.ref_room_row][int(r)] += 1
        n_eval = agree_f = agree_c = 0
        for i, c in votes.items():
            tot = sum(c.values())
            m, k = c.most_common(1)[0]
            if tot < 3 or k / tot < 0.6:  # only reference rooms with a clear object-implied run room
                continue
            n_eval += 1
            agree_f += bool(foot.rep[i, m])
            agree_c += bool(cent.rep[i, m])
        matched_c = int(cent.rep.any(axis=1).sum())
        rep_f = foot.rep.any(axis=1)
        cent_false_missing = int(((~cent.rep.any(axis=1)) & rep_f).sum())
        cent_wrong = int(sum(1 for i, j in zip(*np.nonzero(cent.rep)) if not foot.rep[i, j]))
        evf = Q.evaluate(ref, v, cfg, queries)
        evc = Q.evaluate(ref, v, QueryConfig(room_method="centroid"), queries)
        only_c = sum(1 for a, b in zip(evf.results, evc.results) if a.ok and not b.ok)
        only_f = sum(1 for a, b in zip(evf.results, evc.results) if b.ok and not a.ok)
        rows.append({"run": n, "fault": info[n]["fault"], "severity": info[n]["severity"], "seed": info[n]["seed"], "role": info[n]["role"],
                     "run_rooms": v.n_room, "footprint_unrepresented_ref_rooms": int((~rep_f).sum()),
                     "footprint_structures": ";".join(foot.structures()), "centroid_unmatched_ref_rooms": ref.n_room - matched_c,
                     "centroid_false_missing_rooms": cent_false_missing, "centroid_matches_without_footprint_support": cent_wrong,
                     "implied_rooms_evaluated": n_eval, "implied_agree_footprint": agree_f, "implied_agree_centroid": agree_c,
                     "failure_rate_footprint": round(evf.summary()["failure_rate"], 4), "failure_rate_centroid": round(evc.summary()["failure_rate"], 4),
                     "queries_failed_only_by_centroid": only_c, "queries_failed_only_by_footprint": only_f,
                     "room_missing_footprint": evf.summary()["modes"][Q.ROOM_MISSING], "room_missing_centroid": evc.summary()["modes"][Q.ROOM_MISSING]})
    return pd.DataFrame(rows)


def sensitivity_table(ref, views, info, base_evs, base_queries):
    """Effect of every free parameter on the benign band and on the number of changed outcomes."""
    variants = [("default", CFG)]
    variants += [(f"tol_m={t}", QueryConfig(tol_m=t)) for t in (0.5, 0.75, 1.5, 2.0)]
    variants += [(f"rep_overlap={t}", QueryConfig(rep_overlap=t)) for t in (0.15, 0.2, 0.4, 0.5, 0.6)]
    variants += [(f"min_ref_share={t}", QueryConfig(min_ref_share=t)) for t in (0.0, 0.1, 0.2)]
    variants += [("gate_m=radius", QueryConfig(gate_m=None))] + [(f"gate_m={g}", QueryConfig(gate_m=g)) for g in (0.75, 1.5)]
    variants += [(f"cell_m={c}", QueryConfig(cell_m=c)) for c in (0.25, 1.0)]
    variants += [("exclusive=False", QueryConfig(exclusive=False)), ("room_method=centroid", QueryConfig(room_method="centroid"))]
    variants += [("room_method=centroid,gate=5", QueryConfig(room_method="centroid", centroid_gate_m=5.0))]
    base_modes = {n: [r.mode for r in ev.results] for n, ev in base_evs.items()}
    rows = []
    for label, cfg in variants:
        evs, _ = evaluate_all(ref, views, cfg)
        ben = [ev.summary()["failure_rate"] for n, ev in evs.items() if info[n]["role"] == "benign"]
        changed = changed_ben = total = total_ben = 0
        for n, ev in evs.items():
            ms = [r.mode for r in ev.results]
            diff = sum(a != b for a, b in zip(ms, base_modes[n]))
            changed += diff
            total += len(ms)
            if info[n]["role"] == "benign":
                changed_ben += diff
                total_ben += len(ms)
        main = {(info[n]["fault"], info[n]["severity"]): ev.summary()["failure_rate"] for n, ev in evs.items()
                if info[n]["role"] == "main"}
        rows.append({"variant": label, "benign_hi": round(max(ben), 4), "benign_mean": round(float(np.mean(ben)), 4),
                     "outcomes_changed_vs_default": changed, "of": total, "changed_in_benign_runs": changed_ben, "of_benign": total_ben,
                     "n_main_runs_above_benign_hi": int(sum(v > max(ben) + 1e-12 for v in main.values())), "n_main_runs": len(main)})
    return pd.DataFrame(rows)


def masked_table(runs_df):
    ben = runs_df[runs_df["role"] == "benign"]
    band = Q.benign_band(ben["failure_rate"])
    own = ben.groupby("fault")["failure_rate"].max().to_dict()
    sband = Q.benign_band(ben["strict_failure_rate"])
    rows = []
    for _, r in runs_df[runs_df["role"] == "main"].iterrows():
        rows.append({"run": r["run"], "fault": r["fault"], "severity": r["severity"], "seed": r["seed"],
                     "n_fail": r["n_fail"], "failure_rate": r["failure_rate"], "benign_hi_pooled": band.hi_rate,
                     "masked_pooled": Q.is_masked(r["failure_rate"], band), "benign_hi_own_fault": own[r["fault"]],
                     "masked_vs_own_control": r["failure_rate"] <= own[r["fault"]] + 1e-12,
                     "strict_failure_rate": r["strict_failure_rate"], "strict_benign_hi": sband.hi_rate,
                     "strict_masked": Q.is_masked(r["strict_failure_rate"], sband),
                     "adhoc_failure_rate": r["adhoc_failure_rate"], "adhoc_benign_hi": float(ben["adhoc_failure_rate"].max()),
                     "adhoc_masked": r["adhoc_failure_rate"] <= float(ben["adhoc_failure_rate"].max()) + 1e-12})
    return pd.DataFrame(rows), band


def border_analysis(ref, queries, evs, info):
    """Benign wrong_room failures versus distance of the reference object to another reference room's places (xy)."""
    d_other = []
    for q in queries:
        other = ref.place_room != q.ref_room_row
        d_other.append(float(np.linalg.norm(ref.place_pos[other][:, :2] - q.pos[:2], axis=1).min()))
    d_other = np.array(d_other)
    wr = [q.qid for n, ev in evs.items() if info[n]["role"] == "benign" for q, r in zip(queries, ev.results) if r.mode == Q.WRONG_ROOM]
    res = {"n_benign_wrong_room": len(wr), "n_reference_objects": len(queries),
           "reference_objects_within_1.0m_of_another_room": int((d_other <= 1.0).sum()),
           "reference_objects_within_1.5m_of_another_room": int((d_other <= 1.5).sum())}
    for thr in (1.0, 1.5):
        res[f"benign_wrong_room_within_{thr}m"] = int(sum(d_other[i] <= thr for i in wr))
    res["distinct_objects_in_benign_wrong_room"] = len(set(wr))
    return res


def pos_by_run(evs, info):
    rows = []
    for n, ev in evs.items():
        e = np.array([r.err_any_m for r in ev.results])
        ef = e[np.isfinite(e)]
        rows.append({"run": n, "fault": info[n]["fault"], "severity": info[n]["severity"], "seed": info[n]["seed"], "role": info[n]["role"],
                     **{f"q{int(p * 100):02d}_m": round(float(np.quantile(ef, p)), 3) for p in (.5, .75, .9, .95)},
                     **{f"frac_le_{t}": round(float((e <= t).mean()), 3) for t in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)}})
    return pd.DataFrame(rows)


def label_rule_effect(ref, views, evs, queries, tol):
    """Failed queries that have an other-class node within tol: the explained-label rule kept them out of wrong_label."""
    kept = total_fail = 0
    for n, ev in evs.items():
        v = views[n]
        for q, r in zip(queries, ev.results):
            if r.ok or r.mode in (Q.ROOM_MISSING, Q.WRONG_LABEL):
                continue
            total_fail += 1
            d = np.linalg.norm(v.obj_pos - q.pos, axis=1)
            kept += bool(((v.obj_label != q.label) & (d <= tol)).any())
    return kept, total_fail


# ------------------------------------------------------------------------------------------------ main
def main():
    TABLES.mkdir(exist_ok=True)
    BIG.mkdir(parents=True, exist_ok=True)
    views, info = load_all()
    ref = views["uh2_clean"]
    evs, queries = evaluate_all(ref, views, CFG)
    runs = run_rows(evs, info, views, ref, CFG)
    runs.to_csv(TABLES / "runs.csv", index=False)
    pd.DataFrame([r for ev in evs.values() for r in ev.rows()]).to_csv(BIG / "query_rows_all_runs.csv", index=False)

    out = {"config": {k: getattr(CFG, k) for k in CFG.__dataclass_fields__}, "n_runs": len(evs),
           "runs": {r: info[r] for r in sorted(info)}}
    out["reference"] = reference_informativeness(ref, queries, CFG)
    (TABLES / "reference_informativeness.json").write_text(json.dumps(out["reference"], indent=1))

    # benign band and masked faults
    masked, band = masked_table(runs)
    masked.to_csv(TABLES / "masked_faults.csv", index=False)
    ben = runs[runs["role"] == "benign"][["run", "fault", "seed", "n_fail", "failure_rate", "strict_failure_rate", "adhoc_failure_rate"] + [f"n_{m}" for m in Q.FAILURE_MODES]]
    ben.to_csv(TABLES / "benign_runs.csv", index=False)
    out["benign_band"] = {"n_runs": band.n_runs, "hi_rate": band.hi_rate, "rates_sorted": list(band.rates),
                          "hi_strict": float(runs[runs.role == "benign"].strict_failure_rate.max()),
                          "hi_adhoc": float(runs[runs.role == "benign"].adhoc_failure_rate.max()),
                          "clean_b_failure_rate": float(runs[runs.run == "uh2_clean_b"].failure_rate.iloc[0])}

    # severity curve and modes per fault
    main_runs = runs[runs["role"].isin(["benign", "main"])]
    curve = main_runs.groupby(["fault", "severity"]).agg(n_runs=("run", "count"), failure_rate=("failure_rate", "mean"),
                                                        failure_rate_min=("failure_rate", "min"), failure_rate_max=("failure_rate", "max"),
                                                        strict_failure_rate=("strict_failure_rate", "mean"),
                                                        adhoc_failure_rate=("adhoc_failure_rate", "mean"),
                                                        **{f"mean_{m}": (f"n_{m}", "mean") for m in Q.FAILURE_MODES}).reset_index()
    curve["_f"] = curve["fault"].map({f: i for i, f in enumerate(FAULTS)})
    curve = curve.sort_values(["_f", "severity"]).drop(columns="_f").round(4)
    curve.to_csv(TABLES / "severity_curve.csv", index=False)
    modes = main_runs.groupby(["fault"])[[f"n_{m}" for m in Q.FAILURE_MODES] + ["n_fail"]].sum().reindex(FAULTS)
    modes.to_csv(TABLES / "modes_by_fault_all_severities.csv")

    # tolerance
    pe, hist, tail = position_error_tables(evs, info)
    pe.to_csv(TABLES / "benign_position_error.csv", index=False)
    hist.to_csv(TABLES / "benign_position_error_hist.csv", index=False)
    out["benign_position_error_top25_m"] = [round(float(x), 3) for x in tail]
    tol = tolerance_sweep(ref, views, info, (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0))
    tol.to_csv(TABLES / "tolerance_sweep.csv", index=False)

    # room matching
    rm = room_matching_table(ref, views, info, queries, CFG)
    rm.to_csv(TABLES / "room_matching.csv", index=False)

    # sensitivity
    sens = sensitivity_table(ref, views, info, evs, queries)
    sens.to_csv(TABLES / "sensitivity.csv", index=False)

    out["benign_wrong_room_at_borders"] = border_analysis(ref, queries, evs, info)
    out["uninformed_pick_hit_rate"] = float(np.mean([1.0 / q.n_same_class_in_room for q in queries]))
    pos_by_run(evs, info).to_csv(TABLES / "position_error_by_run.csv", index=False)
    pd.DataFrame([{"qid": q.qid, "ref_node": Q.node_name(q.ref_id), "label": q.label, "label_name": q.label_name,
                   "x": round(float(q.pos[0]), 3), "y": round(float(q.pos[1]), 3), "z": round(float(q.pos[2]), 3),
                   "ref_places": ";".join(Q.node_name(p) for p in q.ref_place_ids), "ref_room": Q.node_name(q.ref_room_id),
                   "n_same_class_in_room": q.n_same_class_in_room, "n_neighbours_within_2tol": len(q.nbr_row)} for q in queries]
                  ).to_csv(TABLES / "reference_queries.csv", index=False)
    pd.DataFrame(evs["uh2_pose_drift_s50"].rows()).to_csv(TABLES / "example_query_rows_pose_drift_s50.csv", index=False)

    kept, tot = label_rule_effect(ref, views, evs, queries, CFG.tol_m)
    out["explained_label_rule"] = {"failed_queries_not_label_or_room": tot, "with_other_class_node_within_tol": kept}

    # informativeness of the run side
    out["run_side"] = {"mean_same_class_nodes_in_representing_rooms": float(np.mean([r.n_same_class_in_rooms for ev in evs.values() for r in ev.results])),
                       "share_of_ok_queries_with_more_than_one_near_candidate": float(np.mean([r.n_near > 1 for ev in evs.values() for r in ev.results if r.ok]))}
    (TABLES / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"runs: {len(evs)}; tables in {TABLES}")
    print(runs[["run", "role", "severity", "n_fail", "failure_rate", "strict_failure_rate", "adhoc_failure_rate"]].to_string())
    print("benign band", out["benign_band"])


if __name__ == "__main__":
    sys.exit(main())
