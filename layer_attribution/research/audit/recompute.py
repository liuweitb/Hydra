"""Independent re-implementation of calibration, layer scores, DAG / naive blame and the dev metrics.
Plain pandas / numpy; does NOT import layer_attribution. Reads the dev tables and writes recompute_out/*.csv|json.
Inputs only: features.csv, features_meta.csv, queries.csv, chosen_config.json (config choice is taken as given).
"""
import json
import pathlib

import numpy as np
import pandas as pd

T = pathlib.Path("D:/hydra_work/results/tables/dev")
E = pathlib.Path("D:/hydra_work/results/eval_dev_v1")
OUT = pathlib.Path("D:/Hydra/layer_attribution/research/audit/recompute_out")
OUT.mkdir(exist_ok=True)

LAYERS = ["mesh", "objects", "places", "rooms"]
TAG2LAYER = {"mesh_geom": "mesh", "mesh_sem": "objects", "objects": "objects", "places": "places", "rooms": "rooms"}
PARENTS = {"mesh": [], "places": [], "objects": ["mesh"], "rooms": ["places"]}  # data-flow DAG (DESIGN / PROJECT_CONTEXT)
ACCEPT = {"label_confusion": {"objects"}, "label_dropout": {"objects"}, "depth_noise": {"mesh", "places"},
          "depth_holes": {"mesh", "places"}, "pose_drift": {"mesh", "places"}, "frame_drop": {"mesh", "places"},
          "objects_min_size": {"objects"}, "objects_tol_merge": {"objects"}, "rooms_min_size": {"rooms"},
          "rooms_max_dilation": {"rooms"}}
MARGIN = 1.5
CAP = 1e3
EXCLUDE = {"depth_holes"}

f = pd.read_csv(T / "features.csv")
m = pd.read_csv(T / "features_meta.csv").set_index("name")
q = pd.read_csv(T / "queries.csv").set_index("run")
run_meta = f[["run", "split", "kind", "fault", "severity", "seed"]].set_index("run")
X = f.drop(columns=["split", "kind", "fault", "severity", "seed"]).set_index("run").astype(float)
cfg = json.loads((E / "chosen_config.json").read_text())

# evidence layer derived from the tag, cross-checked against the stored evidence_layer column
m["ev"] = m["layer"].map(TAG2LAYER)
mismatch = m[(m["ev"].fillna("-") != m["evidence_layer"].fillna("-"))]
print("evidence_layer column vs tag mapping mismatches:", len(mismatch))

# ---------------------------------------------------------------- (a) thresholds
benign = [r for r in run_meta.index if run_meta.at[r, "split"] == "benign" and run_meta.at[r, "fault"] not in EXCLUDE]
print("benign calibration runs:", len(benign))
REF = "uh2_clean"
center, thr, bmax = {}, {}, {}
for name in m.index:
    x = X[name]
    if m.at[name, "kind"] == "reference_based":
        c = m.at[name, "ideal"]
    else:
        c = np.nanmedian(x.loc[[REF] + benign].values) if x.loc[[REF] + benign].notna().any() else np.nan
    center[name] = c
    dev = (x.loc[benign] - c).abs()
    floor = (0.0 if pd.isna(m.at[name, "tol_abs"]) else m.at[name, "tol_abs"]) + \
            (0.0 if pd.isna(m.at[name, "tol_rel"]) else m.at[name, "tol_rel"]) * (abs(c) if not pd.isna(c) else np.nan)
    if pd.isna(c) or dev.isna().all():
        thr[name], bmax[name] = np.nan, np.nan
    else:
        bmax[name] = dev.max()
        thr[name] = max(MARGIN * dev.max(), floor)          # docstring formula, no 'tiny' guard
center = pd.Series(center); thr = pd.Series(thr); bmax = pd.Series(bmax)
cal = pd.DataFrame({"center": center, "thr": thr, "benign_max_dev": bmax, "kind": m["kind"], "ev": m["ev"],
                    "count_based": m["count_based"]})
cal.to_csv(OUT / "my_thresholds.csv")
print("features with threshold<=0 under the docstring formula:", int((thr <= 0).sum()),
      list(thr.index[thr <= 0]))

# ---------------------------------------------------------------- (b) strengths and layer scores
S = (X[cal.index].sub(center, axis=1)).abs().div(thr, axis=1)
S = S.clip(upper=CAP)
S.to_csv(OUT / "my_strengths.csv")


def usable(mode):
    kinds = {"rb": ["reference_based"], "rf": ["reference_free"], "both": ["reference_based", "reference_free"]}[mode]
    return cal.index[cal["kind"].isin(kinds) & cal["ev"].notna() & cal["thr"].notna()]


def layer_scores(run, mode, k, nocount=False):
    u = usable(mode)
    if nocount:
        u = [n for n in u if not bool(cal.at[n, "count_based"])]
    out = {}
    for L in LAYERS:
        names = [n for n in u if cal.at[n, "ev"] == L]
        v = np.sort(S.loc[run, names].values.astype(float))
        v = v[np.isfinite(v)][::-1]
        out[L] = float(v[min(k, len(v)) - 1]) if len(v) else 0.0
    return out


def roots(dev):
    def anc(l, seen=None):
        seen = set() if seen is None else seen
        for p in PARENTS[l]:
            if p not in seen:
                seen.add(p); anc(p, seen)
        return seen
    return [l for l in LAYERS if l in dev and not (anc(l) & set(dev))]


def dag_blame(sc):
    dev = [l for l in LAYERS if sc[l] >= 1.0]
    return sorted(roots(dev), key=lambda l: -sc[l]), dev


ties = []


def naive_blame(sc):
    dev = [l for l in LAYERS if sc[l] >= 1.0]
    if not dev:
        return [], dev
    mx = max(sc.values())
    tied = [l for l in LAYERS if sc[l] == mx]
    if len(tied) > 1:
        ties.append(tied)
    return [tied[0]], dev


def downstream_blame(sc):
    dev = [l for l in LAYERS if sc[l] >= 1.0]
    def depth(l):
        seen = set(); todo = list(PARENTS[l])
        while todo:
            p = todo.pop()
            if p not in seen:
                seen.add(p); todo.extend(PARENTS[p])
        return len(seen)
    return sorted(dev, key=lambda l: (-depth(l), -sc[l]))[:1], dev


# ---------------------------------------------------------------- masking
all_benign = [r for r in run_meta.index if run_meta.at[r, "split"] == "benign"]
hi = q.loc[all_benign, "failure_rate"].max()
hi_excl = q.loc[benign, "failure_rate"].max()
print(f"benign band hi (all 12 benign) = {hi:.6f};  hi without depth_holes benign = {hi_excl:.6f}")

# ---------------------------------------------------------------- records
scored = [r for r in run_meta.index if run_meta.at[r, "split"] == "dev" and run_meta.at[r, "kind"] in ("sensor", "config")]
rows = []
for r in scored:
    fault = run_meta.at[r, "fault"]
    masked = bool(q.at[r, "failure_rate"] <= hi + 1e-12)
    for mode in ("rb", "rf"):
        c = cfg[mode]
        sc = layer_scores(r, mode, c["k"], c["exclude_count_based"])
        for meth, fn in (("dag", dag_blame), ("naive", naive_blame), ("downstream", downstream_blame)):
            blame, dev = fn(sc)
            acc = ACCEPT[fault]
            rows.append(dict(run=r, fault=fault, severity=run_meta.at[r, "severity"], method=f"{meth}_{mode}", blame="+".join(blame),
                             deviating="+".join(dev), masked=masked, accept="+".join(sorted(acc)),
                             lenient=bool(blame and blame[0] in acc), strict=bool(blame and set(blame) == acc),
                             **{f"s_{L}": sc[L] for L in LAYERS}))
R = pd.DataFrame(rows)
R.to_csv(OUT / "my_records.csv", index=False)
print("naive ties encountered (layers sharing the max score):", ties)


def acc_table(R, method, include_masked=False, col="lenient"):
    d = R[R.method == method]
    d = d if include_masked else d[~d.masked]
    per = d.groupby("fault")[col].mean()
    n = d.groupby("fault")[col].size()
    return per, n, float(per.mean())


summary = {}
for meth in ("dag_rb", "naive_rb", "dag_rf", "naive_rf", "downstream_rb", "downstream_rf"):
    per, n, macro = acc_table(R, meth)
    perA, nA, macroA = acc_table(R, meth, include_masked=True)
    perS, nS, macroS = acc_table(R, meth, col="strict")
    summary[meth] = dict(per_fault=per.round(4).to_dict(), n=n.to_dict(), macro=round(macro, 4),
                         all_runs_per_fault=perA.round(4).to_dict(), all_runs_macro=round(macroA, 4),
                         strict_per_fault=perS.round(4).to_dict(), strict_macro=round(macroS, 4))
# confusion
for meth in ("dag_rb", "naive_rb", "dag_rf"):
    d = R[(R.method == meth) & (~R.masked)].copy()
    d["blamed"] = d.blame.map(lambda s: s.split("+")[0] if s else "none")
    summary[meth]["confusion"] = {k: v.to_dict() for k, v in d.groupby(["accept", "blamed"]).size().unstack(fill_value=0).iterrows()}
# exact expected random
u = R[(R.method == "dag_rb") & (~R.masked)]
micro = float(np.mean([len(ACCEPT[x]) / 4 for x in u.fault]))
macro_rand = float(np.mean([len(ACCEPT[x]) / 4 for x in sorted(set(u.fault))]))
summary["random_expected_micro_over_runs"] = micro
summary["random_expected_macro_over_faults"] = macro_rand
# prior
cand = {}
for L in LAYERS:
    per_fault = u.groupby("fault").apply(lambda g: np.mean([L in ACCEPT[x] for x in g.fault]))
    cand[L] = float(per_fault.mean())
summary["prior_candidates_macro_lenient"] = cand
summary["n_scored"] = int(len(u)); summary["n_masked"] = int(R[R.method == "dag_rb"].masked.sum())
(OUT / "my_summary.json").write_text(json.dumps(summary, indent=1, default=str))
print(json.dumps({k: v for k, v in summary.items() if k in ("random_expected_micro_over_runs", "random_expected_macro_over_faults",
      "prior_candidates_macro_lenient", "n_scored", "n_masked")}, indent=1))
for meth in ("dag_rb", "naive_rb", "dag_rf", "naive_rf"):
    s = summary[meth]
    print(meth, "macro", s["macro"], "per_fault", s["per_fault"], "| all-runs macro", s["all_runs_macro"], "| strict macro", s["strict_macro"])
