"""Compare the independent recomputation (recompute.py outputs) with the code and with the stored eval_dev_v1 files.
Also: re-run report/select into a scratch dir (reproducibility), hand checks of bootstrap_ci and loo_false_alarm.
Run with PYTHONPATH=D:/Hydra/layer_attribution/src.
"""
import json
import pathlib
import tempfile

import numpy as np
import pandas as pd

from layer_attribution import calibration as C
from layer_attribution import diagnosis as D
from layer_attribution.evaluation import evaluate as EV
from layer_attribution.evaluation import metrics as M

T = pathlib.Path("D:/hydra_work/results/tables/dev")
E = pathlib.Path("D:/hydra_work/results/eval_dev_v1")
A = pathlib.Path("D:/Hydra/layer_attribution/research/audit/recompute_out")
out = {}

table, runmeta = EV.load_feature_table(T / "features.csv")
meta = EV.load_feature_meta(T / "features_meta.csv")
queries = EV.load_queries(T / "queries.csv")
ben = EV.benign_runs(runmeta)
cal = C.fit(table, meta, ben, EV.REFERENCE, C.MARGIN)

# ---- (a) thresholds / centers
mine = pd.read_csv(A / "my_thresholds.csv", index_col=0)
dc = (cal.center.reindex(mine.index) - mine["center"]).abs()
dt = (cal.threshold.reindex(mine.index) - mine["thr"]).abs()
nan_mismatch = [n for n in mine.index if pd.isna(cal.threshold.get(n)) != pd.isna(mine.at[n, "thr"])]
rel = (dt / mine["thr"].abs().clip(lower=1e-12)).max()
out["thresholds"] = dict(n_features=len(mine), max_abs_center_diff=float(np.nanmax(dc)), max_abs_thr_diff=float(np.nanmax(dt)),
                         max_rel_thr_diff=float(rel), nan_mismatch=nan_mismatch,
                         n_nan_thresholds_code=int(cal.threshold.isna().sum()), n_nan_thresholds_mine=int(mine["thr"].isna().sum()),
                         nan_threshold_features=list(cal.threshold.index[cal.threshold.isna()]))
# ---- strengths
mS = pd.read_csv(A / "my_strengths.csv", index_col=0)
cS = cal.strengths(table)
common = [c for c in mS.columns if c in cS.columns]
d = (cS[common].reindex(mS.index) - mS[common]).abs()
nanmm = int((cS[common].reindex(mS.index).isna() != mS[common].isna()).sum().sum())
out["strengths"] = dict(shape=list(mS.shape), max_abs_diff=float(np.nanmax(d.values)), nan_pattern_mismatches=nanmm)

# ---- stored predictions vs mine
P = pd.read_csv(E / "predictions_dev.csv")
P["scores"] = P["scores"].map(json.loads)
R = pd.read_csv(A / "my_records.csv").fillna({"blame": "", "deviating": ""})
P = P.fillna({"blame": "", "deviating": ""})
disc = []
maxd = 0.0
cmp_rows = 0
for _, r in R.iterrows():
    p = P[(P.run == r.run) & (P.method == r.method)]
    if len(p) != 1:
        disc.append(f"missing {r.run} {r.method}")
        continue
    p = p.iloc[0]
    cmp_rows += 1
    if p.blame != r.blame:
        disc.append(f"blame {r.run} {r.method}: code={p.blame!r} mine={r.blame!r}")
    if p.deviating != r.deviating:
        disc.append(f"deviating {r.run} {r.method}: code={p.deviating!r} mine={r.deviating!r}")
    if bool(p.masked) != bool(r.masked):
        disc.append(f"masked {r.run}")
    for L in ("mesh", "objects", "places", "rooms"):
        dd = abs(p.scores[L] - r[f"s_{L}"])
        maxd = max(maxd, dd)
        if dd > 1e-9:
            disc.append(f"score {r.run} {r.method} {L}: code={p.scores[L]} mine={r[f's_{L}']}")
out["predictions_vs_mine"] = dict(rows_compared=cmp_rows, max_abs_score_diff=maxd, discrepancies=disc[:20], n_discrepancies=len(disc))

# ---- summary numbers vs mine
S = json.load(open(E / "summary_dev.json"))
mysum = json.load(open(A / "my_summary.json"))
sd = []
for meth in ("dag_rb", "naive_rb", "dag_rf", "naive_rf", "downstream_rb", "downstream_rf"):
    code = S["accuracy_by_fault"][meth]
    for f_, v in mysum[meth]["per_fault"].items():
        if abs(code[f_]["acc"] - v) > 1e-9 or code[f_]["n"] != mysum[meth]["n"][f_]:
            sd.append(f"{meth} {f_}: code {code[f_]} mine {v}")
    if abs(code["macro"]["acc"] - mysum[meth]["macro"]) > 1e-4:
        sd.append(f"{meth} macro code {code['macro']['acc']} mine {mysum[meth]['macro']}")
    allr = S["accuracy_all_runs_by_fault"][meth]
    if abs(allr["macro"]["acc"] - mysum[meth]["all_runs_macro"]) > 1e-4:
        sd.append(f"{meth} all-runs macro code {allr['macro']['acc']} mine {mysum[meth]['all_runs_macro']}")
    st = S["accuracy_strict_by_fault"][meth]
    if abs(st["macro"]["acc"] - mysum[meth]["strict_macro"]) > 1e-4:
        sd.append(f"{meth} strict macro code {st['macro']['acc']} mine {mysum[meth]['strict_macro']}")
for meth in ("dag_rb", "naive_rb", "dag_rf"):
    code_c = S["confusion"][meth]
    for acc_row, cols in mysum[meth]["confusion"].items():
        for col, n in cols.items():
            if code_c.get(acc_row, {}).get(col, 0) != n:
                sd.append(f"confusion {meth} {acc_row}/{col}: code {code_c.get(acc_row, {}).get(col)} mine {n}")
if abs(S["random_expected"] - mysum["random_expected_micro_over_runs"]) > 1e-12:
    sd.append("random_expected differs")
out["summary_vs_mine"] = dict(discrepancies=sd, code_random=S["random_expected"], mine_micro=mysum["random_expected_micro_over_runs"],
                              mine_macro=mysum["random_expected_macro_over_faults"],
                              code_prior=S["prior_layer"], mine_prior_candidates=mysum["prior_candidates_macro_lenient"])

# ---- reproducibility: re-run report and select into a scratch dir (same inputs)
with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    argv = ["--features", str(T / "features.csv"), "--meta", str(T / "features_meta.csv"), "--queries", str(T / "queries.csv"), "--out", str(td)]
    EV.main(["select"] + argv)
    g_new = pd.read_csv(td / "grid.csv"); g_old = pd.read_csv(E / "grid.csv")
    same_grid = bool(np.allclose(g_new.select_dtypes("number").values, g_old.select_dtypes("number").values, equal_nan=True)
                     and list(g_new.config) == list(g_old.config))
    same_cfg = json.loads((td / "chosen_config.json").read_text()) == json.loads((E / "chosen_config.json").read_text())
    EV.main(["report"] + argv)
    p_new = pd.read_csv(td / "predictions_dev.csv"); p_old = pd.read_csv(E / "predictions_dev.csv")
    same_pred = bool(p_new.fillna("").equals(p_old.fillna("")))
    s_new = json.load(open(td / "summary_dev.json"));
    same_sum = (json.dumps(s_new, sort_keys=True) == json.dumps(S, sort_keys=True))
out["reproduce"] = dict(grid_identical=same_grid, chosen_identical=same_cfg, predictions_identical=same_pred, summary_identical=same_sum)

# ---- selection fallback check: which rows tie at the minimum false-alarm rate, and what a tie-aware rule would pick
g = pd.read_csv(E / "grid.csv")
sel = {}
for mode in ("rb", "rf"):
    gg = g[g["mode"] == mode]
    ok = gg[gg["loo_false_alarm"] <= EV.MAX_LOO_FALSE_ALARM]
    mn = gg["loo_false_alarm"].min()
    tied = gg[gg["loo_false_alarm"] == mn]
    best_tie_aware = tied.sort_values(["dag_macro", "k", "exclude_count_based"], ascending=[False, False, False]).iloc[0]
    sel[mode] = dict(n_meeting_cap=int(len(ok)), min_false_alarm=float(mn), tied_configs=list(tied.config),
                     code_pick=json.loads((E / "chosen_config.json").read_text())[mode],
                     tie_aware_pick=f"{best_tie_aware['mode']}-k{int(best_tie_aware['k'])}{'-nocount' if best_tie_aware['exclude_count_based'] else ''}")
out["selection_fallback"] = sel

# ---- hand check: bootstrap_ci
x = [1, 1, 0, 0]
m_, lo, hi = M.bootstrap_ci(x, n_boot=20000, seed=3)
out["bootstrap_hand"] = dict(values=x, code=(m_, lo, hi), exact_expected=(0.5, 0.0, 1.0),
                             note="exact bootstrap of mean(1,1,0,0): P(mean=0)=1/16, P(mean=1)=1/16 -> 2.5 %=0.0, 97.5 %=1.0")
x2 = [1, 1, 1, 0, 0, 0, 0, 0]  # exact: Binomial(8, 3/8)/8 -> compute exact quantiles
from math import comb
pm = [comb(8, j) * (3 / 8) ** j * (5 / 8) ** (8 - j) for j in range(9)]
cdf = np.cumsum(pm)
lo_e = next(j for j in range(9) if cdf[j] >= 0.025) / 8
hi_e = next(j for j in range(9) if cdf[j] >= 0.975) / 8
m2, lo2, hi2 = M.bootstrap_ci(x2, n_boot=200000, seed=1)
out["bootstrap_hand2"] = dict(code=(m2, lo2, hi2), exact_discrete_quantiles=(3 / 8, lo_e, hi_e))
# macro CI vs exact convolution for the stored dag_rf case (6 groups at 1.0, one group [1,0])
recs = [dict(run=f"a{i}", fault="a", severity=0.5, seed=0, split="dev", kind="sensor", accept=["objects"], masked=False,
             failure_rate=0.5, method="m", blame=b, scores={}, deviating=[]) for i, b in enumerate((["objects"], ["mesh"]))]
recs += [dict(run=f"b{i}", fault="b", severity=0.5, seed=0, split="dev", kind="sensor", accept=["objects"], masked=False,
              failure_rate=0.5, method="m", blame=["objects"], scores={}, deviating=[]) for i in range(3)]
acc = M.accuracy(recs, "m", n_boot=20000)
out["macro_ci_hand"] = dict(code=acc, exact="groups a=[1,0] b=[1,1,1]: macro in {0.5,0.75,1.0} with probs {1/4,1/2,1/4} -> mean 0.75 (point 0.75), 2.5%=0.5, 97.5%=1.0")

# ---- hand check: loo_false_alarm on a 3-run example
names = ["uh2_clean", "b1", "b2", "b3"]
tab = pd.DataFrame({"f1": [0.0, 0.0, 0.05, 0.4], "f2": [1.0, 1.0, 1.0, 1.0]}, index=names)
mt = pd.DataFrame({"kind": ["reference_based", "reference_free"], "layer": ["mesh", "objects"], "ideal": [0.0, np.nan],
                   "tol_abs": [0.1, 0.1], "tol_rel": [0.0, 0.0], "count_based": [False, False]}, index=["f1", "f2"])
rm = pd.DataFrame({"fault": ["x", "x", "y"]}, index=["b1", "b2", "b3"])
cfg1 = D.ScoreConfig("rb", 1, False)
r_run = EV.loo_false_alarm(tab, mt, ["b1", "b2", "b3"], cfg1, 1.5, "run", rm)
r_fam = EV.loo_false_alarm(tab, mt, ["b1", "b2", "b3"], cfg1, 1.5, "family", rm)
# by hand (rb, f1 only, k=1): hold b3 -> rest max dev .05 -> thr max(.075,.1)=.1 -> strength 4 -> alarm.
# hold b1 -> rest {b2,b3} max .4 -> thr .6 -> b1 strength 0 -> none; hold b2 -> thr .6 -> .05/.6 -> none. => 1/3.
# family: hold {b1,b2} (fault x) -> rest {b3}: dev .4 -> thr .6; b1: 0, b2: .083 -> no alarms; hold b3 -> rest {b1,b2}: thr .1 -> b3 alarm. => 1/3
out["loo_hand"] = dict(code_run=r_run, code_family=r_fam, by_hand_run_rate=1 / 3, by_hand_family_rate=1 / 3,
                       by_hand_per_layer_mesh=1 / 3)
(pathlib.Path("D:/Hydra/layer_attribution/research/audit/compare_out.json")).write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
