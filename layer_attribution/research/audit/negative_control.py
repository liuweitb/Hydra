"""Negative control: permute the injected-fault labels among the dev fault runs, keep the features, rerun the real pipeline.
Fast path: blame / mask are label-free, so only accept sets move (20000 permutations). Slow path: the real evaluate code
(build_records and select) on 300 / 25 permuted run tables, to confirm the fast path and to measure selection inflation.
Run with PYTHONPATH=D:/Hydra/layer_attribution/src.
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

from layer_attribution import dag
from layer_attribution import diagnosis as D
from layer_attribution.evaluation import evaluate as EV
from layer_attribution.evaluation import metrics as M

T = pathlib.Path("D:/hydra_work/results/tables/dev")
E = pathlib.Path("D:/hydra_work/results/eval_dev_v1")
table, runmeta = EV.load_feature_table(T / "features.csv")
meta = EV.load_feature_meta(T / "features_meta.csv")
queries = EV.load_queries(T / "queries.csv")
chosen = {m: D.ScoreConfig(**v) for m, v in json.loads((E / "chosen_config.json").read_text()).items()}

recs, cal, band = EV.build_records(table, runmeta, meta, queries, chosen, ("dev",))
real = {mth: M.accuracy(recs, mth)["macro"]["acc"] for mth in ("dag_rb", "naive_rb", "dag_rf", "naive_rf", "downstream_rb", "symptom")}
print("real macro (lenient):", {k: round(v, 4) for k, v in real.items()})

dev_runs = [r for r in runmeta.index if runmeta.at[r, "split"] == "dev" and runmeta.at[r, "kind"] in ("sensor", "config")]
faults = [runmeta.at[r, "fault"] for r in dev_runs]
methods = ("dag_rb", "naive_rb", "dag_rf", "naive_rf", "downstream_rb", "downstream_rf")
# label-free pieces per run
blame = {m_: {} for m_ in methods}
masked = {}
for r in recs:
    if r["method"] in methods:
        blame[r["method"]][r["run"]] = r["blame"][0] if r["blame"] else None
    masked[r["run"]] = r["masked"]
rng = np.random.default_rng(12345)
N = 20000


def macro(meth, labels):
    per = {}
    for r, f in zip(dev_runs, labels):
        if masked[r]:
            continue
        b = blame[meth][r]
        per.setdefault(f, []).append(float(b is not None and b in dag.accept_set(f)))
    return float(np.mean([np.mean(v) for v in per.values()])) if per else float("nan")


null = {m_: np.empty(N) for m_ in methods}
for i in range(N):
    perm = list(rng.permutation(faults))
    for m_ in methods:
        null[m_][i] = macro(m_, perm)
# fast path check on the identity permutation
assert abs(macro("dag_rb", faults) - real["dag_rb"]) < 1e-12 and abs(macro("naive_rb", faults) - real["naive_rb"]) < 1e-12
res = {}
for m_ in methods:
    x = null[m_]
    res[m_] = dict(real=real.get(m_, macro(m_, faults)), null_mean=float(x.mean()), null_p2_5=float(np.quantile(x, 0.025)),
                   null_p97_5=float(np.quantile(x, 0.975)), null_max=float(x.max()), p_ge_real=float((x >= macro(m_, faults) - 1e-12).mean()))
print("fast path (20000 label permutations) macro lenient accuracy:")
for k, v in res.items():
    print(f"  {k:14s} real {v['real']:.3f}  shuffled mean {v['null_mean']:.3f}  2.5-97.5 % [{v['null_p2_5']:.3f},{v['null_p97_5']:.3f}] max {v['null_max']:.3f}  P(null>=real)={v['p_ge_real']:.5f}")

# slow path: real code on permuted run tables
slow = []
for i in range(300):
    perm = list(rng.permutation(faults))
    rm2 = runmeta.copy()
    rm2.loc[dev_runs, "fault"] = perm
    rr, _, _ = EV.build_records(table, rm2, meta, queries, chosen, ("dev",))
    slow.append({m_: M.accuracy(rr, m_).get("macro", {}).get("acc", np.nan) for m_ in ("dag_rb", "naive_rb", "dag_rf")})
sl = pd.DataFrame(slow)
print("slow path (real build_records, 300 permutations): mean", sl.mean().round(3).to_dict(), " max", sl.max().round(3).to_dict())

# selection under shuffled labels: best-of-12 dag_macro per mode
best = []
for i in range(25):
    perm = list(rng.permutation(faults))
    rm2 = runmeta.copy()
    rm2.loc[dev_runs, "fault"] = perm
    grid, ch = EV.select(table, rm2, meta, queries)
    best.append(dict(rb_chosen_macro=float(grid[(grid["mode"] == "rb")].set_index("config").loc[ch["rb"].label(), "dag_macro"]),
                     rf_chosen_macro=float(grid[(grid["mode"] == "rf")].set_index("config").loc[ch["rf"].label(), "dag_macro"]),
                     rb_max=float(grid[grid["mode"] == "rb"]["dag_macro"].max()), rf_max=float(grid[grid["mode"] == "rf"]["dag_macro"].max())))
bd = pd.DataFrame(best)
print("select() under 25 shuffled-label tables: mean chosen dag_macro rb/rf", bd[["rb_chosen_macro", "rf_chosen_macro"]].mean().round(3).to_dict(),
      " mean best-of-6 dag_macro rb/rf", bd[["rb_max", "rf_max"]].mean().round(3).to_dict(), " max", bd.max().round(3).to_dict())
out = dict(fast=res, slow_mean=sl.mean().to_dict(), slow_max=sl.max().to_dict(), select_shuffled=bd.mean().to_dict(), select_shuffled_max=bd.max().to_dict())
pathlib.Path("D:/Hydra/layer_attribution/research/audit/negative_control_out.json").write_text(json.dumps(out, indent=1))
