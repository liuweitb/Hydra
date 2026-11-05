#!/usr/bin/env python3
"""Join features, query outcomes and injected ground truth into diagnosis records; select a config on development runs;
report accuracy, confusion, masked faults and propagation. The only place that reads ground truth.

  python -m layer_attribution.evaluation.evaluate select --features F.csv --meta M.csv --queries Q.csv --out DIR
  python -m layer_attribution.evaluation.evaluate report --features F.csv --meta M.csv --queries Q.csv --out DIR
         [--splits dev] [--final]      (--final is required to score the held-out 'test' split)

Inputs: F = wide feature table (features_extract.py: run, split, kind, fault, severity, seed, then one column per feature),
M = feature meta (name, kind, layer or evidence_layer, ideal, tol_abs, tol_rel, count_based), Q = query table (query_table.py).

Protocol. Thresholds are fitted on the benign runs of CALIBRATION_SEEDS, which are disjoint from the development seed (0) and
the test seeds (1, 2) so no calibration run shares random draws with a scored run. The benign runs of VALIDATION_SEEDS are
never used for thresholds: they give the held-out false-alarm rate that the selection checks. A fault run is MASKED when its
query failure rate does not exceed that of the same fault family at negligible dose (pooled over all benign runs for config
faults, which have no negligible-dose runs); masked runs have no failure to explain and are reported, not scored.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from layer_attribution import calibration as C
from layer_attribution import dag
from layer_attribution import diagnosis as D
from layer_attribution import queries as Q
from layer_attribution import runs as R
from layer_attribution.evaluation import metrics as M

META_COLS = ["run", "split", "kind", "fault", "severity", "seed"]
REFERENCE = R.REFERENCE
BENIGN_EXCLUDE = ("depth_holes",)  # step response: the smallest dose already changes the mesh by +27 % (docs/FINDINGS_ROUND1.md)
CALIBRATION_SEEDS = (3, 4, 5)
VALIDATION_SEEDS = (0, 1)
MARGINS = (1.5, 2.0, 3.0)
GRID = [D.ScoreConfig(mode, k, ex, m) for mode in ("rb", "rf") for k in (1, 2, 3) for ex in (False, True) for m in MARGINS]
MAX_FALSE_ALARM = 0.05


# ---------------------------------------------------------------------------------------------------- loading
def load_feature_table(path):
    """(table indexed by run with feature columns only, run meta indexed by run)."""
    df = pd.read_csv(path)
    if "run" not in df.columns:
        raise ValueError(f"{path}: no 'run' column")
    meta = df[[c for c in META_COLS if c in df.columns]].set_index("run")
    return df.drop(columns=[c for c in META_COLS if c in df.columns and c != "run"]).set_index("run"), meta


def load_feature_meta(path) -> pd.DataFrame:
    m = pd.read_csv(path)
    name = "name" if "name" in m.columns else "feature"
    if "evidence_layer" in m.columns:
        layer = m["evidence_layer"]
    else:
        layer = m["layer"].map(dag.layer_of_feature)
    out = pd.DataFrame({"kind": m["kind"], "layer": layer.where(layer.notna(), None), "ideal": m["ideal"],
                        "tol_abs": m["tol_abs"].fillna(0.0), "tol_rel": m["tol_rel"].fillna(0.0),
                        "count_based": m["count_based"].astype(bool)})
    out.index = m[name]
    return out


def load_queries(path) -> pd.DataFrame:
    return pd.read_csv(path).set_index("run")


# ---------------------------------------------------------------------------------------------------- run sets
def _benign(runmeta, exclude, seeds):
    b = runmeta[(runmeta["split"] == "benign") & (~runmeta["fault"].isin(exclude))]
    if seeds is not None:
        b = b[b["seed"].isin(seeds)]
    return list(b.index)


def calibration_runs(runmeta, exclude=BENIGN_EXCLUDE, seeds=CALIBRATION_SEEDS) -> list:
    """Benign runs that set thresholds: calibration seeds, minus the excluded fault families."""
    return _benign(runmeta, exclude, seeds)


def validation_runs(runmeta, exclude=BENIGN_EXCLUDE, seeds=VALIDATION_SEEDS) -> list:
    """Benign runs never used for thresholds: held-out false-alarm check."""
    return _benign(runmeta, exclude, seeds)


def scored_runs(runmeta: pd.DataFrame, splits) -> list:
    sel = runmeta[runmeta["split"].isin(splits) & runmeta["kind"].isin(["sensor", "config"])]
    return list(sel.index)


# ---------------------------------------------------------------------------------------------------- masking bands
def mask_bands(runmeta: pd.DataFrame, queries: pd.DataFrame) -> tuple:
    """(pooled band over all benign runs, {fault family: band of that family's benign runs})."""
    ben = runmeta[runmeta["split"] == "benign"]
    if ben.empty:
        raise ValueError("no benign runs: cannot define masked faults")
    pooled = Q.benign_band(queries.loc[ben.index, "failure_rate"])
    per_family = {f: Q.benign_band(queries.loc[g.index, "failure_rate"]) for f, g in ben.groupby("fault")}
    return pooled, per_family


# ---------------------------------------------------------------------------------------------------- false alarms
def _deviating_layers(cal, table, run, cfg):
    s = cal.strengths(table.loc[[run]]).iloc[0]
    scores = D.layer_scores(s, C.usable_features(cal, cfg.mode, cfg.exclude_count_based), cfg.k)
    return [l for l in dag.LAYERS if scores[l] >= 1.0]


def _fa_summary(devs: list) -> dict:
    return {"rate": float(np.mean([bool(d) for d in devs])) if devs else float("nan"), "n": len(devs),
            "per_layer": {l: float(np.mean([l in d for d in devs])) if devs else float("nan") for l in dag.LAYERS}}


def validation_false_alarm(table, meta, calib, valid, cfg: D.ScoreConfig) -> dict:
    """Fit on `calib`; false alarm = any layer deviates on a held-out benign run in `valid`."""
    if not valid:
        return {"rate": float("nan"), "n": 0, "per_layer": {l: float("nan") for l in dag.LAYERS}}
    cal = C.fit(table, meta, calib, REFERENCE, cfg.margin)
    return _fa_summary([_deviating_layers(cal, table, r, cfg) for r in valid])


def loo_false_alarm(table, meta, calib, cfg: D.ScoreConfig, by="run", runmeta=None) -> dict:
    """Leave-one-out inside the calibration set (by run, or by whole fault family): a weaker, in-sample-ish check."""
    calib, devs = list(calib), []
    for b in calib:
        held = [x for x in calib if runmeta.at[x, "fault"] == runmeta.at[b, "fault"]] if by == "family" else [b]
        rest = [x for x in calib if x not in held]
        if not rest:
            continue
        devs.append(_deviating_layers(C.fit(table, meta, rest, REFERENCE, cfg.margin), table, b, cfg))
    return _fa_summary(devs)


# ---------------------------------------------------------------------------------------------------- records
def build_records(table, runmeta, meta, queries, cfgs: dict, splits, exclude=BENIGN_EXCLUDE, dag_name="dataflow",
                  calib_seeds=CALIBRATION_SEEDS, final=False) -> tuple:
    """-> (records, {margin: calibration}, (pooled band, per-family bands)). cfgs: {'rb': ScoreConfig, 'rf': ScoreConfig}.
    Refuses the held-out 'test' split unless final=True."""
    if "test" in splits and not final:
        raise R.SplitViolation("scoring the 'test' split needs final=True")
    calib = calibration_runs(runmeta, exclude, calib_seeds)
    cals = {}
    for cfg in cfgs.values():
        if cfg.margin not in cals:
            cals[cfg.margin] = C.fit(table, meta, calib, REFERENCE, cfg.margin)
    strengths = {m: cal.strengths(table) for m, cal in cals.items()}
    pooled, fam = mask_bands(runmeta, queries)
    mode_cols = [c for c in queries.columns if c.startswith("n_") and c[2:] in D.SYMPTOM_LAYER]
    records = []
    for run in scored_runs(runmeta, splits):
        rm, qr = runmeta.loc[run], queries.loc[run]
        rate = float(qr["failure_rate"])
        base = dict(run=run, fault=rm["fault"], severity=float(rm["severity"]),
                    seed=None if pd.isna(rm["seed"]) else int(rm["seed"]), split=rm["split"], kind=rm["kind"],
                    accept=list(dag.accept_set(rm["fault"])), failure_rate=rate,
                    masked=bool(Q.is_masked(rate, fam.get(rm["fault"], pooled))), masked_pooled=bool(Q.is_masked(rate, pooled)))
        diags = {}
        for cfg in cfgs.values():
            for name, d in D.diagnose_run(strengths[cfg.margin].loc[run], cals[cfg.margin], cfg, dag_name).items():
                diags[name] = (cfg.label(), d)
        diags["symptom"] = ("-", D.symptom_rule({c[2:]: qr[c] for c in mode_cols}))
        for name, (label, d) in diags.items():
            records.append(dict(base, method=name, config=label, blame=list(d.blame), scores=d.scores,
                                deviating=list(d.deviating), nan_counts=d.nan_counts))
    return records, cals, (pooled, fam)


# ---------------------------------------------------------------------------------------------------- selection
def select(table, runmeta, meta, queries, exclude=BENIGN_EXCLUDE, calib_seeds=CALIBRATION_SEEDS,
           valid_seeds=VALIDATION_SEEDS) -> tuple:
    """Grid over ScoreConfig on the development runs. Rule per mode: among configs whose held-out false-alarm rate is
    <= MAX_FALSE_ALARM take the best dev macro accuracy of the DAG rule; if none qualifies take the lowest false-alarm
    rate first. Ties go to the larger margin, the larger k, then count-based features excluded. Returns (grid, chosen)."""
    calib, valid = calibration_runs(runmeta, exclude, calib_seeds), validation_runs(runmeta, exclude, valid_seeds)
    rows = []
    for cfg in GRID:
        recs, _, _ = build_records(table, runmeta, meta, queries, {cfg.mode: cfg}, ("dev",), exclude, calib_seeds=calib_seeds)
        acc, naive = M.accuracy(recs, f"dag_{cfg.mode}"), M.accuracy(recs, f"naive_{cfg.mode}")
        fa = validation_false_alarm(table, meta, calib, valid, cfg)
        rows.append(dict(config=cfg.label(), mode=cfg.mode, k=cfg.k, exclude_count_based=cfg.exclude_count_based, margin=cfg.margin,
                         dag_macro=acc.get("macro", {}).get("acc", float("nan")),
                         naive_macro=naive.get("macro", {}).get("acc", float("nan")),
                         n_scored=acc.get("macro", {}).get("n", 0), valid_false_alarm=fa["rate"], n_valid=fa["n"]))
    grid = pd.DataFrame(rows)
    return grid, choose_configs(grid)


def choose_configs(grid: pd.DataFrame) -> dict:
    """The selection rule of select(), on a grid with columns mode, k, exclude_count_based, margin, dag_macro,
    valid_false_alarm (NaN counts as 1)."""
    order = ["dag_macro", "margin", "k", "exclude_count_based"]
    chosen = {}
    for mode in ("rb", "rf"):
        g = grid[grid["mode"] == mode].copy()
        fa = g["valid_false_alarm"].fillna(1.0)
        ok = g[fa <= MAX_FALSE_ALARM + 1e-12]
        cap_met = bool(len(ok))
        pool = ok if cap_met else g[fa <= fa.min() + 1e-12]  # nothing qualifies: all the least-alarming configs stay in the tie-break
        best = pool.sort_values(order, ascending=[False, False, False, False]).iloc[0]
        chosen[mode] = dict(config=vars(D.ScoreConfig(mode, int(best["k"]), bool(best["exclude_count_based"]), float(best["margin"]))),
                            cap_met=cap_met, valid_false_alarm=float(best["valid_false_alarm"]), dev_macro=float(best["dag_macro"]))
    return chosen


def load_chosen(path) -> tuple:
    raw = json.loads(Path(path).read_text())
    return {m: D.ScoreConfig(**v["config"]) for m, v in raw.items()}, raw


# ---------------------------------------------------------------------------------------------------- report
def block_summary(records, dev_records) -> dict:
    """All tables for one set of runs (all sensor faults, or all config faults)."""
    methods = sorted({r["method"] for r in records})
    out = {"methods": methods, "n_runs": len({r["run"] for r in records}),
           "n_masked": len({r["run"] for r in records if r["masked"]}),
           "n_masked_pooled_band": len({r["run"] for r in records if r["masked_pooled"]})}
    out["accuracy_by_fault"] = {m: M.accuracy(records, m) for m in methods}
    out["accuracy_all_runs_by_fault"] = {m: M.accuracy(records, m, include_masked=True) for m in methods}
    out["accuracy_strict_by_fault"] = {m: M.accuracy(records, m, strict=True) for m in methods}
    out["accuracy_by_severity"] = {m: M.accuracy(records, m, by="severity") for m in methods}
    out["accuracy_by_seed"] = {m: M.accuracy(records, m, by=lambda r: f"seed {r['seed']}") for m in methods}
    pooled_mask = [dict(r, masked=r["masked_pooled"]) for r in records]
    out["accuracy_pooled_band_by_fault"] = {m: M.accuracy(pooled_mask, m) for m in methods}
    out["confusion"] = {m: M.confusion(records, m) for m in methods}
    out["masked_table"] = {f"{f}@{s}": v for (f, s), v in M.masked_table(records).items()}
    out["propagation"] = {m: M.propagation_table(records, m) for m in methods if m.startswith(("dag_", "naive_"))}
    out["named_member"] = {m: M.named_member(records, m) for m in methods if m.startswith(("dag_", "naive_"))}
    out["random_expected"] = M.expected_random(records)
    base = [r for r in records if r["method"] == methods[0]] if methods else []
    best = M.fit_prior(dev_records) if dev_records else None
    out["prior_layer_dev"] = best
    constants = {f"const_{l}": M.prior_records(base, l, f"const_{l}") for l in dag.LAYERS}
    if best:
        constants["prior_dev"] = M.prior_records(base, best, "prior_dev")
    out["constants"] = {name: M.accuracy(recs, name) for name, recs in constants.items()}
    out["permutation_null"] = {m: M.permutation_null(records, m) for m in methods if m.startswith(("dag_", "naive_"))}
    return out


def summarize(records, dev_records) -> dict:
    return {kind: block_summary([r for r in records if r["kind"] == kind], [r for r in dev_records if r["kind"] == kind])
            for kind in ("sensor", "config") if any(r["kind"] == kind for r in records)}


def _cell(a) -> str:
    if not a:
        return "-"
    if a["n"] < M.MIN_CI_N:
        return f"{a['acc']:.2f} (n={a['n']})"
    return f"{a['acc']:.2f} [{a['lo']:.2f},{a['hi']:.2f}] n={a['n']}"


def _table(acc_by_method: dict, faults: list, methods=None) -> list:
    lines = ["| method | " + " | ".join(faults) + " | macro |", "|---|" + "---|" * (len(faults) + 1)]
    for m, acc in acc_by_method.items():
        if methods is not None and m not in methods:
            continue
        lines.append(f"| {m} | " + " | ".join(_cell(acc.get(f)) for f in faults + ["macro"]) + " |")
    return lines


def markdown(summary: dict, title: str, extra: dict) -> str:
    lines = [f"# {title}", ""]
    for key in ("chosen", "false_alarm", "cap"):
        if key in extra:
            lines.append(f"- {key}: {extra[key]}")
    lines += ["", "Intervals are shown only for n >= 5 and treat runs as independent, which they are not (severities of one seed share draws).", ""]
    for kind, b in summary.items():
        faults = sorted({f for t in (b["accuracy_by_fault"], b["accuracy_all_runs_by_fault"]) for v in t.values() for f in v if f != "macro"})
        lines += [f"## {kind} faults: {b['n_runs']} runs, {b['n_masked']} masked (own-family band), {b['n_masked_pooled_band']} masked (pooled band)", "",
                  "### Top-1 accuracy on runs with failed queries (lenient: blamed layer in the accept set)", ""]
        lines += _table(b["accuracy_by_fault"], faults)
        lines += ["", "Constant baselines (always blame one layer) and chance:", ""]
        lines += _table(b["constants"], faults)
        pn = b["permutation_null"]
        lines += ["", "Label-permutation null of the macro (fault labels shuffled over the scored runs; real vs null mean [2.5 %, 97.5 %], one-sided p):", ""]
        for m, v in pn.items():
            lines.append(f"- {m}: {v['real']:.2f} vs {v['null_mean']:.2f} [{v['null_lo']:.2f}, {v['null_hi']:.2f}], p={v['p_ge']:.3f}, n={v['n']}")
        lines += ["", f"Uniform random top-1 over the four layers (exact): {b['random_expected']:.2f}", "",
                  "### Strict accuracy (the blame SET equals the accept set)", ""]
        lines += _table(b["accuracy_strict_by_fault"], faults, [m for m in b["accuracy_strict_by_fault"] if m.startswith(("dag_", "naive_"))])
        seeds = sorted({s for v in b["accuracy_by_seed"].values() for s in v if s != "macro"})
        if len(seeds) > 1:
            lines += ["", "### By seed (seed 1 shares injector draws with benign seed 1; seed 2 has no benign counterpart)", ""]
            lines += _table(b["accuracy_by_seed"], seeds, [m for m in b["accuracy_by_seed"] if m.startswith(("dag_", "naive_"))])
        lines += ["", "### Which layer is named when the accept set has two members", ""]
        for m, v in b["named_member"].items():
            lines.append(f"- {m}: {json.dumps(v)}")
        lines += ["", "### All fault runs, also those whose queries still succeed (attribution of the fault itself)", ""]
        lines += _table(b["accuracy_all_runs_by_fault"], faults)
        lines += ["", "### Same as the first table with the pooled masking band (sensitivity)", ""]
        lines += _table(b["accuracy_pooled_band_by_fault"], faults, [m for m in b["accuracy_pooled_band_by_fault"] if m.startswith(("dag_", "naive_"))])
        lines += ["", "### Masked faults (failure rate inside the benign band of the same fault family)", "",
                  "| fault@severity | runs | mean failure rate | masked |", "|---|---|---|---|"]
        for k, v in b["masked_table"].items():
            lines.append(f"| {k} | {v['n']} | {v['failure_rate']:.3f} | {v['masked']:.2f} |")
        lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------- provenance
def _sha256(path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance(args, recs, raw_chosen) -> dict:
    """Everything needed to say which code and data produced a report (there is no commit to point at)."""
    import platform
    import time

    src = Path(__file__).resolve().parents[1]
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": sys.argv[1:], "splits": args.splits.split(","), "final": bool(args.final),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "inputs": {k: {"path": str(getattr(args, k)), "sha256": _sha256(getattr(args, k))} for k in ("features", "meta", "queries")},
        "chosen_config": raw_chosen, "chosen_config_file_sha256": _sha256(args.out / "chosen_config.json"),
        "benign_exclude": args.benign_exclude, "calibration_seeds": list(CALIBRATION_SEEDS),
        "validation_seeds": list(VALIDATION_SEEDS), "dag": args.dag, "max_false_alarm": MAX_FALSE_ALARM,
        "source_sha256": {str(p.relative_to(src)): _sha256(p) for p in sorted(src.rglob("*.py")) if "__pycache__" not in p.parts},
        "scored_runs": sorted({r["run"] for r in recs}),
    }


# ---------------------------------------------------------------------------------------------------- cli
def _inputs(args):
    table, runmeta = load_feature_table(args.features)
    return table, runmeta, load_feature_meta(args.meta), load_queries(args.queries)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("select", "report"))
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", default="dev")
    ap.add_argument("--final", action="store_true", help="allow scoring the held-out 'test' split")
    ap.add_argument("--benign-exclude", default=",".join(BENIGN_EXCLUDE))
    ap.add_argument("--dag", default="dataflow", choices=sorted(dag.DAGS))
    ap.add_argument("--calib-seeds", default=",".join(map(str, CALIBRATION_SEEDS)), help="'all' = every benign seed")
    args = ap.parse_args(argv)
    splits = args.splits.split(",")
    if "test" in splits and not args.final:
        raise R.SplitViolation("the 'test' split needs --final")
    exclude = tuple(x for x in args.benign_exclude.split(",") if x)
    seeds = None if args.calib_seeds == "all" else tuple(int(x) for x in args.calib_seeds.split(","))
    args.out.mkdir(parents=True, exist_ok=True)
    table, runmeta, meta, queries = _inputs(args)

    if args.cmd == "select":
        grid, chosen = select(table, runmeta, meta, queries, exclude, seeds)
        grid.to_csv(args.out / "grid.csv", index=False)
        (args.out / "chosen_config.json").write_text(json.dumps(chosen, indent=1))
        print(grid.to_string(index=False))
        print("chosen:", {m: (D.ScoreConfig(**v["config"]).label(), "cap met" if v["cap_met"] else "CAP NOT MET") for m, v in chosen.items()})
        return 0

    cfgs, raw = load_chosen(args.out / "chosen_config.json")
    recs, cals, _ = build_records(table, runmeta, meta, queries, cfgs, splits, exclude, args.dag, seeds, args.final)
    dev_recs, _, _ = build_records(table, runmeta, meta, queries, cfgs, ("dev",), exclude, args.dag, seeds, False)
    calib, valid = calibration_runs(runmeta, exclude, seeds), validation_runs(runmeta, exclude)
    fa = {m: {"validation": validation_false_alarm(table, meta, calib, valid, c),
              "loo_run": loo_false_alarm(table, meta, calib, c, "run", runmeta),
              "loo_family": loo_false_alarm(table, meta, calib, c, "family", runmeta)} for m, c in cfgs.items()}
    summ = summarize(recs, dev_recs)
    pooled, fam = mask_bands(runmeta, queries)
    summ_all = {"blocks": summ, "false_alarm": fa, "chosen": raw,
                "band": {"pooled": {"n": pooled.n_runs, "hi": pooled.hi_rate},
                         "per_family": {f: {"n": b.n_runs, "hi": b.hi_rate} for f, b in fam.items()}},
                "calibration_runs": calib, "validation_runs": valid}
    tag = "_".join(splits)
    (args.out / f"provenance_{tag}.json").write_text(json.dumps(provenance(args, recs, raw), indent=1))
    (args.out / f"summary_{tag}.json").write_text(json.dumps(summ_all, indent=1, default=str))
    extra = {"chosen": {m: c.label() for m, c in cfgs.items()},
             "false_alarm": {m: f"held-out benign runs {v['validation']['rate']:.2f} (n={v['validation']['n']}), LOO by run {v['loo_run']['rate']:.2f}, LOO by family {v['loo_family']['rate']:.2f}"
                             for m, v in fa.items()},
             "cap": {m: ("met" if raw[m]["cap_met"] else f"NOT MET (target {MAX_FALSE_ALARM:.2f})") for m in raw}}
    (args.out / f"report_{tag}.md").write_text(markdown(summ, f"Attribution report ({tag})", extra))
    pd.DataFrame([dict(r, accept="+".join(r["accept"]), blame="+".join(r["blame"]), deviating="+".join(r["deviating"]),
                       scores=json.dumps(r["scores"]), nan_counts=json.dumps(r["nan_counts"])) for r in recs]
                 ).to_csv(args.out / f"predictions_{tag}.csv", index=False)
    print((args.out / f"report_{tag}.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
