"""Guard / invariance tests on synthetic inputs (no real test-run data touched).
1) test-split guard in runs.select, features_extract, query_table, evaluate; defaults contain no 'test'.
2) calibration.fit invariance to non-benign rows; evaluate.select/report invariance to an injected fake 'test' row.
"""
import json, pathlib, shutil, sys, tempfile, io, contextlib
import numpy as np, pandas as pd
from layer_attribution import runs as R, calibration as C, features_extract as FX
from layer_attribution import diagnosis as D
from layer_attribution.evaluation import evaluate as EV, metrics as M

res = {}
tmp = pathlib.Path(tempfile.mkdtemp(prefix="guard_"))
def mk(name, manifest=None):
    d = tmp / name / "backend"; d.mkdir(parents=True); (d / "dsg_with_mesh.json").write_text("{}")
    if manifest: (tmp / name / "injection_manifest.json").write_text(json.dumps(manifest))
mk("uh2_clean")
mk("uh2_label_dropout_s50_r1", dict(fault="label_dropout", severity=0.5, seed=1, expected_first_layer="objects"))
mk("uh2_label_dropout_s50", dict(fault="label_dropout", severity=0.5, seed=0, expected_first_layer="objects"))
mk("uh2_label_dropout_s03_r1", dict(fault="label_dropout", severity=0.03, seed=1, expected_first_layer="objects"))
mk("uh2_cfg_rooms_min_size_s60", dict(fault="rooms_min_size", severity=0.6, seed=None, corrupted_input="config"))
idx = R.index_runs(tmp)
res["splits_synthetic"] = {r.name: r.split for r in idx}
# runs.select guard
for sp in (("test",), ("dev", "test"), R.ALL_SPLITS):
    try:
        R.select(idx, sp); res[f"runs.select{sp}"] = "NOT REFUSED"
    except R.SplitViolation: res[f"runs.select{sp}"] = "refused"
res["runs.select default (DEV_SPLITS) names"] = [r.name for r in R.select(idx)]
# features_extract
for args in (["--splits", "test"], ["--splits", "dev,test"], []):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = FX.main(["--root", str(tmp), "--out", str(tmp / "o.csv"), "--min-age-s", "0", "--dry-run"] + args)
    res[f"features_extract {' '.join(args) or '(default)'}"] = f"rc={rc} " + " | ".join(l.strip() for l in buf.getvalue().splitlines()[:6])
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = FX.main(["--root", str(tmp), "--out", str(tmp / "o.csv"), "--min-age-s", "0", "--dry-run", "--splits", "test", "--final"])
res["features_extract --splits test --final"] = f"rc={rc} " + " | ".join(l.strip() for l in buf.getvalue().splitlines()[:4])
# query_table
try:
    from layer_attribution import query_table as QT
    try:
        QT.main(["--root", str(tmp), "--out", str(tmp / "q.csv"), "--splits", "test"]); res["query_table --splits test"] = "NOT REFUSED"
    except R.SplitViolation: res["query_table --splits test"] = "refused (SplitViolation)"
    except Exception as e: res["query_table --splits test"] = f"other exception {type(e).__name__}: {e}"
    import inspect
    res["query_table default splits"] = QT.R.DEV_SPLITS
except Exception as e:
    res["query_table import"] = f"{type(e).__name__}: {e}"
# evaluate
T = pathlib.Path("D:/hydra_work/results/tables/dev"); E = pathlib.Path("D:/hydra_work/results/eval_dev_v1")
argv = ["--features", str(T / "features.csv"), "--meta", str(T / "features_meta.csv"), "--queries", str(T / "queries.csv"), "--out", str(tmp / "ev")]
for sp in ("test", "dev,test"):
    try:
        EV.main(["report", "--splits", sp] + argv); res[f"evaluate report --splits {sp}"] = "NOT REFUSED"
    except R.SplitViolation: res[f"evaluate report --splits {sp}"] = "refused"
import argparse
# invariance: calibration ignores non-benign rows
table, runmeta = EV.load_feature_table(T / "features.csv"); meta = EV.load_feature_meta(T / "features_meta.csv"); queries = EV.load_queries(T / "queries.csv")
ben = EV.benign_runs(runmeta)
cal0 = C.fit(table, meta, ben, EV.REFERENCE)
t2 = table.copy(); other = [r for r in t2.index if r not in ben and r != EV.REFERENCE]
rng = np.random.default_rng(0); t2.loc[other] = rng.normal(size=(len(other), t2.shape[1])) * 1e3
cal1 = C.fit(t2, meta, ben, EV.REFERENCE)
res["calibration invariant to non-benign/non-reference rows (garbage in 24 rows)"] = bool(np.allclose(cal0.threshold, cal1.threshold, equal_nan=True) and np.allclose(cal0.center, cal1.center, equal_nan=True))
t3 = table.copy(); t3.loc[EV.REFERENCE] = rng.normal(size=t3.shape[1]) * 1e3
cal2 = C.fit(t3, meta, ben, EV.REFERENCE)
rb = meta.index[meta.kind == "reference_based"]; rf = meta.index[meta.kind == "reference_free"]
res["reference row garbage: thresholds of reference_based unchanged / reference_free changed"] = (bool(np.allclose(cal0.threshold[rb], cal2.threshold[rb], equal_nan=True)), bool(not np.allclose(cal0.threshold[rf], cal2.threshold[rf], equal_nan=True)))
# fake test row injected into the tables: select / records / prior unchanged
chosen = {m: D.ScoreConfig(**v) for m, v in json.loads((E / "chosen_config.json").read_text()).items()}
g0, c0 = EV.select(table, runmeta, meta, queries)
recs0, _, band0 = EV.build_records(table, runmeta, meta, queries, chosen, ("dev",)); dev0 = recs0
s0 = EV.summarize(recs0, dev0)
fake = "FAKE_TEST_RUN"
tf = table.copy(); tf.loc[fake] = 1e6
rmf = runmeta.copy(); rmf.loc[fake] = dict(split="test", kind="sensor", fault="depth_noise", severity=0.8, seed=1.0)
qf = queries.copy(); qf.loc[fake] = queries.loc["uh2_depth_noise_s80_r0"]; qf.loc[fake, "failure_rate"] = 0.99
g1, c1 = EV.select(tf, rmf, meta, qf)
recs1, _, band1 = EV.build_records(tf, rmf, meta, qf, chosen, ("dev",))
s1 = EV.summarize(recs1, recs1)
res["select(): grid + chosen unchanged by a fake test row"] = bool(g0.equals(g1) and {k: v.label() for k, v in c0.items()} == {k: v.label() for k, v in c1.items()})
res["report records/summary/band/prior unchanged by a fake test row"] = bool(json.dumps(s0, default=str) == json.dumps(s1, default=str) and band0 == band1)
try:
    EV.build_records(tf, rmf, meta, qf, chosen, ("test",)); res["build_records(splits=('test',)) library call"] = "NOT guarded (library level; only the CLI main() checks)"
except Exception as e:
    res["build_records(splits=('test',)) library call"] = f"raised {type(e).__name__}"
shutil.rmtree(tmp, ignore_errors=True)
for k, v in res.items(): print(f"{k}: {v}")
pathlib.Path("D:/Hydra/layer_attribution/research/audit/guard_tests_out.json").write_text(json.dumps(res, indent=1, default=str))
