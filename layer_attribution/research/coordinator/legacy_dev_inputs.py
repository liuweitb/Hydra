"""Convert the signal_probe tables (feature x run) into the formats evaluate.py reads, until features_extract.py exists.

usage: legacy_dev_inputs.py OUT_DIR [RUNS_ROOT]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from layer_attribution import dag
from layer_attribution import runs as R

SP = Path(__file__).resolve().parents[1] / "signal_probe"
out = Path(sys.argv[1])
root = Path(sys.argv[2] if len(sys.argv) > 2 else r"\\wsl.localhost\Ubuntu-24.04\home\hydra\runs\uh2")
out.mkdir(parents=True, exist_ok=True)

vals = pd.read_csv(SP / "feature_values.csv").set_index("feature").T  # run x feature
thr = pd.read_csv(SP / "thresholds.csv").set_index("feature")
index = {r.name: r for r in R.index_runs(root)}

rows = []
for run in vals.index:
    r = index[run]
    rows.append(dict(run=run, split=r.split, kind=r.kind, fault=r.fault if r.fault else "none", severity=r.severity,
                     seed=r.seed if r.seed is not None else np.nan))
feat = pd.concat([pd.DataFrame(rows).set_index("run"), vals], axis=1).rename_axis("run").reset_index()
feat.to_csv(out / "features.csv", index=False)

meta = pd.DataFrame({
    "name": thr.index, "kind": thr["kind"].values, "layer": thr["layer"].values,
    "evidence_layer": [dag.layer_of_feature(t) for t in thr["layer"]],
    "ideal": [c if k == "reference_based" else np.nan for c, k in zip(thr["center"], thr["kind"])],
    "tol_abs": thr["floor"].values, "tol_rel": 0.0, "count_based": thr["count_based"].astype(bool).values})
meta.to_csv(out / "meta.csv", index=False)
print(f"{len(feat)} runs, {vals.shape[1]} features; splits: {feat['split'].value_counts().to_dict()}")
print("evidence layers:", meta["evidence_layer"].value_counts(dropna=False).to_dict())
