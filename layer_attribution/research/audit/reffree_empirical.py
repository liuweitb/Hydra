"""Empirical check that reference_free features do not depend on the reference: call each with a poisoned `ref`
(any attribute access raises) and with None, compare with the stored table values.
Run with PYTHONPATH=D:/Hydra/layer_attribution/src and LA_* env vars pointing at the WSL share. Dev / benign / reference runs only.
"""
import math
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from layer_attribution import features as F
from layer_attribution import snapshot as S

RUNS = sys.argv[1:] or ["uh2_label_confusion_s03_r0", "uh2_depth_noise_s50"]
T = pd.read_csv("D:/hydra_work/results/tables/dev/features.csv").set_index("run")


class Poison:
    def __getattr__(self, name):
        raise RuntimeError(f"reference accessed: {name}")


rf = [n for n, s in F.REGISTRY.items() if s.kind == "reference_free"]
print("reference_free features:", len(rf), "snapshot schema", S.SCHEMA)
res = {}
for run in RUNS:
    t0 = time.time()
    snap = S.Snapshot.open(run)
    print(f"{run}: snapshot from_cache={snap.from_cache}", flush=True)
    view = F.FeatureView(snap)
    bad = []
    n_equal_none = n_equal_table = n_nan = 0
    maxrel = 0.0
    for n in rf:
        spec = F.REGISTRY[n]
        vals = {}
        for tag, ref in (("poison", Poison()), ("none", None)):
            try:
                with warnings.catch_warnings(), np.errstate(all="ignore"):
                    warnings.simplefilter("ignore")
                    v = float(spec.fn(view, ref))
                vals[tag] = float("nan") if math.isinf(v) else v
            except Exception as e:
                vals[tag] = ("ERR", f"{type(e).__name__}: {e}")
        pv, nv = vals["poison"], vals["none"]
        tv = float(T.at[run, n])
        if isinstance(pv, tuple):
            bad.append((n, "poison run raised", pv[1][:80]))
            continue
        same_none = (pv == nv) or (math.isnan(pv) and isinstance(nv, float) and math.isnan(nv))
        n_equal_none += bool(same_none)
        if math.isnan(tv) and math.isnan(pv):
            n_equal_table += 1; n_nan += 1
        elif not math.isnan(tv) and not math.isnan(pv):
            rel = abs(pv - tv) / max(abs(tv), 1e-12)
            maxrel = max(maxrel, rel)
            if rel <= 1e-6 or abs(pv - tv) <= 1e-9:
                n_equal_table += 1
            else:
                bad.append((n, "table mismatch", f"poison={pv!r} table={tv!r}"))
        else:
            bad.append((n, "nan mismatch", f"poison={pv!r} table={tv!r}"))
    res[run] = dict(n=len(rf), equal_poison_vs_none=n_equal_none, equal_to_table=n_equal_table, nan_both=n_nan, max_rel_diff=maxrel, bad=bad)
    print(run, res[run], f"{time.time() - t0:.0f}s", flush=True)
    snap.close()
