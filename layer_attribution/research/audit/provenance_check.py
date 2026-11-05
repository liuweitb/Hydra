import hashlib, json, pathlib, numpy as np, pandas as pd
from layer_attribution import features as F
E = pathlib.Path("D:/hydra_work/results/eval_dev_v1"); T = pathlib.Path("D:/hydra_work/results/tables/dev")
p = json.load(open(E / "provenance_dev.json"))
sha = lambda path: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
print("input hash mismatches:", [k for k, v in p["inputs"].items() if sha(v["path"]) != v["sha256"]])
print("chosen_config hash mismatch:", sha(E / "chosen_config.json") != p["chosen_config_file_sha256"])
src = pathlib.Path("D:/Hydra/layer_attribution/src/layer_attribution")
cur = {str(q.relative_to(src)).replace("\\", "/"): sha(q) for q in sorted(src.rglob("*.py")) if "__pycache__" not in q.parts}
old = {k.replace("\\", "/"): v for k, v in p["source_sha256"].items()}
print("source files changed since the report:", sorted(k for k in set(cur) | set(old) if cur.get(k) != old.get(k)))
# features_meta.csv vs the live registry
m = pd.read_csv(T / "features_meta.csv").set_index("name"); live = pd.DataFrame(F.meta()).set_index("name")
print("meta rows csv/live:", len(m), len(live), " names equal:", list(m.index) == list(live.index))
bad = []
for c in ("kind", "layer", "evidence_layer", "count_based"):
    a, b = m[c].fillna("-").astype(str), live[c].fillna("-").astype(str)
    bad += [(c, n) for n in m.index if a[n] != b.get(n)]
for c in ("ideal", "tol_abs", "tol_rel"):
    a, b = m[c].astype(float), live[c].astype(float)
    bad += [(c, n) for n in m.index if not (np.isclose(a[n], b[n], equal_nan=True))]
print("meta csv vs live registry differences:", bad)
# feature value cache hashes recorded in the table's cache vs live source hashes (dev runs only)
h = F.source_hashes()
import os
cache = pathlib.Path("//wsl.localhost/Ubuntu-24.04/home/hydra/runs/cache/feature_values")
stale = {}
for run in ("uh2_clean", "uh2_pose_drift_s50", "uh2_label_confusion_s03_r0"):
    d = json.loads((cache / f"{run}.json").read_text())
    stale[run] = sorted(n for n, e in d["values"].items() if h.get(n) != e["h"])
print("cached feature values whose source hash differs from the live code:", stale)
