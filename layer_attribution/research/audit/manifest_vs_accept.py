import json, pathlib, pandas as pd
from layer_attribution import dag
root = pathlib.Path("//wsl.localhost/Ubuntu-24.04/home/hydra/runs/uh2")
f = pd.read_csv("D:/hydra_work/results/tables/dev/features.csv")[["run", "split", "fault"]]
rows = set()
for _, r in f[f.split.isin(["dev"])].iterrows():
    mp = root / r.run / "injection_manifest.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    rows.add((r.fault, m.get("expected_first_layer"), tuple(m.get("also_hits_directly") or ()), tuple(dag.ACCEPT[r.fault])))
for x in sorted(rows): print(x)
