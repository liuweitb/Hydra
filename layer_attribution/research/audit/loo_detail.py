import json, pathlib, numpy as np, pandas as pd
from layer_attribution import calibration as C, diagnosis as D, dag
from layer_attribution.evaluation import evaluate as EV
T = pathlib.Path("D:/hydra_work/results/tables/dev")
table, runmeta = EV.load_feature_table(T / "features.csv"); meta = EV.load_feature_meta(T / "features_meta.csv")
ben = EV.benign_runs(runmeta)
for label, cfg in (("rb-k1", D.ScoreConfig("rb", 1, False)), ("rf-k2", D.ScoreConfig("rf", 2, False))):
    print(label)
    for by in ("run", "family"):
        rows = []
        for b in ben:
            held = [b] if by == "run" else [x for x in ben if runmeta.at[x, "fault"] == runmeta.at[b, "fault"]]
            rest = [x for x in ben if x not in held]
            cal = C.fit(table, meta, rest, EV.REFERENCE)
            s = cal.strengths(table.loc[[b]]).iloc[0]
            sc = D.layer_scores(s, C.usable_features(cal, cfg.mode, cfg.exclude_count_based), cfg.k)
            dev = [l for l in dag.LAYERS if sc[l] >= 1]
            if dev:
                top = {l: s[C.usable_features(cal, cfg.mode)[l]].idxmax() for l in dev}
                rows.append(f"    {b:30s} dev={'+'.join(dev):28s} scores={ {l: round(sc[l],2) for l in dev} } top_feature={top}")
        print(f"  LOO by {by}: {len(rows)} of {len(ben)} benign runs raise a deviation")
        print("\n".join(rows))
