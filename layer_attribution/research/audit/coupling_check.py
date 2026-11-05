"""Does calibrating on s03 runs that share the evaluation run's seed matter? Dev (seed 0) vs benign seeds.
Uses the real code with modified run tables (benign rows relabelled 'ignore'). Dev runs only."""
import json, pathlib, numpy as np, pandas as pd
from layer_attribution import diagnosis as D
from layer_attribution.evaluation import evaluate as EV
from layer_attribution.evaluation import metrics as M
T = pathlib.Path("D:/hydra_work/results/tables/dev")
table, runmeta = EV.load_feature_table(T / "features.csv")
meta = EV.load_feature_meta(T / "features_meta.csv"); queries = EV.load_queries(T / "queries.csv")
chosen = {m: D.ScoreConfig(**v) for m, v in json.loads(pathlib.Path("D:/hydra_work/results/eval_dev_v1/chosen_config.json").read_text()).items()}
def run(rm, label, cfgs=chosen):
    recs, cal, band = EV.build_records(table, rm, meta, queries, cfgs, ("dev",))
    out = {}
    for m_ in ("dag_rb", "naive_rb", "dag_rf", "naive_rf"):
        a = M.accuracy(recs, m_)
        out[m_] = (round(a["macro"]["acc"], 3), a["macro"]["n"])
    n_dev_dag = sum(1 for r in recs if r["method"] == "dag_rb" and r["deviating"])
    print(f"{label:46s} benign={len(cal.info['benign']):2d} band_hi={band.hi_rate:.4f} masked={sum(1 for r in recs if r['method']=='dag_rb' and r['masked'])} {out} runs_with_any_deviation(rb)={n_dev_dag}")
    return recs, cal
base = run(runmeta, "baseline (seeds 0+1 benign)")
for keep_seed, label in ((1, "decoupled from dev: benign seed 1 only"), (0, "coupled: benign seed 0 only")):
    rm = runmeta.copy()
    b = rm[(rm.split == "benign") & (rm.seed != keep_seed)].index
    rm.loc[b, "split"] = "ignore"
    run(rm, label)
# severity response of one feature across the same-seed family (confirms scaling of the s03 clone)
cols = ["mesh_extra_mean", "obj_count_ratio", "place_chamfer_mean", "mesh_missing_frac_gt10cm", "obj_recall_pos"]
fam = {"pose_drift": ["uh2_pose_drift_s03_r0", "uh2_pose_drift_s20_r0", "uh2_pose_drift_s50", "uh2_pose_drift_s80_r0"],
       "depth_noise": ["uh2_depth_noise_s03_r0", "uh2_depth_noise_s20_r0", "uh2_depth_noise_s50", "uh2_depth_noise_s80_r0"],
       "label_dropout": ["uh2_label_dropout_s03_r0", "uh2_label_dropout_s20_r0", "uh2_label_dropout_s50", "uh2_label_dropout_s80_r0"]}
for f_, runs_ in fam.items():
    print(f_); print(table.loc[runs_, cols].round(4).to_string())
