import json, numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
T = "D:/hydra_work/results/tables/dev/"
f = pd.read_csv(T + "features.csv"); m = pd.read_csv(T + "features_meta.csv").set_index("name")
err = pd.read_csv(T + "features_errors.csv")
print("errors file rows:", len(err)); print(err.to_string()[:1500])
X = f.drop(columns=["split","kind","fault","severity","seed"]).set_index("run").astype(float)
rm = f[["run","split","kind","fault","severity","seed"]].set_index("run")
nanc = X.isna()
print("\nNaN cells total:", int(nanc.values.sum()), "of", X.size)
cols = nanc.columns[nanc.any()]
for c in cols:
    rows = list(nanc.index[nanc[c]])
    print(f"  {c:34s} ev={str(m.at[c,'evidence_layer']):8s} kind={m.at[c,'kind'][10:]:5s} NaN in {len(rows)}: {rows[:6]}")
S = pd.read_csv("D:/Hydra/layer_attribution/research/audit/recompute_out/my_strengths.csv", index_col=0)
th = pd.read_csv("D:/Hydra/layer_attribution/research/audit/recompute_out/my_thresholds.csv", index_col=0)
print("\nstrength cells at cap 1000:", int((S >= 999.999).values.sum()))
cap = (S >= 999.999)
for r in cap.index[cap.any(axis=1)]:
    print("  ", r, list(cap.columns[cap.loc[r]])[:6], "n=", int(cap.loc[r].sum()))
# thresholds: how many are floor-dominated vs benign-dominated
ev = th["ev"].notna()
floor = m["tol_abs"].fillna(0) + m["tol_rel"].fillna(0) * th["center"].abs()
bm = 1.5 * th["benign_max_dev"]
dom = pd.Series(np.where(floor >= bm, "floor", "benign"), index=th.index)
print("\nthreshold set by floor / by 1.5*benign max (layer-evidence features):", dom[ev].value_counts().to_dict())
print("features with benign_max_dev == 0 (constant on the 10 benign runs):", list(th.index[(th['benign_max_dev'] == 0) & ev]))
tiny = th[(th["thr"] < 1e-6) & ev]
print("features with threshold < 1e-6:", list(tiny.index))
# usable features per layer / mode
for mode, kind in (("rb","reference_based"),("rf","reference_free")):
    u = th[(th.kind == kind) & ev]
    print(mode, u.groupby("ev").size().to_dict(), " count_based:", int(u.count_based.sum()))
# any reference_based feature whose ideal is NaN / feature with nan center
print("center NaN:", list(th.index[th['center'].isna()]))
