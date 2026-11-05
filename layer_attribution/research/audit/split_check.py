import re, collections, pathlib, sys
sys.path.insert(0, "D:/Hydra/layer_attribution/src")
from layer_attribution import runs as R
import pandas as pd
names = [l.strip() for l in open("D:/Hydra/layer_attribution/research/audit/run_names.txt") if l.strip()]
SENS = ["label_confusion","label_dropout","depth_noise","depth_holes","pose_drift","frame_drop"]
rows = []
for n in names:
    if n.startswith("tmp_"): rows.append((n, "tmp", None, None, None, None)); continue
    if n in ("uh2_clean","uh2_clean_b"):
        rows.append((n, R.assign_split("reference" if n=="uh2_clean" else "repeat", 0.0, None), "-", 0.0, None, "ref")); continue
    m = re.fullmatch(r"uh2_cfg_(\w+)_s(\d+)", n)
    if m:
        fault, sev = m.group(1), int(m.group(2))/100
        rows.append((n, R.assign_split("config", sev, None, fault), fault, sev, None, "config")); continue
    m = re.fullmatch(r"uh2_(\w+?)_s(\d+)_r(\d+)", n)
    if m:
        fault, sev, seed = m.group(1), int(m.group(2))/100, int(m.group(3))
        rows.append((n, R.assign_split("sensor", sev, seed, fault), fault, sev, seed, "sensor")); continue
    m = re.fullmatch(r"uh2_(\w+?)_s(\d+)", n)   # legacy: seed 0
    if m:
        fault, sev = m.group(1), int(m.group(2))/100
        rows.append((n, R.assign_split("sensor", sev, 0, fault), fault, sev, 0, "legacy")); continue
    rows.append((n, "UNPARSED", None, None, None, None))
df = pd.DataFrame(rows, columns=["name","split","fault","sev","seed","kind"])
print(df.split.value_counts().to_dict())
print(df[df.split.isin(["UNPARSED","tmp"])].to_string())
# expected by hand
def expect(r):
    if r.kind=="ref": return "reference" if r["name"]=="uh2_clean" else "repeat"
    if r.sev <= 0.05: return "benign"
    if r.kind=="config": return "dev" if abs(r.sev - {"objects_min_size":0.6,"objects_tol_merge":0.85,"rooms_min_size":0.8,"rooms_max_dilation":0.3}.get(r.fault,-1))<1e-6 else "test"
    return "dev" if r.seed==0 else "test"
bad = [(r["name"], r.split, expect(r)) for _, r in df.iterrows() if r.kind and r.split not in ("tmp","UNPARSED") and r.split != expect(r)]
print("rule mismatches:", bad)
# the benign seeds
print(df[df.split=="benign"].groupby(["fault","seed"]).size().unstack(fill_value=0).to_string())
# config severities present
print(df[df.kind=="config"].sort_values(["fault","sev"])[["name","split"]].to_string())
# the dev/benign/reference set implied by names vs the features.csv rows
feat = pd.read_csv("D:/hydra_work/results/tables/dev/features.csv")
implied = set(df[df.split.isin(["reference","repeat","benign","dev"])].name)
print("in features.csv:", len(feat), " name-implied dev-set:", len(implied), " features.csv - implied:", sorted(set(feat.run)-implied), " implied - features.csv:", sorted(implied-set(feat.run)))
print("test-named runs (count only):", int((df.split=='test').sum()))
