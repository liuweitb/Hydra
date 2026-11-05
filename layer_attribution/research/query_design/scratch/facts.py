import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, "D:/Hydra/layer_attribution/src"); sys.path.insert(0, "D:/Hydra/layer_attribution/research/query_design")
import measure as M
from layer_attribution import queries as Q
views, info = M.load_all()
ref = views["uh2_clean"]; qs = Q.build_queries(ref, M.CFG)
rows = []
for n, v in views.items():
    if n == "uh2_clean" or info[n]["role"] != "benign": continue
    m = Q.match_rooms(ref, v, M.CFG)
    s = m.structures()
    r2, r3 = set(m.rep_rooms(2)), set(m.rep_rooms(3))
    qin = sum(1 for q in qs if s[q.ref_room_row] != "one_to_one")
    rows.append((n, v.n_room, bool(r2 & r3), all(x == "one_to_one" for x in s), qin))
df = pd.DataFrame(rows, columns=["run", "run_rooms", "R2_R3_share_a_run_room", "partition_identical_to_reference", "queries_in_changed_rooms"])
print(df.to_string())
print("benign runs:", len(df), " R2+R3 merged in", int(df.R2_R3_share_a_run_room.sum()), " partition identical in", int(df.partition_identical_to_reference.sum()))
print("geometry benign runs (8): partition identical in", int(df[~df.run.str.contains("label")].partition_identical_to_reference.sum()))
# main runs: partitions
rows = []
for n, v in views.items():
    if n == "uh2_clean" or info[n]["role"] != "main": continue
    m = Q.match_rooms(ref, v, M.CFG)
    s = m.structures()
    rows.append((n, v.n_room, "".join({"one_to_one": "1", "merged": "M", "split": "S", "merged_split": "X", "missing": "-"}[x] for x in s)))
print(pd.DataFrame(rows, columns=["run", "run_rooms", "structure_R0..R7 (1=one_to_one M=merged S=split X=both -=missing)"]).to_string())
