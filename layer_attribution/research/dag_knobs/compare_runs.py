"""Compare saved scene graphs of truncated runs: per-layer node counts and position sets.

usage: compare_runs.py OUT.json RUN_A RUN_B ...   (run dirs; reads <run>/backend/dsg.json and frontend/dsg.json)
"""
import collections
import json
import sys

import numpy as np
import spark_dsg as dsg


def key_str(k):
    return f"{k.layer}" if k.partition == 0 else f"{k.layer}[{k.partition}]"


def summarize(path):
    G = dsg.DynamicSceneGraph.load(path)
    layers = collections.defaultdict(list)
    for n in G.nodes:
        layers[key_str(n.layer)].append((str(n.id), np.asarray(n.attributes.position, dtype=float)))
    out = {}
    for k, items in sorted(layers.items()):
        pos = np.array([p for _, p in items]) if items else np.zeros((0, 3))
        out[k] = {
            "n": len(items),
            "ids": sorted(i for i, _ in items),
            "pos_sorted": pos[np.lexsort(pos.T[::-1])] if len(pos) else pos,
        }
    return out


def main():
    out_path, runs = sys.argv[1], sys.argv[2:]
    res = {}
    for stage in ("frontend", "backend"):
        S = {r.rstrip("/").split("/")[-1]: summarize(f"{r}/{stage}/dsg.json") for r in runs}
        names = list(S)
        res[stage] = {"counts": {n: {k: v["n"] for k, v in S[n].items()} for n in names}, "pairwise": {}}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = S[names[i]], S[names[j]]
                row = {}
                for k in sorted(set(a) | set(b)):
                    if k not in a or k not in b:
                        row[k] = "layer missing in one run"
                        continue
                    same_ids = a[k]["ids"] == b[k]["ids"]
                    if a[k]["n"] == b[k]["n"] and a[k]["n"] > 0:
                        d = float(np.abs(a[k]["pos_sorted"] - b[k]["pos_sorted"]).max())
                    else:
                        d = None
                    row[k] = {"n_a": a[k]["n"], "n_b": b[k]["n"], "same_ids": same_ids, "max_abs_pos_diff_sorted": d}
                res[stage]["pairwise"][f"{names[i]} vs {names[j]}"] = row
    json.dump(res, open(out_path, "w"), indent=1, sort_keys=True)
    print(json.dumps(res, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
