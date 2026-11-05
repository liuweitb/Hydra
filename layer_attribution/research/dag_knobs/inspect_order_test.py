"""Building -> room edges and building position vs room centroid in the two functor-order test runs.

usage: python inspect_order_test.py RUNS_ROOT OUT_JSON
"""
import json
import sys

import numpy as np
import spark_dsg as dsg


def info(path):
    G = dsg.DynamicSceneGraph.load(path)
    rooms = list(G.get_layer("ROOMS").nodes)
    b = [n for n in G.nodes if n.layer.layer == 5]
    out = {"n_rooms": len(rooms), "n_buildings": len(b),
           "room_num_parents": [len(list(r.parents())) for r in rooms]}
    if b:
        out["building_num_children"] = len(list(b[0].children()))
        if rooms:
            rc = np.mean([np.asarray(r.attributes.position, dtype=float) for r in rooms], axis=0)
            out["abs_diff_building_vs_room_centroid_m"] = float(np.abs(np.asarray(b[0].attributes.position) - rc).max())
    return out


res = {}
for name in ("tmp_dk_ordbase", "tmp_dk_ordtest"):
    res[name] = info(f"{sys.argv[1]}/{name}/backend/dsg.json")
json.dump(res, open(sys.argv[2], "w"), indent=1)
print(json.dumps(res, indent=1))
