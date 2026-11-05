#!/usr/bin/env python3
"""Room-finder emulator over fine parameter grids and over the severity ladder of every rooms fault.

Runs on Windows python (reads cached places graphs, no spark_dsg). Screening only: room sizes can differ by a few places
from Hydra at room borders (tie-breaking of the neighbour flood fill), counts and thresholds are what carries over.

usage: scan_room_params.py CACHE_DIR OUT_JSON
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import emulate_rooms as er  # noqa: E402
from layer_attribution.pipeline import config_faults as cf  # noqa: E402

GRAPHS = {"full_clean": "places_uh2_clean.json", "trunc_ref": "places_tmp_cfg_ref.json"}
GRIDS = {
    "min_room_size": [10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150, 180, 200, 250, 300, 350, 400],
    "min_component_size": [1, 3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200],
    "plateau_ratio": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0],
    "max_dilation_m": [1.4, 1.2, 1.1, 1.05, 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55],
    "min_dilation_m": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.05, 1.1, 1.15],
}


def main():
    cache, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    res = {}
    for gname, fn in GRAPHS.items():
        dist, edges = er.load_cached_places(cache / fn)
        res[gname] = {"n_places": len(dist), "n_edges": len(edges), "default": er.find_rooms(dist, edges), "grids": {}, "ladders": {}}
        for key, vals in GRIDS.items():
            res[gname]["grids"][key] = [{"value": v, **er.find_rooms(dist, edges, **{key: v})} for v in vals]
        for name, f in cf.FAULTS.items():
            if f.layer != "rooms":
                continue
            ladder = []
            for s in [i / 20 for i in range(21)]:
                over = {k.rsplit(".", 1)[1]: v for k, v in cf.values_at(name, s).items()}
                ladder.append({"severity": s, "params": over, **er.find_rooms(dist, edges, **over)})
            res[gname]["ladders"][name] = ladder
    out.write_text(json.dumps(res, indent=1))
    for gname, g in res.items():
        print("=====", gname, g["n_places"], "places; default", g["default"]["n_rooms"], g["default"]["sizes"], g["default"]["threshold"])
        for key, rows in g["grids"].items():
            print(" --", key)
            for r in rows:
                print(f"    {r['value']:>6} -> {r['n_rooms']} rooms {r['sizes']} thr={r['threshold']} comps={r['components']}")


if __name__ == "__main__":
    main()
