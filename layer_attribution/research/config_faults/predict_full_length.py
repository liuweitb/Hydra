#!/usr/bin/env python3
"""Predicted outcome of every recommended fault at its launch severities on the FULL-length clean run (uh2_clean).

objects_*  : object emulator (clustering of the clean final mesh), checked exactly against 4 real truncated runs
rooms_min_size : exact, the clean rooms with >= T places stay (nothing else changes)
rooms_*    : room-finder + query emulator on the clean places layer (n_rooms and threshold matched Hydra in every check)
places_*   : no emulator, see the truncated real runs
Windows python.

usage: predict_full_length.py OUT_JSON
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import emulate_queries as eq  # noqa: E402
from layer_attribution.pipeline import config_faults as cf  # noqa: E402

R = pathlib.Path("D:/hydra_work/results/config_faults")


def main():
    scene = eq.load_scene(R / "scene_uh2_clean.json")
    obj = json.loads((R / "emulate_objects_ladders_full_clean.json").read_text())["by_tol"]
    sizes = {rid: r["n_children"] for rid, r in scene["rooms"].items()}
    room_of_obj = []
    for o in scene["objects"]:
        pl = scene["places"].get(o["parent"]) if o["parent"] is not None else None
        if pl is not None and pl["parent"] in sizes:
            room_of_obj.append(pl["parent"])
    out = {"reference_full_clean": {"objects": len(scene["objects"]), "places": len(scene["places"]), "rooms": len(sizes),
                                    "room_sizes": sorted(sizes.values(), reverse=True), "queries": len(room_of_obj)},
           "runs": {}}
    for row in cf.launch_plan():
        name, fault, sev = row["name"], row["fault"], row["severity"]
        (key, value), = row["values"].items()
        pred = {"fault": fault, "severity": sev, "key": key, "value": value, "layer": row["layer"]}
        if fault == "objects_min_size":
            e = obj[f"tol0.25_min{value}"]
            pred.update(source="object emulator", objects_expected=e["n_clusters"], objects_clean=len(scene["objects"]),
                        objects_found_fraction=e["real_objects_found_one_to_one"], by_label=e["by_label"])
        elif fault == "objects_tol_merge":
            e = obj[f"tol{float(value)}_min40"]
            pred.update(source="object emulator", objects_expected=e["n_clusters"], objects_clean=len(scene["objects"]),
                        objects_found_fraction=e["real_objects_found_one_to_one"], by_label=e["by_label"])
        elif fault == "rooms_min_size":
            kept = [r for r in sizes if sizes[r] >= value]
            ok = sum(1 for r in room_of_obj if r in kept)
            pred.update(source="exact from the real clean rooms", rooms_expected=len(kept),
                        room_sizes=sorted((sizes[r] for r in kept), reverse=True), query_success=round(ok / len(room_of_obj), 3),
                        buildings_expected=1 if kept else 0)
        elif fault.startswith("rooms_"):
            r = eq.emulate(scene, {key.rsplit(".", 1)[1]: value})
            pred.update(source="room-finder + query emulator", rooms_expected=r["n_rooms"], room_sizes=r["sizes"], threshold=r["threshold"],
                        query_success=r["query_success"], rooms_matched=r["rooms_matched"],
                        note="query_success from the emulator carries about 0.03 border noise (default config gives 0.99 vs real 1.00)")
        else:
            pred.update(source="none", note="no emulator for the places layer; see truncated real runs")
        out["runs"][name] = pred
        print(name, json.dumps({k: v for k, v in pred.items() if k not in ("by_label",)}))
    pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
