#!/usr/bin/env python3
"""Object emulator (clustering of the final labeled mesh) vs the real truncated objects_* runs. Windows python, json only.

usage: compare_objects.py ANALYSIS_JSON OUT_JSON EMU_JSON [EMU_JSON ...]     EMU_JSON = emulate_objects.py outputs on the truncated reference
"""

import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
emu = {}
for f in sys.argv[3:]:
    emu.update(json.loads(pathlib.Path(f).read_text())["by_tol"])
rows = {}
for name, rec in data["runs"].items():
    if not rec.get("fault", "").startswith("objects_") or "metrics" not in rec:
        continue
    (key, p), = rec["params"].items()
    value = p["value"]
    case = f"tol0.25_min{int(value)}" if key.endswith("min_cluster_size") else f"tol{float(value)}_min40"
    m = rec["metrics"]
    row = {"fault": rec["fault"], "severity": rec["severity"], "key": key, "value": value, "emulator_case": case,
           "real": {"n_objects": m["objects_run"], "objects_found_with_label": m["objects_found_with_label"],
                    "query_success": m["query_success"], "query_status": m["query_status"]},
           "emulated": None}
    if case in emu:
        e = emu[case]
        row["emulated"] = {"n_clusters": e["n_clusters"], "objects_found_with_label": e["real_objects_found_one_to_one"]}
        row["abs_err_found"] = round(abs(e["real_objects_found_one_to_one"] - m["objects_found_with_label"]), 3)
        row["abs_err_count"] = abs(e["n_clusters"] - m["objects_run"])
    rows[name] = row
    print(name, value, "| found emu", row["emulated"] and row["emulated"]["objects_found_with_label"], "real", m["objects_found_with_label"],
          "| n emu", row["emulated"] and row["emulated"]["n_clusters"], "real", m["objects_run"], "| query_success real", m["query_success"])
pathlib.Path(sys.argv[2]).write_text(json.dumps(rows, indent=1))
