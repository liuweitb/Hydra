#!/usr/bin/env python3
"""One-screen summary of runs in an analysis json (Windows python, json only).

usage: show_runs.py ANALYSIS_JSON [NAME_SUBSTRING ...]
"""

import json
import pathlib
import sys

d = json.loads(pathlib.Path(sys.argv[1]).read_text())
pats = sys.argv[2:]
for name, r in d["runs"].items():
    if pats and not any(p in name for p in pats):
        continue
    if "error" in r:
        print("=====", name, "ERROR", r["error"])
        continue
    ident = r["identical"]
    nodes = {k: v["nodes"] for k, v in ident.items() if k != "mesh"}
    edges = {k: v["nodes_and_parent_edge"] for k, v in ident.items() if k != "mesh"}
    changed = [k for k, v in nodes.items() if not v]
    edge_only = [k for k in nodes if nodes[k] and not edges[k]]
    m = r["metrics"]
    print("=====", name, {k.split(".")[-1]: v["value"] for k, v in r.get("params", {}).items()})
    print("  effective echo ok:", {k.split(".")[-1]: (v["echoed"], v["ok"]) for k, v in r.get("effective", {}).items()})
    print("  mesh identical:", ident["mesh"], "| node-level changes:", changed or "none", "| parent-edge-only changes:", edge_only or "none")
    print("  counts ref->run:", {k: v for k, v in r["counts"].items()}, "room sizes", r["room_sizes"])
    print("  objects_found_with_label", m["objects_found_with_label"], "places_covered", m["places_covered"], "places_spurious", m["places_spurious"],
          "rooms_matched", m["rooms_matched"], "room_assignment_agreement", m["room_assignment_agreement"], "buildings", m["buildings_run"])
    print("  queries", m["queries"], "success", m["query_success"], m["query_status"], "| mesh_vertices_ratio", m["mesh_vertices_ratio"])
    lg = r["log"]
    print("  log: errors", lg["n_error_lines"], lg["errors"][:2], "| no_rooms_calls", lg["n_no_rooms"], "| last room call: places",
          lg.get("last_places_for_rooms"), "best", lg.get("last_best"), "steps", lg["steps_logged"], "| hydra", r.get("hydra"))
