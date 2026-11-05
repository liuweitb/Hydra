#!/usr/bin/env python3
"""Condense analysis_*.json (+ emulator checks) into research/config_faults/validation.json.

Runs on Windows python (reads json only). usage: build_validation.py OUT ANALYSIS_JSON [ANALYSIS_JSON ...] [--emu EMU_JSON]
"""

import json
import pathlib
import sys

ORDER = ["mesh", "objects", "places", "mesh_places", "rooms", "buildings"]
DOWNSTREAM = {"objects": [], "places": ["rooms", "buildings"], "rooms": ["buildings"]}
KEEP = ["mesh_vertices_ratio", "objects_ref", "objects_run", "objects_found", "objects_found_with_label",
        "objects_precision", "places_ref", "places_run", "places_covered", "places_spurious", "rooms_ref", "rooms_run",
        "rooms_matched", "room_assignment_agreement", "buildings_run", "queries", "query_success", "query_status"]


def layer_changed(rec):
    out = []
    if not rec["identical"]["mesh"]:
        out.append("mesh")
    for l in ORDER[1:]:
        if not rec["identical"][l]["nodes"]:
            out.append(l)
    return out


def edges_changed(rec):
    return [l for l in ORDER[1:] if rec["identical"][l]["nodes"] and not rec["identical"][l]["nodes_and_parent_edge"]]


def summarize(name, rec):
    s = {"fault": rec.get("fault", "none"), "severity": rec.get("severity"), "hydra": rec["hydra"]}
    if "error" in rec:
        s["error"] = rec["error"]
        return s
    if rec["hydra"] is None:  # batch status line lost with the killed first batch process
        s["hydra"] = {"rc": None, "note": f"no status line; dsg_with_mesh.json written, {rec['log']['steps_logged']} steps logged"}
    s["params"] = {k: v["value"] for k, v in rec.get("params", {}).items()}
    s["effective_config_echo"] = rec.get("effective")
    changed = layer_changed(rec)
    s["layers_changed_exact"] = changed
    s["parent_edges_changed_only"] = edges_changed(rec)
    s["first_changed_layer"] = changed[0] if changed else None
    exp = rec.get("expected_first_layer")
    s["expected_first_layer"] = exp
    if exp:
        allowed = {exp, *DOWNSTREAM.get(exp, [])}
        s["confined_to_expected_and_downstream"] = bool(changed) and set(changed) <= allowed
        s["first_layer_matches_expected"] = bool(changed) and changed[0] == exp
    s["counts_ref_vs_run"] = rec["counts"]
    s["room_sizes_run"] = rec["room_sizes"]
    s["room_finder_last_call"] = {k: rec["log"].get(k) for k in ("last_places_for_rooms", "last_best", "n_no_rooms")}
    s["log_errors"] = rec["log"]["errors"]
    m = rec["metrics"]
    s["metrics"] = {k: m.get(k) for k in KEEP}
    return s


def main():
    args = sys.argv[1:]
    emu = None
    if "--emu" in args:
        i = args.index("--emu")
        emu = json.loads(pathlib.Path(args[i + 1]).read_text())
        del args[i : i + 2]
    out, files = pathlib.Path(args[0]), args[1:]
    result = {"runs": {}, "reference": None}
    for f in files:
        d = json.loads(pathlib.Path(f).read_text())
        result["reference"] = result["reference"] or d["reference"]
        for name, rec in d["runs"].items():
            result["runs"][name] = summarize(name, rec)
    if emu:
        result["room_finder_emulator_vs_hydra_truncated"] = emu
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out}: {len(result['runs'])} runs")


if __name__ == "__main__":
    main()
