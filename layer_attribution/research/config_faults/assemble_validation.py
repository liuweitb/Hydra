#!/usr/bin/env python3
"""Assemble research/config_faults/validation.json from the result files under D:/hydra_work/results/config_faults/.

Windows python, json only; safe to rerun at any time, missing inputs are skipped.
usage: assemble_validation.py [OUT_JSON]
"""

import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
import build_validation as bv  # noqa: E402

R = pathlib.Path("D:/hydra_work/results/config_faults")
LABEL_FAULTS = ("label_confusion", "label_dropout")


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def rooms_default_summary(rows):
    items = {k: v["rooms"] for k, v in rows.items()}
    agree = [v["place_assignment_agreement"] for v in items.values() if v["place_assignment_agreement"] is not None]
    diffs = [v["size_abs_diff_max"] for v in items.values() if v["size_abs_diff_max"] is not None]
    return {
        "n_graphs": len(items),
        "n_rooms_equal": sum(v["n_rooms_equal"] for v in items.values()),
        "threshold_equal": sum(v["threshold_equal"] for v in items.values()),
        "real_n_rooms_range": [min(v["real"]["n_rooms"] for v in items.values()), max(v["real"]["n_rooms"] for v in items.values())],
        "place_assignment_agreement": {"min": min(agree), "mean": round(statistics.mean(agree), 4), "max": max(agree)},
        "room_size_abs_diff_max_over_graphs": max(diffs) if diffs else None,
        "room_size_abs_diff_max_median": statistics.median(diffs) if diffs else None,
    }


def objects_default_summary(rows):
    clean, label = {}, {}
    for k, v in rows.items():
        (label if any(f in k for f in LABEL_FAULTS) else clean)[k] = v["objects"]

    def stats(d):
        if not d:
            return None
        err = [abs(v["emulated_n"] - v["real_n"]) / v["real_n"] for v in d.values()]
        found = [v["real_found_one_to_one_by_emulated"] for v in d.values()]
        return {"n_runs": len(d), "count_rel_error_mean": round(statistics.mean(err), 3), "count_rel_error_max": round(max(err), 3),
                "found_min": min(found), "found_mean": round(statistics.mean(found), 3)}
    return {"clean_labels_and_geometry_faults": stats(clean), "label_fault_runs": stats(label)}


def main():
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "validation.json"
    analysis = load("analysis_v1.json")
    res = {"protocol": {
        "dataset": "D:/hydra_work/data/uhumans2_office_mp3d, clean uHumans2 office, 2077 frames",
        "truncated_runs": "first 800 frames (--max-steps 800), clean data, one config key changed via make_config_dir()",
        "parallelism": "at most 2 Hydra jobs, pinned to cores 8,9, run_batch.sh",
        "reference": "tmp_cfg_ref (severity-0 config dir); tmp_cfg_ref_default (no --config-path) is identical in every layer",
        "layer_identity": "exact equality of sorted node rows (position, label, size/distance) and of parent-edge rows per layer; mesh vertices+labels",
        "query_test": "analyze_runs.py: object of class c in room R, gates rooms 3 m, query 1 m (58 queries truncated, 105 full length)",
    }}
    if analysis:
        res["reference"] = analysis["reference"]
        res["runs"] = {n: bv.summarize(n, r) for n, r in analysis["runs"].items()}
    for key, fn in (("room_finder_emulator_vs_hydra_truncated", "emulator_check_v1.json"),
                    ("room_query_emulator_vs_hydra_truncated", "query_emulator_check_v1.json"),
                    ("object_emulator_vs_hydra_truncated", "object_emulator_check_v1.json"),
                    ("full_length_predictions", "full_length_predictions_v1.json")):
        d = load(fn)
        if d is not None:
            res[key] = d
    rows = load("emulator_default_validation.json")
    if rows:
        res["emulators_default_config_vs_hydra_full_length_dev_runs"] = {
            "what": "each dev-data run's own places layer / labeled mesh through the emulator at default config vs that run's real rooms / objects",
            "rooms": rooms_default_summary(rows),
            "objects": objects_default_summary(rows),
            "per_run": rows,
        }
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}: runs={len(res.get('runs', {}))} sections={[k for k in res if k not in ('protocol', 'runs', 'reference')]}")


if __name__ == "__main__":
    main()
