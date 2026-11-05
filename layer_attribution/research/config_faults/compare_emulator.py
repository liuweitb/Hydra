#!/usr/bin/env python3
"""Emulated vs real room finding for the truncated rooms runs.

usage: compare_emulator.py RUNS_ROOT REF_NAME ANALYSIS_JSON OUT_JSON [PLACES_CACHE_JSON]
With a places cache (dump_places.py) no spark_dsg is needed, RUNS_ROOT is then unused.
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import emulate_rooms as er  # noqa: E402

runs_root, ref_name, analysis, out = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3]), pathlib.Path(sys.argv[4])
data = json.loads(analysis.read_text())
dist, edges = er.load_cached_places(sys.argv[5]) if len(sys.argv) > 5 else er.load_places(runs_root / ref_name)
rows = {}
ref = data["reference"]
rows[ref_name] = dict(
    params={},
    emulated=er.find_rooms(dist, edges),
    real=dict(n_rooms=ref["counts"]["rooms"], sizes=ref["room_sizes"], best=ref["log"].get("last_best")),
)
for name, rec in data["runs"].items():
    if not rec.get("fault", "").startswith("rooms_") or "counts" not in rec:
        continue
    over = {k.rsplit(".", 1)[1]: v["value"] for k, v in rec["params"].items()}
    rows[name] = dict(
        params=over,
        emulated=er.find_rooms(dist, edges, **over),
        real=dict(n_rooms=rec["counts"]["rooms"][1], sizes=rec["room_sizes"], best=rec["log"].get("last_best")),
    )
for name, r in rows.items():
    e, rl = r["emulated"], r["real"]
    best = rl["best"] or {}
    r["agree_n_rooms"] = e["n_rooms"] == rl["n_rooms"]
    r["agree_threshold"] = (best.get("threshold_m") is not None and e["threshold"] is not None
                            and abs(best["threshold_m"] - e["threshold"]) < 1e-4) or (best == {} and e["threshold"] is None)
    r["size_abs_diff_max"] = (max(abs(a - b) for a, b in zip(e["sizes"], rl["sizes"]))
                              if e["n_rooms"] == rl["n_rooms"] and e["sizes"] else None)
    print(name, r["params"], "| emu", e["n_rooms"], e["sizes"], e["threshold"], "| real", rl["n_rooms"], rl["sizes"], best,
          "| n_rooms", r["agree_n_rooms"], "thr", r["agree_threshold"], "maxdiff", r["size_abs_diff_max"])
out.write_text(json.dumps(rows, indent=1))
