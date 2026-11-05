#!/usr/bin/env python3
"""Query-level emulation vs the real truncated rooms runs (same places, only room_finder differs). Windows python.

usage: compare_queries.py ANALYSIS_JSON TRUNC_REF_SCENE_JSON OUT_JSON
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import emulate_queries as eq  # noqa: E402

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
scene = eq.load_scene(sys.argv[2])
rows = {}
for name, rec in data["runs"].items():
    if not rec.get("fault", "").startswith("rooms_") or "metrics" not in rec:
        continue
    over = {k.rsplit(".", 1)[1]: v["value"] for k, v in rec["params"].items()}
    emu = eq.emulate(scene, over)
    m = rec["metrics"]
    real = {k: m[k] for k in ("rooms_run", "rooms_matched", "room_assignment_agreement", "queries", "query_success", "query_status")}
    rows[name] = {"params": over, "emulated": emu, "real": real,
                  "abs_err_query_success": None if emu["query_success"] is None or real["query_success"] is None
                  else round(abs(emu["query_success"] - real["query_success"]), 3)}
    print(name, over, "| query_success emu", emu["query_success"], "real", real["query_success"], "| rooms emu", emu["n_rooms"], "real",
          real["rooms_run"], "| matched emu", emu["rooms_matched"], "real", real["rooms_matched"], "| agreement emu",
          emu["room_assignment_agreement"], "real", real["room_assignment_agreement"], "| status emu", emu["query_status"], "real", real["query_status"])
pathlib.Path(sys.argv[3]).write_text(json.dumps(rows, indent=1))
