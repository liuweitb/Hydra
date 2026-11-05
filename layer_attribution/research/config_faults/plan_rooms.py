#!/usr/bin/env python3
"""Severity ladders of every rooms fault on the full-length clean scene, through the query-level emulator, with a stability
check (same room count when the parameter moves by +-2 % and +-5 % of its range). Windows python.

usage: plan_rooms.py FULL_CLEAN_SCENE_JSON OUT_JSON [--rules-from-module]
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import emulate_queries as eq  # noqa: E402
from layer_attribution.pipeline import config_faults as cf  # noqa: E402

INT_KEYS = ("min_room_size", "min_component_size")


def cast(key, v):
    return int(round(v)) if key in INT_KEYS else float(v)


def main():
    scene = eq.load_scene(sys.argv[1])
    out = {}
    for name, f in cf.FAULTS.items():
        if f.layer != "rooms":
            continue
        (full_key, rule), = f.rules.items()
        key = full_key.rsplit(".", 1)[1]
        lo, hi = sorted((rule.default, rule.extreme))
        span = hi - lo
        rows = []
        for s in [i / 20 for i in range(21)]:
            v = rule(s)
            base = eq.emulate(scene, {key: v})
            stable = True
            for rel in (-0.05, -0.02, 0.02, 0.05):
                w = min(max(v + rel * span, lo), hi)
                r = eq.emulate(scene, {key: cast(key, w)})
                if (r["n_rooms"], r["sizes"]) != (base["n_rooms"], base["sizes"]) and r["n_rooms"] != base["n_rooms"]:
                    stable = False
            rows.append({"severity": s, "value": v, **base, "n_rooms_stable_pm5pct_of_range": stable})
        out[name] = {"key": full_key, "rule": {"default": rule.default, "extreme": rule.extreme, "scale": rule.scale}, "ladder": rows}
        print("==", name, full_key, rule)
        for r in rows:
            print(f"  s={r['severity']:.2f} value={r['value']:<8} rooms={r['n_rooms']} sizes={r['sizes']} qsucc={r['query_success']} "
                  f"matched={r['rooms_matched']} agree={r['room_assignment_agreement']} stable={r['n_rooms_stable_pm5pct_of_range']}")
    pathlib.Path(sys.argv[2]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
