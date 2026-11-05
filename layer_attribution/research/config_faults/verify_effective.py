#!/usr/bin/env python3
"""Check that the perturbed value of each config-fault run is what Hydra echoed in its INFO log.

usage: verify_effective.py RUNS_ROOT CFG_ROOT NAME [NAME ...]
"""

import json
import pathlib
import sys

from layer_attribution.pipeline import config_faults as cf

runs, cfgs = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
out = {}
for name in sys.argv[3:]:
    man = json.loads((cfgs / name / cf.MANIFEST).read_text())
    expected = {k: v["value"] for k, v in man["params"].items()}
    res = cf.check_effective(runs / name / "logs" / "hydra_python.INFO", expected)
    out[name] = {k: {"expected": expected[k], "echoed": t, "ok": ok} for k, (t, ok) in res.items()}
    # entries that differ from struct defaults inside the perturbed section, for context
    print(name, json.dumps(out[name]))
