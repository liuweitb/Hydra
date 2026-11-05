"""Compact view of research/config_faults analysis_v1.json (read-only cross-check input)."""
import json
import sys

d = json.load(open(sys.argv[1]))
for name, r in d["runs"].items():
    ident = r.get("identical", {})
    changed = [k for k, v in ident.items() if (v is False) or (isinstance(v, dict) and not v.get("nodes_and_parent_edge", True))]
    eff = {k: (v["echoed"], v["ok"]) for k, v in r.get("effective", {}).items()}
    c = r.get("counts", {})
    print(f"{name:40s} fault={r.get('fault')} sev={r.get('severity')} exp={r.get('expected_first_layer')} changed={changed}")
    print(f"    effective={eff}")
    print(f"    counts(ref,run)={ {k: v for k, v in c.items() if v[0] != v[1]} }")
