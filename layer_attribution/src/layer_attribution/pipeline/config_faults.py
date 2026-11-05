#!/usr/bin/env python3
"""Parameter-level (config) faults: clean data, one perturbed Hydra parameter.

make_config_dir() writes a copy of Hydra's config/ tree with ONE key of datasets/<dataset>.yaml changed; pass the
result as --config-path to pipeline/run_hydra.py (scripts/run_job.sh NAME DATASET CONFIG_PATH [STEPS]).
Hydra's loader (hydra_python.pipeline.load_configs) reads only datasets/<dataset>.yaml and
label_spaces/<label_space>_label_space.yaml, but the whole tree is copied so the dir is self-contained.

  python -m layer_attribution.pipeline.config_faults --list
  python -m layer_attribution.pipeline.config_faults --plan
  python -m layer_attribution.pipeline.config_faults --fault rooms_min_size --severity 0.5 --dest DIR

Hydra silently ignores unknown keys, so every key is checked against the base file here, and
check_effective() re-reads the values Hydra echoed into <run>/logs/hydra_python.INFO.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

LAYER_ORDER = ["mesh", "objects", "places", "rooms", "buildings"]
DATASET = "uhumans2"
LABEL_SPACE = "uhumans2_office"
DEFAULT_BASE = Path(__file__).resolve().parents[4] / "config"
MANIFEST = "config_fault.json"

_RF = "backend.update_functors.rooms.room_finder"
_GP = "frontend.freespace_places"


@dataclass(frozen=True)
class Rule:
    """severity -> value of one key. 0 gives `default` exactly, 1 gives `extreme`, monotone in between."""

    default: float
    extreme: float
    scale: str = "lin"  # lin: straight line; geo: constant ratio per unit severity (ends > 0)
    integer: bool = False  # size_t keys must stay YAML ints
    digits: int = 4

    def __post_init__(self):
        if self.scale not in ("lin", "geo"):
            raise ValueError(f"scale must be lin or geo, got {self.scale!r}")
        if self.scale == "geo" and not (self.default > 0 and self.extreme > 0):
            raise ValueError("geo rule needs positive ends")

    def __call__(self, severity: float):
        if severity <= 0:
            v = self.default
        elif severity >= 1:
            v = self.extreme
        elif self.scale == "lin":
            v = self.default + severity * (self.extreme - self.default)
        else:
            v = self.default * (self.extreme / self.default) ** severity
        if self.integer:
            return int(math.floor(v + 0.5))
        return round(float(v), self.digits)


@dataclass(frozen=True)
class ConfigFault:
    layer: str  # first layer whose output changes
    rules: dict  # dotted key in datasets/<dataset>.yaml -> Rule
    downstream: tuple = ()  # layers that change only as a consequence
    note: str = ""
    recommended: bool = False
    launch_severities: tuple = ()  # three severities for full-length runs, recommended faults only


# fault name -> (layer, {key: severity-to-value rule}); severity 0 = Hydra's uhumans2 value
FAULTS: dict[str, ConfigFault] = {
    "objects_min_size": ConfigFault(
        "objects",
        {"frontend.objects.min_cluster_size": Rule(40, 1000, "geo", integer=True)},
        note="clusters below this many mesh vertices never become objects: small classes (computer, painting) vanish first",
        recommended=True,
        launch_severities=(0.3, 0.6, 1.0),
    ),
    "objects_tol_merge": ConfigFault(
        "objects",
        {"frontend.objects.cluster_tolerance": Rule(0.25, 4.0, "geo")},
        note="clustering radius grows: same-class neighbours merge, the centroid lands between the originals (wrong or missing target)",
        recommended=True,
        launch_severities=(0.7, 0.85, 1.0),
    ),
    "objects_tol_split": ConfigFault(
        "objects",
        {"frontend.objects.cluster_tolerance": Rule(0.25, 0.05, "geo")},
        note="radius below the ~0.1 m mesh vertex spacing: objects vanish; 0.10-0.12 m is a cliff, not a graded response",
    ),
    "rooms_min_size": ConfigFault(
        "rooms",
        {f"{_RF}.min_room_size": Rule(10, 400, "geo", integer=True)},
        downstream=("buildings",),
        note="clusters with fewer places than this are dropped, the others are unchanged; no rooms and no building at the top of the range",
        recommended=True,
        launch_severities=(0.6, 0.8, 1.0),
    ),
    "rooms_min_component": ConfigFault(
        "rooms",
        {f"{_RF}.min_component_size": Rule(10, 200, "geo", integer=True)},
        downstream=("buildings",),
        note="rooms merge; non-monotone response (8, 7, 6, 4, 3, 2, 1 rooms), not graded",
    ),
    "rooms_plateau_ratio": ConfigFault(
        "rooms",
        {f"{_RF}.plateau_ratio": Rule(0.15, 1.0)},
        downstream=("buildings",),
        note="rooms merge in one step (8 to 3 rooms near 0.45), identical above: not graded",
    ),
    "rooms_max_dilation": ConfigFault(
        "rooms",
        {f"{_RF}.max_dilation_m": Rule(1.2, 0.6)},
        downstream=("buildings",),
        note="dilation window ceiling lowered: lower threshold, rooms merge (8, 7, 6, 3, 1 rooms)",
        recommended=True,
        launch_severities=(0.16, 0.3, 1.0),
    ),
    "rooms_min_dilation": ConfigFault(
        "rooms",
        {f"{_RF}.min_dilation_m": Rule(0.5, 1.15)},
        downstream=("buildings",),
        note="dilation window floor raised: weak and non-monotone (8, 7, 6 rooms)",
    ),
    "places_compression": ConfigFault(
        "places",
        {f"{_GP}.graph.compression_distance_m": Rule(1.5, 6.0, "geo")},
        downstream=("rooms", "buildings"),
        note="GVD graph compressed to one place per larger cell: sparse places",
    ),
    "places_min_node_dist": ConfigFault(
        "places",
        {f"{_GP}.graph.min_node_distance_m": Rule(0.4, 1.4)},
        downstream=("rooms", "buildings"),
        note="places need more clearance: narrow passages lose their places",
    ),
    "places_max_dist": ConfigFault(
        "places",
        {f"{_GP}.gvd.max_distance_m": Rule(4.5, 0.7)},
        downstream=("rooms", "buildings"),
        note="free-space distance capped: open-space places and their radii are clipped",
    ),
}
RECOMMENDED = tuple(n for n, f in FAULTS.items() if f.recommended)


def _check_severity(severity: float) -> float:
    if isinstance(severity, bool) or not isinstance(severity, (int, float)) or not math.isfinite(severity):
        raise ValueError(f"severity must be a finite number, got {severity!r}")
    if not 0.0 <= severity <= 1.0:
        raise ValueError(f"severity must be in [0, 1], got {severity}")
    return float(severity)


def _get(fault: str) -> ConfigFault:
    if fault not in FAULTS:
        raise ValueError(f"unknown config fault {fault!r}; valid: {sorted(FAULTS)}")
    return FAULTS[fault]


def values_at(fault: str, severity: float) -> dict:
    """{dotted key: value} the fault writes at this severity (pure, no files)."""
    spec = _get(fault)
    severity = _check_severity(severity)
    return {key: rule(severity) for key, rule in spec.rules.items()}


def run_name(fault: str, severity: float, prefix: str = "uh2_cfg") -> str:
    """<prefix>_<fault>_s<NN>, NN = severity x 100 (s100 for 1.0); the sweep scheme without a seed (config faults are deterministic)."""
    _get(fault)
    return f"{prefix}_{fault}_s{round(_check_severity(severity) * 100):02d}"


def launch_plan() -> list[dict]:
    """Runs for the coordinator: every recommended fault at its launch severities, on clean data, full length."""
    plan = []
    for name in RECOMMENDED:
        spec = FAULTS[name]
        for s in spec.launch_severities:
            plan.append({"name": run_name(name, s), "fault": name, "severity": s, "layer": spec.layer,
                         "downstream": list(spec.downstream), "values": values_at(name, s)})
    return plan


def _walk(tree: dict, dotted: str):
    """Parent dict and last key of a dotted path; every element must already exist (no silent new keys)."""
    parts = dotted.split(".")
    node = tree
    for p in parts[:-1]:
        if not isinstance(node, dict) or p not in node:
            raise KeyError(f"config key {dotted!r} not in base config (missing {p!r})")
        node = node[p]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise KeyError(f"config key {dotted!r} not in base config")
    return node, parts[-1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_config_dir(
    fault: str,
    severity: float,
    dest,
    base_config_dir=None,
    *,
    dataset: str = DATASET,
    label_space: str = LABEL_SPACE,
) -> Path:
    """Write a config tree with `fault` at `severity` to `dest` (new or empty dir); returns dest."""
    spec = _get(fault)
    severity = _check_severity(severity)
    base = Path(base_config_dir) if base_config_dir is not None else DEFAULT_BASE
    base = base.resolve()
    dest = Path(dest).resolve()
    ds_rel = Path("datasets") / f"{dataset}.yaml"
    ls_rel = Path("label_spaces") / f"{label_space}_label_space.yaml"
    for rel in (ds_rel, ls_rel):
        if not (base / rel).is_file():
            raise FileNotFoundError(base / rel)
    if dest == base or base in dest.parents or dest in base.parents:
        raise ValueError("dest must not overlap the base config dir")
    if dest.exists() and any(dest.iterdir()):
        raise ValueError(f"dest not empty: {dest}")

    tree = yaml.safe_load((base / ds_rel).read_text(encoding="utf-8"))
    original = copy.deepcopy(tree)
    changes = {}
    for key, rule in spec.rules.items():
        parent, last = _walk(tree, key)
        old = parent[last]
        if isinstance(old, bool) or not isinstance(old, (int, float)):
            raise TypeError(f"{key} is {old!r} in the base config, not a number")
        if not math.isclose(old, rule.default, rel_tol=1e-9, abs_tol=1e-12) or isinstance(old, int) != rule.integer:
            raise ValueError(f"rule default for {key} ({rule.default}, integer={rule.integer}) != base value {old!r}")
        new = rule(severity)
        parent[last] = new
        changes[key] = {"default": old, "value": new}

    shutil.copytree(base, dest, dirs_exist_ok=True)
    text = yaml.safe_dump(tree, sort_keys=False, default_flow_style=False)
    if yaml.safe_load(text) != tree:  # dump must round-trip, Hydra re-parses the file with safe_load
        raise RuntimeError("yaml round trip changed the config")
    (dest / ds_rel).write_text(text, encoding="utf-8", newline="\n")

    manifest = {
        "schema_version": 1,
        "fault": fault,
        "severity": severity,
        "seed": None,
        "params": changes,
        "expected_first_layer": spec.layer,
        "downstream_layers": list(spec.downstream),
        "corrupted_input": "config",
        "note": spec.note,
        "layer_order": LAYER_ORDER,
        "dataset": dataset,
        "label_space": label_space,
        "base_config_dir": str(base),
        "base_dataset_yaml_sha256": _sha256(base / ds_rel),
        "dataset_yaml_sha256": _sha256(dest / ds_rel),
        "n_keys_changed_vs_base": len(_diff(original, tree)),
    }
    (dest / MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    return dest


def _diff(a, b, prefix=""):
    """Dotted paths whose values differ between two nested dicts."""
    out = []
    for k in sorted(set(a) | set(b)):
        p = f"{prefix}{k}"
        if k not in a or k not in b:
            out.append(p)
        elif isinstance(a[k], dict) and isinstance(b[k], dict):
            out += _diff(a[k], b[k], p + ".")
        elif a[k] != b[k]:
            out.append(p)
    return out


def diff_configs(dir_a, dir_b, dataset: str = DATASET) -> list[str]:
    """Dotted keys that differ between the dataset yamls of two config dirs."""
    load = lambda d: yaml.safe_load((Path(d) / "datasets" / f"{dataset}.yaml").read_text(encoding="utf-8"))
    return _diff(load(dir_a), load(dir_b))


# ------------------------------------------------------------------ what Hydra actually used

_GLOG = re.compile(r"^[IWEF]\d{8} ")
_BANNER = re.compile(r"^\*\s+(\w+)\s+\*$")
# "name", "name[key]" or "name [units]" then optional "[Type]" and "(default)" before the colon
_LINE = re.compile(r"^(\s*)([A-Za-z_]\w*(?:\[[^\]]*\])?)(?:\s+\[[^\]]*\])*(?:\s+\(default\))?:\s*(.*?)\s*$")


def read_effective_config(info_log) -> dict:
    """{(module, section, ..., key): (value_text, is_default)} from the config echo at the top of
    <run>/logs/hydra_python.INFO. hydra_config.yaml only holds the global PipelineConfig, not frontend/backend."""
    out, stack, module = {}, [], None
    with open(info_log, errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if _GLOG.match(line):
                if "Got input @" in line:
                    break
                continue
            b = _BANNER.match(line.strip())
            if b:
                module, stack = b.group(1), []
                continue
            if module is None or line.lstrip().startswith("="):
                continue
            m = _LINE.match(line)
            if not m:
                continue
            indent, name, value = len(m.group(1)), m.group(2), m.group(3)
            tokens = [t for t in re.split(r"[\[\]]", name) if t]
            while stack and stack[-1][0] >= indent:
                stack.pop()
            if value == "":
                stack.append((indent, tokens))
                continue
            is_default = value.endswith("(default)")
            value = value[: -len("(default)")].strip() if is_default else value
            path = (module, *[t for _, ts in stack for t in ts], *tokens)
            out[path] = (value, is_default)
    return out


def find_effective(cfg: dict, dotted: str):
    """Entry for a yaml key. The dump nests flattened structs (objects.clustering.*), so match the yaml path
    as an ordered subsequence of the dump path with the same module and leaf."""
    want = dotted.split(".")
    hits = []
    for path, val in cfg.items():
        if path[0] != want[0] or path[-1] != want[-1]:
            continue
        it = iter(path)
        if all(any(p == q for q in it) for p in want):
            hits.append((path, val))
    if len(hits) != 1:
        raise KeyError(f"{dotted!r} matches {len(hits)} entries in the effective config: {[h[0] for h in hits]}")
    return hits[0]


def _same_number(text: str, expected) -> bool:
    try:
        return math.isclose(float(text), float(expected), rel_tol=1e-5, abs_tol=1e-9)
    except ValueError:
        return False


def check_effective(info_log, expected: dict) -> dict:
    """{key: (echoed_text, matches)} for yaml-key -> expected value, e.g. values_at(fault, severity)."""
    cfg = read_effective_config(info_log)
    res = {}
    for key, want in expected.items():
        try:
            _, (text, _) = find_effective(cfg, key)
        except KeyError as e:
            res[key] = (str(e), False)
            continue
        res[key] = (text, _same_number(text, want))
    return res


def find_info_log(run_dir) -> Path:
    """<run>/logs/hydra_python.INFO is a symlink Windows cannot follow over the WSL share: fall back to the newest real file."""
    logs = Path(run_dir) / "logs"
    link = logs / "hydra_python.INFO"
    try:
        if link.is_file():
            return link
    except OSError:
        pass
    real = sorted(logs.glob("hydra_python.*.INFO.*"), key=lambda p: p.stat().st_mtime)
    if not real:
        raise FileNotFoundError(f"no hydra_python INFO log in {logs}")
    return real[-1]


def verify_run(run_dir, config_dir) -> dict:
    """check_effective() for a finished run against the manifest in the config dir it was started with."""
    manifest = json.loads((Path(config_dir) / MANIFEST).read_text(encoding="utf-8"))
    expected = {k: v["value"] for k, v in manifest["params"].items()}
    return check_effective(find_info_log(run_dir), expected)


# ------------------------------------------------------------------ cli


def _table() -> str:
    rows = ["fault | layer | recommended | key | s=0 | 0.25 | 0.5 | 0.75 | 1.0", "-|-|-|-|-|-|-|-|-"]
    for name, f in FAULTS.items():
        for key, rule in f.rules.items():
            vals = " | ".join(str(rule(s)) for s in (0, 0.25, 0.5, 0.75, 1))
            rows.append(f"{name} | {f.layer} | {'yes' if f.recommended else 'no'} | {key} | {vals}")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the fault table and exit")
    ap.add_argument("--plan", action="store_true", help="print the recommended runs (fault x launch severities) as json lines and exit")
    ap.add_argument("--verify", nargs=2, metavar=("RUN_DIR", "CONFIG_DIR"), help="check a finished run's echoed config against its config dir")
    ap.add_argument("--fault", choices=sorted(FAULTS))
    ap.add_argument("--severity", type=float)
    ap.add_argument("--dest", type=Path)
    ap.add_argument("--base", type=Path, default=None, help=f"base config dir (default {DEFAULT_BASE})")
    a = ap.parse_args(argv)
    if a.list:
        print(_table())
        return
    if a.plan:
        for row in launch_plan():
            print(json.dumps(row))
        return
    if a.verify:
        res = verify_run(*a.verify)
        print(json.dumps({k: {"echoed": t, "ok": ok} for k, (t, ok) in res.items()}))
        raise SystemExit(0 if all(ok for _, ok in res.values()) else 1)
    if a.fault is None or a.severity is None or a.dest is None:
        ap.error("--fault, --severity and --dest are required")
    dest = make_config_dir(a.fault, a.severity, a.dest, a.base)
    print(json.dumps(json.loads((dest / MANIFEST).read_text())["params"]))


if __name__ == "__main__":
    main()
