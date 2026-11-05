"""Check which dataset-yaml keys Hydra actually applied, from the effective-config dump in <run>/logs.

Hydra prints every module config (with "(default)" marks) into the glog INFO file at start-up;
hydra_config.yaml only holds the global PipelineConfig. A yaml key with no matching printed field
was ignored (misspelled / wrong namespace); a printed value that differs was not applied.

usage: python verify_applied_config.py RUN_DIR_OR_LOG DATASET_YAML [--json OUT]
"""
import argparse
import glob
import json
import math
import os
import re
import sys

import yaml

GLOG_PREFIX = re.compile(r"^[IWEF]\d{8} ")
BANNER = re.compile(r"^\* (\w+)\s+\*$")
HEADER = re.compile(r"^=+ .* =+$")
SECTION = re.compile(r"^(?P<ind> *)(?P<name>[\w\.]+(?:\[[^\]]*\])?)(?P<rest>(?: \[[^\]]*\])*)(?: \(default\))?:\s*$")
UNINIT = re.compile(r"^(?P<ind> *)(?P<name>[\w\.]+(?:\[[^\]]*\])?) \[Uninitialized Virtual Config\]\s*$")
LEAF = re.compile(r"^(?P<ind> *)(?P<name>[\w\.]+)(?: \[[^\]]*\])?:\s+(?P<val>\S.*?)\s*$")
ROOT_ALIAS = {"reconstruction": "active_window"}


def find_log(path):
    if os.path.isfile(path):
        return path
    cands = [p for p in glob.glob(os.path.join(path, "logs", "hydra_python*INFO*")) if not p.endswith("hydra_python.INFO")]
    if not cands:
        raise FileNotFoundError(f"no glog INFO file under {path}/logs")
    return sorted(cands)[0]


def tokens(name):
    m = re.match(r"^([\w\.]+)\[([^\]]*)\]$", name)
    return [m.group(1), m.group(2)] if m else [name]


def parse_dump(log_path):
    """-> (leaves, sections): leaves[path tuple] = (text, is_default); sections[path tuple] = plain|virtual_init|virtual_uninit"""
    leaves, sections = {}, {}
    root, stack = (), []
    seen_modules = False
    with open(log_path, errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if GLOG_PREFIX.match(line):
                if "reconstruction_module.cpp" in line and "Got input" in line:
                    break
                continue
            m = BANNER.match(line)
            if m:
                root, stack, seen_modules = (ROOT_ALIAS.get(m.group(1), m.group(1)),), [], True
                continue
            if HEADER.match(line) or not line.strip() or set(line.strip()) <= {"=", "*"}:
                if not seen_modules or HEADER.match(line):
                    stack = []
                continue
            ind = len(line) - len(line.lstrip(" "))
            if ind >= 25:
                continue  # continuation of a multi-line value
            mu = UNINIT.match(line)
            if mu:
                while stack and stack[-1][0] >= ind:
                    stack.pop()
                path = root + tuple(t for _, ts in stack for t in ts) + tuple(tokens(mu.group("name")))
                sections[path] = "virtual_uninit"
                continue
            ms = SECTION.match(line)
            if ms:
                while stack and stack[-1][0] >= ind:
                    stack.pop()
                toks = tokens(ms.group("name"))
                stack.append((ind, toks))
                kind = "virtual_init" if "Virtual Config" in ms.group("rest") else "plain"
                sections[root + tuple(t for _, ts in stack for t in ts)] = kind
                continue
            ml = LEAF.match(line)
            if ml:
                while stack and stack[-1][0] >= ind:
                    stack.pop()
                val = ml.group("val")
                is_default = val.endswith("(default)")
                if is_default:
                    val = val[: -len("(default)")].rstrip()
                path = root + tuple(t for _, ts in stack for t in ts) + (ml.group("name"),)
                leaves[path] = (val, is_default)
    return leaves, sections


def flatten(d, prefix=()):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flatten(v, prefix + (str(k),))
        else:
            yield prefix + (str(k),), v


def is_subsequence(needle, hay):
    it = iter(hay)
    return all(any(n == h for h in it) for n in needle)


# sub-configs the printer nests but the yaml flattens; yaml-only namespaces (pgmo, covariance, gnc) are not printed
FLATTENED = {"clustering", "frames", "graph", "label_space"}


def path_matches(yaml_path, dump_path):
    """root and leaf equal; every printed section between them must also be a yaml key, in order."""
    if yaml_path[0] != dump_path[0] or yaml_path[-1] != dump_path[-1]:
        return False
    printed = [t for t in dump_path[1:-1] if t not in FLATTENED]
    return is_subsequence(printed, yaml_path[1:-1])


def section_matches(yaml_path, section_path):
    if yaml_path[0] != section_path[0] or yaml_path[-1] != section_path[-1]:
        return False
    if len(yaml_path) == 1 and len(section_path) == 1:
        return True
    printed = [t for t in section_path[1:-1] if t not in FLATTENED]
    return is_subsequence(printed, yaml_path[1:-1])


def same_value(printed, wanted):
    try:
        p = yaml.safe_load(printed)
    except yaml.YAMLError:
        p = printed
    if isinstance(wanted, bool) or isinstance(p, bool):
        return p == wanted
    if isinstance(wanted, (int, float)) and isinstance(p, (int, float)):
        return math.isclose(float(p), float(wanted), rel_tol=1e-5, abs_tol=1e-9)
    return p == wanted or str(p) == str(wanted)


def verify(log_path, yaml_path):
    leaves, sections = parse_dump(log_path)
    cfg = yaml.safe_load(open(yaml_path))
    rows = []
    for path, wanted in flatten(cfg):
        if path[-1] == "type" and len(path) > 1:
            parent = path[:-1]
            cands = [s for s in sections if section_matches(parent, s)]
            kinds = {sections[s] for s in cands}
            if kinds == {"virtual_init"}:
                status = "applied"
            elif "virtual_uninit" in kinds:
                status = "type_not_resolved"
            else:
                status = "ignored_no_such_field"  # section missing or not a virtual config: no 'type' field
            rows.append((".".join(path), wanted, ",".join(sorted(kinds)) or None, status))
            continue
        cands = [(p, v) for p, v in leaves.items() if path_matches(path, p)]
        if not cands:
            rows.append((".".join(path), wanted, None, "ignored_no_such_field"))
            continue
        p, (val, dflt) = max(cands, key=lambda c: len(c[0]))
        ok = same_value(val, wanted)
        rows.append((".".join(path), wanted, val + (" (default)" if dflt else ""), "applied" if ok else "value_differs"))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_or_log")
    ap.add_argument("dataset_yaml")
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = verify(find_log(a.run_or_log), a.dataset_yaml)
    bad = [r for r in rows if r[3] != "applied"]
    for r in rows:
        print(f"{r[3]:24s} {r[0]:70s} yaml={r[1]!r} effective={r[2]}")
    print(f"\n{len(rows)} leaf keys, {len(rows) - len(bad)} applied, {len(bad)} not applied")
    if a.json:
        json.dump([dict(key=r[0], yaml_value=r[1], effective=r[2], status=r[3]) for r in rows], open(a.json, "w"), indent=1, default=str)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
