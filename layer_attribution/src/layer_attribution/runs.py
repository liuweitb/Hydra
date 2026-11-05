"""Index of Hydra run directories and the development / test split.

Thresholds and rules are fitted on 'reference', 'repeat', 'benign' and 'dev' runs only. 'test' runs are returned only
when the caller passes final=True, so no development script can read them by accident.

  python -m layer_attribution.runs ~/runs/uh2
"""

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REFERENCE = "uh2_clean"
REPEAT = "uh2_clean_b"
BENIGN_MAX_SEVERITY = 0.05
DEV_SEED = 0
# config faults are deterministic (no seed): the middle launch severity of each fault is dev, the others test
DEV_CONFIG_SEVERITY = {"objects_min_size": 0.6, "objects_tol_merge": 0.85, "rooms_min_size": 0.8, "rooms_max_dilation": 0.3}
DEV_SPLITS = ("reference", "repeat", "benign", "dev")
ALL_SPLITS = DEV_SPLITS + ("test",)


class SplitViolation(RuntimeError):
    pass


@dataclass(frozen=True)
class Run:
    name: str
    path: Path
    kind: str  # reference | repeat | sensor | config | unknown
    fault: str | None
    severity: float
    seed: int | None
    split: str  # reference | repeat | benign | dev | test | unknown
    expected_first_layer: str | None
    also_hits_directly: tuple
    complete: bool

    @property
    def dsg_path(self) -> Path:
        return self.path / "backend" / "dsg_with_mesh.json"


def assign_split(kind: str, severity: float, seed: int | None, fault: str | None = None) -> str:
    if kind in ("reference", "repeat"):
        return kind
    if kind == "unknown":
        return "unknown"
    if severity <= BENIGN_MAX_SEVERITY:
        return "benign"
    if kind == "config":
        return "dev" if abs(severity - DEV_CONFIG_SEVERITY.get(fault, -1.0)) < 1e-6 else "test"
    return "dev" if seed == DEV_SEED else "test"


def _also(value) -> tuple:
    if not value:
        return ()
    return tuple(value) if isinstance(value, (list, tuple)) else (value,)


def describe(path: Path) -> Run:
    path = Path(path)
    name = path.name
    complete = (path / "backend" / "dsg_with_mesh.json").is_file()
    man_path = path / "injection_manifest.json"
    if man_path.is_file():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        kind = "config" if man.get("corrupted_input") == "config" else "sensor"
        severity = float(man["severity"])
        seed = man.get("seed")
        return Run(name, path, kind, man["fault"], severity, seed, assign_split(kind, severity, seed, man["fault"]),
                   man.get("expected_first_layer"), _also(man.get("also_hits_directly")), complete)
    if name in (REFERENCE, REPEAT):
        kind = "reference" if name == REFERENCE else "repeat"
        return Run(name, path, kind, None, 0.0, None, kind, None, (), complete)
    return Run(name, path, "unknown", None, float("nan"), None, "unknown", None, (), complete)


def index_runs(root, include_tmp: bool = False) -> list[Run]:
    root = Path(root)
    runs = [describe(p) for p in sorted(root.iterdir()) if p.is_dir() and (include_tmp or not p.name.startswith("tmp_"))]
    return runs


def select(runs, splits=DEV_SPLITS, final: bool = False, faults=None, complete_only: bool = True) -> list[Run]:
    splits = tuple(splits)
    if "test" in splits and not final:
        raise SplitViolation("'test' runs are for the final evaluation only (pass final=True)")
    out = [r for r in runs if r.split in splits]
    if faults is not None:
        out = [r for r in out if r.fault in set(faults)]
    if complete_only:
        out = [r for r in out if r.complete]
    return out


def reference_run(runs) -> Run:
    for r in runs:
        if r.kind == "reference" and r.complete:
            return r
    raise FileNotFoundError(f"reference run {REFERENCE} missing or incomplete")


def summary(runs) -> str:
    lines = [f"{len(runs)} run dirs"]
    counts = Counter((r.split, r.complete) for r in runs)
    for (split, complete), n in sorted(counts.items()):
        lines.append(f"  {split:10s} {'complete' if complete else 'INCOMPLETE':10s} {n}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--include-tmp", action="store_true")
    args = ap.parse_args()
    runs = index_runs(args.root, args.include_tmp)
    print(summary(runs))
    for r in runs:
        print(f"{r.name:34s} {r.split:9s} {str(r.fault):16s} sev={r.severity:<5} seed={r.seed} "
              f"{'ok' if r.complete else 'incomplete'}")


if __name__ == "__main__":
    main()
