#!/usr/bin/env python3
"""Build the config dirs for the truncated validation runs and the job list for run_batch.sh.

  python build_jobs.py [--faults a,b,...] [--severities 0.5,1.0] [--out jobs.txt]
Config dirs go to D:/hydra_work/results/config_faults/cfg/<job name>/ (WSL: /mnt/d/...).
"""

import argparse
import shutil
from pathlib import Path

from layer_attribution.pipeline import config_faults as cf

CFG_ROOT = Path("D:/hydra_work/results/config_faults/cfg")
PREFIX = "tmp_cfg_"


def wsl_path(p: Path) -> str:
    p = p.resolve()
    return "/mnt/" + p.drive.rstrip(":").lower() + "/" + "/".join(p.parts[1:])


def job_name(fault: str, severity: float, tag: str = "") -> str:
    return f"{PREFIX}{tag}{fault}_s{round(severity * 100):03d}"


def build(fault: str, severity: float, tag: str = "") -> tuple[str, str]:
    name = job_name(fault, severity, tag)
    dest = CFG_ROOT / name
    if dest.exists():
        shutil.rmtree(dest)
    cf.make_config_dir(fault, severity, dest)
    return name, wsl_path(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faults", default=",".join(cf.FAULTS))
    ap.add_argument("--severities", default="1.0,0.5")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--refs", action="store_true", help="add the two severity-0 references")
    ap.add_argument("--tag", default="", help="inserted after tmp_cfg_ so a re-scaled table does not reuse old run names")
    a = ap.parse_args()
    lines = []
    if a.refs:
        dest = CFG_ROOT / f"{PREFIX}ref"
        if dest.exists():
            shutil.rmtree(dest)
        cf.make_config_dir("rooms_min_size", 0.0, dest)
        lines.append(f"{PREFIX}ref {wsl_path(dest)}")
        lines.append(f"{PREFIX}ref_default -")
    for fault in a.faults.split(","):
        for sev in [float(x) for x in a.severities.split(",")]:
            name, cfg = build(fault, sev, a.tag)
            lines.append(f"{name} {cfg}")
    a.out.write_text("\n".join(lines) + "\n", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
