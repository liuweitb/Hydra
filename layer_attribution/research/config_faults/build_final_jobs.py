#!/usr/bin/env python3
"""Config dirs and a jobs file for the FULL-length config-fault runs (recommended faults x launch severities, clean data).

  python build_final_jobs.py --out jobs_final.txt [--cfg-root DIR]
Each jobs line is `NAME CONFIG_PATH` (WSL path). One run per line, for example:
  bash scripts/run_job.sh NAME /mnt/d/hydra_work/data/uhumans2_office_mp3d CONFIG_PATH -        (- = all frames)
Afterwards check every run: python -m layer_attribution.pipeline.config_faults --verify RUN_DIR CONFIG_DIR
"""

import argparse
import shutil
from pathlib import Path

from layer_attribution.pipeline import config_faults as cf

CFG_ROOT = Path("D:/hydra_work/results/config_faults/final_cfg")


def wsl_path(p: Path) -> str:
    p = p.resolve()
    return "/mnt/" + p.drive.rstrip(":").lower() + "/" + "/".join(p.parts[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cfg-root", type=Path, default=CFG_ROOT)
    a = ap.parse_args()
    lines = []
    for row in cf.launch_plan():
        dest = a.cfg_root / row["name"]
        if dest.exists():
            shutil.rmtree(dest)
        cf.make_config_dir(row["fault"], row["severity"], dest)
        lines.append(f"{row['name']} {wsl_path(dest)}")
    a.out.write_text("\n".join(lines) + "\n", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
