#!/usr/bin/env python3
"""Fault sweep driver: inject one fault (Windows, hard-linked dataset) -> run Hydra (WSL) -> keep the run directory.

Resume-safe (finished runs are skipped). Run with the Windows venv that has numpy/pillow/tifffile/scipy/pyyaml:

  python -m layer_attribution.sweep.orchestrate --spec configs/sweep_main.json --workers 7 [--dry-run]
"""

import argparse
import itertools
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
INJECT = PROJECT / "src" / "layer_attribution" / "injection" / "inject.py"
_lock = threading.Lock()


def to_wsl(p):
    p = Path(p).resolve()
    return "/mnt/" + p.drive.rstrip(":").lower() + "/" + "/".join(p.parts[1:])


def job_name(scene, fault, severity, seed):
    return f"{scene}_{fault}_s{round(severity * 100):02d}_r{seed}"


def legacy_name(scene, fault, severity, seed):
    """First sweep (seed 0) used names without the seed suffix."""
    return f"{scene}_{fault}_s{round(severity * 100):02d}" if seed == 0 else None


def enumerate_jobs(spec):
    jobs, seen = [], set()
    cfg = spec.get("config_faults")
    if cfg:  # deterministic on clean data: one run per (fault, severity), no seed; queued first, there are few
        plan = cfg if isinstance(cfg, list) else [dict(fault=f, severities=cfg["severities"]) for f in cfg["faults"]]
        for entry in plan:
            for sev in entry["severities"]:
                jobs.append(dict(type="config", fault=entry["fault"], severity=sev, seed=0, role="config", legacy=None,
                                 name=f"{spec['scene']}_cfg_{entry['fault']}_s{round(sev * 100):02d}"))
    for g in spec.get("groups", []):
        for seed, sev, fault in itertools.product(g["seeds"], g["severities"], g.get("faults", spec["faults"])):
            if (fault, sev, seed) in seen:
                continue
            seen.add((fault, sev, seed))
            jobs.append(dict(type="sensor", fault=fault, severity=sev, seed=seed, role=g["role"],
                             name=job_name(spec["scene"], fault, sev, seed),
                             legacy=legacy_name(spec["scene"], fault, sev, seed)))
    return jobs


def wsl(spec, *args):
    return subprocess.run(["wsl", "-d", spec["wsl"]["distro"], "--", *args], capture_output=True, text=True)


def run_exists(spec, name):
    return wsl(spec, "test", "-f", f"{spec['wsl']['runs_root']}/{name}/backend/dsg_with_mesh.json").returncode == 0


def inject_cmd(spec, job, dst):
    cmd = [sys.executable, str(INJECT), spec["source_dataset"], str(dst), "--fault", job["fault"],
           "--severity", str(job["severity"]), "--seed", str(job["seed"])]
    if job["fault"].startswith("label_"):
        cmd += ["--label-space", spec["label_space"]]
    if job["fault"] == "label_confusion":
        cmd += ["--param", "confusion_map=" + json.dumps(spec["confusion_map"])]
    return cmd


def log_status(path, rec):
    rec["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    line = json.dumps(rec)
    with _lock:
        print(line, flush=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def run_job(spec, job, opts):
    name = job["name"] + opts.name_suffix
    rec = dict(name=name, fault=job["fault"], severity=job["severity"], seed=job["seed"], role=job["role"])
    t0 = time.time()
    try:
        for n in (name, job["legacy"] if not opts.name_suffix else None):
            if n and run_exists(spec, n):
                rec.update(status="skipped_done", existing=n)
                return log_status(opts.status_file, rec)
        cfg_arg = "-"
        if job["type"] == "config":
            from layer_attribution.pipeline import config_faults

            dataset = Path(spec["source_dataset"])
            cfg_dir = Path(spec["config_root"]) / job["name"]
            if not cfg_dir.exists():
                config_faults.make_config_dir(job["fault"], job["severity"], cfg_dir)
            cfg_arg = to_wsl(cfg_dir)
        else:
            root = Path(spec["dataset_root"])
            dataset = root / job["name"]
            if not (dataset / "injection_manifest.json").exists():
                tmp = root / (job["name"] + ".tmp")
                if tmp.exists():
                    shutil.rmtree(tmp)  # partial output of an interrupted attempt
                r = subprocess.run(inject_cmd(spec, job, tmp), capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError("inject failed: " + (r.stderr or r.stdout)[-400:])
                if dataset.exists():
                    shutil.rmtree(dataset)
                tmp.rename(dataset)
        rec["inject_s"] = round(time.time() - t0)
        r = wsl(spec, "bash", to_wsl(PROJECT / "scripts" / "run_job.sh"), name, to_wsl(dataset), cfg_arg,
                str(opts.max_steps) if opts.max_steps else "-")
        rec.update(status="ok" if r.returncode == 0 else "hydra_failed", rc=r.returncode,
                   seconds=round(time.time() - t0), tail=(r.stdout + r.stderr)[-200:].strip())
        if r.returncode == 0 and opts.delete_datasets and job["type"] == "sensor":
            shutil.rmtree(dataset)
    except Exception as e:  # keep the sweep going; the status file records it
        rec.update(status="error", error=str(e)[-400:], seconds=round(time.time() - t0))
    log_status(opts.status_file, rec)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="first N jobs only (smoke tests)")
    ap.add_argument("--only", default=None, help="comma-separated job names")
    ap.add_argument("--max-steps", type=int, default=None, help="truncate Hydra runs (smoke tests only)")
    ap.add_argument("--name-suffix", default="", help="appended to run names, e.g. _smoke")
    ap.add_argument("--delete-datasets", action="store_true", help="delete the injected dataset after a successful run")
    ap.add_argument("--status-file", type=Path, default=None)
    opts = ap.parse_args()
    spec = json.loads(opts.spec.read_text(encoding="utf-8"))
    jobs = enumerate_jobs(spec)
    if opts.only:
        keep = set(opts.only.split(","))
        jobs = [j for j in jobs if j["name"] in keep]
    if opts.limit:
        jobs = jobs[: opts.limit]
    opts.status_file = opts.status_file or Path("D:/hydra_work/sweep_logs") / f"status_{opts.spec.stem}.jsonl"
    opts.status_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} jobs, {opts.workers} workers, status -> {opts.status_file}", flush=True)
    if opts.dry_run:
        for j in jobs:
            print(j["type"], j["role"], j["name"], "(legacy " + j["legacy"] + ")" if j["legacy"] else "")
        return
    with ThreadPoolExecutor(max_workers=opts.workers) as pool:
        list(pool.map(lambda j: run_job(spec, j, opts), jobs))
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
