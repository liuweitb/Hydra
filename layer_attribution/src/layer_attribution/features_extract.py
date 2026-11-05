"""Run the feature battery over Hydra run directories and write one wide csv (one row per run).

  python -m layer_attribution.features_extract --root ~/runs/uh2 --out features_dev.csv
         [--splits reference,repeat,benign,dev] [--jobs N] [--only f1,f2] [--final] [--cache-dir DIR]

Runs come from runs.index_runs / runs.select, so the 'test' split is refused unless --final is given. Snapshots are cached in
--cache-dir (default snapshot.CACHE_ROOT); building one needs spark_dsg, loading a cached one needs numpy only. Every feature
is computed against the reference run (uh2_clean). Per (run, feature) values are cached next to the snapshots, keyed by both
snapshot sources and the feature's source hash.
Outputs: OUT (run, split, kind, fault, severity, seed, one column per feature), OUT_meta.csv (features.meta()),
OUT_errors.csv (run, feature, error; feature '(snapshot)' = run skipped), OUT_timing.csv.
Injected fault, severity and seed come from the run index and only label the rows; features never see them.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # --jobs is the only parallelism; BLAS threads would exceed the core budget

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pathlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

from layer_attribution import dag, features as F, runs as R, snapshot as S  # noqa: E402

RUN_COLUMNS = ("run", "split", "kind", "fault", "severity", "seed")
SPLIT_RANK = {s: i for i, s in enumerate(R.ALL_SPLITS)}
FAULT_RANK = {f: i for i, f in enumerate(list(dag.SENSOR_ACCEPT) + list(dag.CONFIG_ACCEPT))}
_STATE: dict = {}  # per worker process


# ------------------------------------------------------------------------------------------------ result cache
def snapshot_key(snap) -> str:
    """Identity of the stored snapshot content: schema plus source stat (path dropped, it differs between Windows and WSL)."""
    src = {k: v for k, v in snap.meta["src"].items() if k != "path"}
    return hashlib.sha1(json.dumps({"schema": snap.meta["schema"], "src": src}, sort_keys=True).encode()).hexdigest()[:16]


def _cache_file(cache_dir, run_name) -> pathlib.Path:
    return pathlib.Path(cache_dir).expanduser() / "feature_values" / f"{run_name}.json"


def _read_cache(path, run_key, ref_key) -> dict:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if d.get("run_key") != run_key or d.get("ref_key") != ref_key:
        return {}
    return {n: {"h": e["h"], "v": float("nan") if e["v"] is None else float(e["v"]), "e": e.get("e")}
            for n, e in d.get("values", {}).items()}


def _write_cache(path, run_key, ref_key, entries) -> None:
    doc = {"run_key": run_key, "ref_key": ref_key,
           "values": {n: {"h": e["h"], "v": None if math.isnan(e["v"]) else e["v"], "e": e.get("e")} for n, e in entries.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc), encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------ worker
def _init_worker(ref_path, cache_dir, names, use_cache) -> None:
    _STATE.clear()
    ref_snap = S.Snapshot.open(ref_path, cache_root=cache_dir)
    _STATE.update(ref_name=pathlib.Path(ref_path).name, ref_snap=ref_snap, ref_view=F.FeatureView(ref_snap),
                  ref_key=snapshot_key(ref_snap), cache_dir=cache_dir, names=list(names), use_cache=use_cache,
                  hashes=F.source_hashes())


def _close_worker() -> None:
    snap = _STATE.pop("ref_snap", None)
    if snap is not None:
        snap.close()
    _STATE.clear()


def _work(task) -> dict:
    """Features of one run against the worker's reference; failures come back as data so one run cannot stop the sweep."""
    name, path = task
    st = _STATE
    t0 = time.perf_counter()
    try:
        snap = S.Snapshot.open(path, cache_root=st["cache_dir"])
    except Exception as e:  # SnapshotError, unreadable cache directory, ...
        return {"run": name, "failed": f"{type(e).__name__}: {e}"}
    try:
        t1 = time.perf_counter()
        key = snapshot_key(snap)
        cf = _cache_file(st["cache_dir"], name)
        cached = _read_cache(cf, key, st["ref_key"]) if st["use_cache"] else {}
        todo = [n for n in st["names"] if cached.get(n, {}).get("h") != st["hashes"][n]]
        if todo:
            view = st["ref_view"] if name == st["ref_name"] else F.FeatureView(snap)
            vals, errs = F.compute(view, st["ref_view"], names=todo)
            for n in todo:
                cached[n] = {"h": st["hashes"][n], "v": vals[n], "e": errs.get(n)}
            if st["use_cache"]:
                try:
                    _write_cache(cf, key, st["ref_key"], cached)
                except OSError:
                    pass  # a read-only cache directory only costs recomputation
        t2 = time.perf_counter()
        return {"run": name, "values": {n: cached[n]["v"] for n in st["names"]},
                "errors": {n: cached[n]["e"] for n in st["names"] if cached[n].get("e")},
                "snapshot_s": t1 - t0, "snapshot_cached": bool(snap.from_cache), "features_s": t2 - t1,
                "n_computed": len(todo), "n_cached": len(st["names"]) - len(todo), "warnings": list(snap.meta.get("warnings", []))}
    except Exception as e:  # defensive: compute() already isolates single features
        return {"run": name, "failed": f"{type(e).__name__}: {e}"}
    finally:
        snap.close()


# ------------------------------------------------------------------------------------------------ driver
def _fmt(v) -> str:
    return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else repr(float(v))


def _order_key(run) -> tuple:
    return (SPLIT_RANK.get(run.split, 99), run.severity if run.severity == run.severity else -1.0,
            FAULT_RANK.get(run.fault, -1), -1 if run.seed is None else run.seed, run.name)


def select_runs(root, splits, final=False, min_age_s=0.0, now=None):
    """(runs to process, reference run, skipped as still being written). Raises runs.SplitViolation for 'test' without final."""
    index = R.index_runs(root)
    chosen = R.select(index, splits, final=final)
    ref = R.reference_run(index)
    now = time.time() if now is None else now

    def settled(r):
        return min_age_s <= 0 or now - r.dsg_path.stat().st_mtime >= min_age_s

    skipped = [r for r in chosen + [ref] if not settled(r)]
    if ref in skipped:
        raise RuntimeError(f"reference run {ref.name} was modified less than {min_age_s:.0f} s ago")
    return sorted((r for r in chosen if r not in skipped), key=_order_key), ref, skipped


def write_outputs(out, feature_names, runs_done, results, failed, timing) -> None:
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def stem(suffix):
        return out.with_name(out.stem + suffix + out.suffix)

    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(list(RUN_COLUMNS) + list(feature_names))
        for r in runs_done:
            w.writerow([r.name, r.split, r.kind, r.fault or "none", _fmt(r.severity), "" if r.seed is None else r.seed]
                       + [_fmt(results[r.name]["values"][n]) for n in feature_names])
    keep = set(feature_names)
    cols = ("name", "kind", "layer", "evidence_layer", "ideal", "tol_abs", "tol_rel", "count_based", "unit", "definition")
    with open(stem("_meta"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for m in F.meta():
            if m["name"] in keep:
                w.writerow(["" if m[c] is None else m[c] for c in cols])
    with open(stem("_errors"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(("run", "feature", "error"))
        for name, msg in failed.items():
            w.writerow((name, "(snapshot)", msg))
        for r in runs_done:
            for n, msg in results[r.name]["errors"].items():
                w.writerow((r.name, n, msg))
    with open(stem("_timing"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(("run", "snapshot_s", "snapshot_from_cache", "features_s", "features_computed", "features_cached"))
        for t in timing:
            w.writerow((t["run"], f"{t['snapshot_s']:.2f}", t["snapshot_cached"], f"{t['features_s']:.2f}", t["n_computed"], t["n_cached"]))


def summary(names, runs_done, results, timing) -> str:
    """NaN count per feature (features without NaN are not listed) and seconds per run."""
    nan = {n: sum(1 for r in runs_done if results[r.name]["values"][n] != results[r.name]["values"][n]) for n in names}
    lines = ["NaN per feature: " + (", ".join(f"{n}={c}" for n, c in nan.items() if c) or "none")]
    if timing:
        snap, feat = [t["snapshot_s"] for t in timing], [t["features_s"] for t in timing]
        lines.append(f"seconds per run: snapshot mean {sum(snap) / len(snap):.1f} max {max(snap):.1f}, "
                     f"features mean {sum(feat) / len(feat):.1f} max {max(feat):.1f}")
    return "\n".join(lines)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=pathlib.Path, default=S.RUNS_ROOT, help="directory of run directories")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--splits", default=",".join(R.DEV_SPLITS))
    ap.add_argument("--final", action="store_true", help="allow the held-out 'test' split (final evaluation only)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only", default=None, help="comma list of feature names")
    ap.add_argument("--cache-dir", type=pathlib.Path, default=S.CACHE_ROOT, help="snapshot cache; feature values go to <dir>/feature_values")
    ap.add_argument("--no-feature-cache", action="store_true")
    ap.add_argument("--min-age-s", type=float, default=120.0, help="skip runs whose json changed more recently (still being written)")
    ap.add_argument("--dry-run", action="store_true", help="print the selected runs and exit")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    splits = [s for s in args.splits.split(",") if s]
    bad = [s for s in splits if s not in R.ALL_SPLITS]
    if bad:
        print(f"unknown split(s) {bad}; known: {list(R.ALL_SPLITS)}", file=sys.stderr)
        return 2
    names = list(F.REGISTRY)
    if args.only:
        want = [n for n in args.only.split(",") if n]
        unknown = [n for n in want if n not in F.REGISTRY]
        if unknown:
            print(f"unknown feature(s) {unknown}", file=sys.stderr)
            return 2
        names = [n for n in names if n in set(want)]
    root = args.root.expanduser()
    try:
        runs_sel, ref, skipped = select_runs(root, splits, final=args.final, min_age_s=args.min_age_s)
    except (R.SplitViolation, FileNotFoundError, RuntimeError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    for r in skipped:
        print(f"skip {r.name}: json modified < {args.min_age_s:.0f} s ago (still being written?)", flush=True)
    print(f"{len(runs_sel)} runs ({', '.join(splits)}), reference {ref.name}, {len(names)} features, jobs {args.jobs}", flush=True)
    if args.dry_run:
        for r in runs_sel:
            print(f"  {r.name:36s} {r.split:9s} {r.fault or '-':16s} sev={r.severity} seed={r.seed}")
        return 0

    init = (ref.path, args.cache_dir, names, not args.no_feature_cache)
    tasks = [(r.name, r.path) for r in runs_sel]
    t_start = time.perf_counter()
    results, failed, timing = {}, {}, []

    def collect(i, res):
        if "failed" in res:
            failed[res["run"]] = res["failed"]
            print(f"[{i}/{len(tasks)}] {res['run']}: FAILED {res['failed']}", flush=True)
            return
        results[res["run"]] = res
        timing.append({k: res[k] for k in ("run", "snapshot_s", "snapshot_cached", "features_s", "n_computed", "n_cached")})
        n_nan = sum(1 for v in res["values"].values() if v != v)
        print(f"[{i}/{len(tasks)}] {res['run']}: snapshot {res['snapshot_s']:.1f}s ({'cache' if res['snapshot_cached'] else 'built'}), "
              f"features {res['features_s']:.1f}s ({res['n_computed']} computed, {res['n_cached']} cached), NaN {n_nan}, "
              f"errors {len(res['errors'])}", flush=True)

    if args.jobs <= 1:
        _init_worker(*init)
        try:
            for i, t in enumerate(tasks, 1):
                collect(i, _work(t))
        finally:
            _close_worker()
    else:
        S.Snapshot.open(ref.path, cache_root=args.cache_dir).close()  # build the reference cache once, before the workers need it
        with ProcessPoolExecutor(max_workers=args.jobs, initializer=_init_worker, initargs=init) as ex:
            for i, res in enumerate(ex.map(_work, tasks), 1):
                collect(i, res)

    done = [r for r in runs_sel if r.name in results]
    write_outputs(args.out, names, done, results, failed, timing)
    print(f"wrote {args.out} ({len(done)} runs x {len(names)} features, {len(failed)} failed) in {time.perf_counter() - t_start:.0f} s", flush=True)
    print(summary(names, done, results, timing), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
