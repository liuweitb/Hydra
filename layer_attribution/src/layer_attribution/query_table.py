#!/usr/bin/env python3
"""Per-run query outcomes: one csv row per run (failure rate, failure-mode counts), optional per-query csv.

WSL only (spark_dsg). The reference graph is uh2_clean; queries come from it, never from the run under test.

  python -m layer_attribution.query_table --root ~/runs/uh2 --out runs_queries.csv [--per-query q.csv]
         [--splits reference,repeat,benign,dev] [--final]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from layer_attribution import queries as Q
from layer_attribution import runs as R
from layer_attribution.query_views import view_from_dsg

META = ("run", "split", "kind", "fault", "severity", "seed")


def row_for(run, ev) -> dict:
    s, st = ev.summary(), ev.summary(strict=True)
    row = dict(run=run.name, split=run.split, kind=run.kind, fault=run.fault, severity=run.severity, seed=run.seed,
               n_queries=s["n_queries"], n_fail=s["n_fail"], failure_rate=s["failure_rate"],
               strict_n_fail=st["n_fail"], strict_failure_rate=st["failure_rate"])
    row.update({f"n_{m}": c for m, c in s["modes"].items()})
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("~/runs/uh2").expanduser())
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-query", type=Path, default=None)
    ap.add_argument("--splits", default=",".join(R.DEV_SPLITS))
    ap.add_argument("--final", action="store_true", help="allow the held-out 'test' split")
    args = ap.parse_args(argv)

    index = R.index_runs(args.root)
    todo = R.select(index, splits=args.splits.split(","), final=args.final)
    ref_run = R.reference_run(index)
    ref = view_from_dsg(ref_run.dsg_path, name=ref_run.name)
    queries = Q.build_queries(ref)
    print(f"reference {ref_run.name}: {len(queries)} queries; {len(todo)} runs", flush=True)

    rows, per_query = [], []
    for run in todo:
        view = ref if run.name == ref_run.name else view_from_dsg(run.dsg_path, name=run.name)
        ev = Q.evaluate(ref, view, queries=queries)
        rows.append(row_for(run, ev))
        if args.per_query is not None:
            per_query += [dict(r, split=run.split, fault=run.fault, severity=run.severity, seed=run.seed) for r in ev.rows()]
        print(f"{run.name:34s} {run.split:9s} fail {rows[-1]['n_fail']:3d}/{rows[-1]['n_queries']} = {rows[-1]['failure_rate']:.3f}",
              flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    if args.per_query is not None:
        args.per_query.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_query).to_csv(args.per_query, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
