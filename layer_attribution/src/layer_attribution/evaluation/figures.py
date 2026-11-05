#!/usr/bin/env python3
"""Figures from the evaluation outputs (needs matplotlib: WSL venv).

  python -m layer_attribution.evaluation.figures --predictions predictions_test.csv --queries queries.csv --out DIR
        [--summary summary_test.json] [--tag test]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LAYERS = ("mesh", "objects", "places", "rooms")
SENSOR = ["label_confusion", "label_dropout", "depth_noise", "depth_holes", "pose_drift", "frame_drop"]


def failure_curves(queries: pd.DataFrame, out: Path, tag: str):
    q = queries[queries["fault"].notna() & (queries["fault"] != "none")].copy()
    ben = q[q["split"] == "benign"]
    hi = float(ben["failure_rate"].max()) if len(ben) else 0.0
    faults = [f for f in SENSOR if f in set(q["fault"])] + sorted(set(q["fault"]) - set(SENSOR))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, kind in zip(axes, ("sensor", "config")):
        sub = q[(q["kind"] == kind) & (q["split"] != "benign")]
        for f in [x for x in faults if x in set(sub["fault"])]:
            g = sub[sub["fault"] == f].groupby("severity")["failure_rate"].agg(["mean", "min", "max"])
            ax.errorbar(g.index, g["mean"], yerr=[g["mean"] - g["min"], g["max"] - g["mean"]], marker="o", capsize=2, label=f)
        ax.axhspan(0, hi, color="0.85", zorder=0, label=f"benign band (max {hi:.2f})")
        ax.set_title(f"{kind} faults"), ax.set_xlabel("severity"), ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("query failure rate")
    fig.tight_layout()
    fig.savefig(out / f"failure_rate_vs_severity_{tag}.png", dpi=130)
    plt.close(fig)


def layer_scores(pred: pd.DataFrame, method: str, out: Path, tag: str):
    p = pred[pred["method"] == method].copy()
    if p.empty:
        return
    p = p.sort_values(["kind", "fault", "severity", "run"])
    scores = np.array([[json.loads(s)[l] for l in LAYERS] for s in p["scores"]], dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 0.22 * len(p) + 1.5))
    im = ax.imshow(np.log10(np.maximum(scores, 0.1)), aspect="auto", cmap="magma", vmin=-1, vmax=2)
    ax.set_xticks(range(4), LAYERS)
    ax.set_yticks(range(len(p)), [f"{r.fault} {r.severity:g}{'' if pd.isna(r.seed) else f' s{int(r.seed)}'}{' (masked)' if r.masked else ''}"
                                  for r in p.itertuples()], fontsize=6)
    for i, row in enumerate(scores):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.1f}" if v < 100 else ">100", ha="center", va="center", fontsize=5, color="w" if v < 30 else "k")
    fig.colorbar(im, label="log10 layer score (>=1 deviates)", shrink=0.6)
    ax.set_title(method, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"layer_scores_{method}_{tag}.png", dpi=130)
    plt.close(fig)


def confusions(summary: dict, out: Path, tag: str):
    for kind, block in summary["blocks"].items():
        _confusion_block(block, out, f"{kind}_{tag}")


def _confusion_block(summary: dict, out: Path, tag: str):
    methods = [m for m in ("dag_rb", "naive_rb", "dag_rf", "naive_rf", "symptom") if m in summary["confusion"]]
    if not methods:
        return
    cols = list(LAYERS) + ["none"]
    fig, axes = plt.subplots(1, len(methods), figsize=(3.3 * len(methods), 3), squeeze=False)
    for ax, m in zip(axes[0], methods):
        t = summary["confusion"][m]
        rows = list(t)
        a = np.array([[t[r].get(c, 0) for c in cols] for r in rows], dtype=float)
        ax.imshow(a, cmap="Blues")
        ax.set_xticks(range(len(cols)), cols, rotation=45, fontsize=7)
        ax.set_yticks(range(len(rows)), rows, fontsize=7)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                ax.text(j, i, int(a[i, j]), ha="center", va="center", fontsize=7, color="k" if a[i, j] < a.max() * 0.6 else "w")
        ax.set_title(m, fontsize=9)
    axes[0][0].set_ylabel("injected (accept set)")
    fig.tight_layout()
    fig.savefig(out / f"confusion_{tag}.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--summary", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tag", default="dev")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    failure_curves(pd.read_csv(a.queries), a.out, a.tag)
    pred = pd.read_csv(a.predictions)
    for m in ("dag_rb", "dag_rf"):
        layer_scores(pred, m, a.out, a.tag)
    if a.summary is not None:
        confusions(json.loads(a.summary.read_text()), a.out, a.tag)


if __name__ == "__main__":
    main()
