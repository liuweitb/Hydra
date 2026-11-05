"""Feature centers and thresholds from benign runs; strengths of a run's features.

A feature value x deviates by d = |x - center|. Reference-based features have an ideal value (the run compared with itself);
reference-free features use the median over the reference and benign runs. The threshold of a feature is
max(margin * max benign deviation, floor): the benign maximum is the largest deviation seen without a real fault,
the floor (tol_abs + tol_rel * |center|) is the smallest change worth calling a move. Strength = d / threshold;
a feature "moves" when its strength is >= 1. Hydra is deterministic, so benign runs (severity <= 0.05) stand in for natural variation.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MARGIN = 1.5
STRENGTH_CAP = 1e3  # finite scores, a 0/tiny threshold must not produce inf
META_COLUMNS = ("kind", "layer", "ideal", "tol_abs", "tol_rel", "count_based")


def check_meta(meta: pd.DataFrame) -> pd.DataFrame:
    """meta indexed by feature name with columns kind, layer (evidence layer or None), ideal, tol_abs, tol_rel, count_based."""
    missing = [c for c in META_COLUMNS if c not in meta.columns]
    if missing:
        raise ValueError(f"feature meta lacks columns {missing}")
    bad = set(meta["kind"]) - {"reference_based", "reference_free"}
    if bad:
        raise ValueError(f"unknown feature kinds {sorted(bad)}")
    if meta.index.has_duplicates:
        raise ValueError("duplicate feature names in meta")
    return meta


@dataclass
class Calibration:
    center: pd.Series
    threshold: pd.Series
    meta: pd.DataFrame
    info: dict = field(default_factory=dict)

    def strengths(self, table: pd.DataFrame) -> pd.DataFrame:
        """Rows of `table` (run x feature) -> |x - center| / threshold, NaN where the feature or threshold is missing."""
        cols = [c for c in table.columns if c in self.center.index]
        dev = (table[cols].astype(float) - self.center[cols]).abs()
        return (dev / self.threshold[cols]).clip(upper=STRENGTH_CAP)


def fit(table: pd.DataFrame, meta: pd.DataFrame, benign, reference: str, margin: float = MARGIN) -> Calibration:
    """Calibrate on the benign runs (names in `benign`) of a run x feature table. The reference row supplies the
    reference-free centers together with the benign rows; it never enters a threshold."""
    meta = check_meta(meta)
    benign = list(benign)
    if not benign:
        raise ValueError("no benign runs to calibrate on")
    feats = [f for f in meta.index if f in table.columns]
    ben = table.loc[benign, feats].astype(float)
    ref_free = [f for f in feats if meta.at[f, "kind"] == "reference_free"]
    center = pd.Series(np.nan, index=feats, dtype=float)
    for f in feats:
        center[f] = meta.at[f, "ideal"] if meta.at[f, "kind"] == "reference_based" else np.nan
    pool = table.loc[[reference] + benign, ref_free].astype(float)
    center[ref_free] = pool.median()
    dev = (ben - center).abs()
    floor = meta.loc[feats, "tol_abs"].astype(float) + meta.loc[feats, "tol_rel"].astype(float) * center.abs()
    tiny = 1e-9 * np.maximum(1.0, center.abs())
    threshold = np.maximum(np.maximum(margin * dev.max(skipna=True), floor), tiny)
    unusable = center.isna() | dev.isna().all()  # no ideal / no finite benign value: the feature cannot be calibrated
    threshold[unusable] = np.nan
    return Calibration(center, threshold, meta.loc[feats], dict(margin=margin, benign=benign, reference=reference))


def usable_features(cal: Calibration, mode: str, exclude_count_based: bool = False) -> dict:
    """{evidence layer: [feature names]} for mode 'rb' (reference_based), 'rf' (reference_free) or 'both'."""
    kinds = {"rb": ("reference_based",), "rf": ("reference_free",), "both": ("reference_based", "reference_free")}[mode]
    m = cal.meta
    ok = m["kind"].isin(kinds) & m["layer"].notna() & cal.threshold.reindex(m.index).notna()
    if exclude_count_based:
        ok &= ~m["count_based"].astype(bool)
    out = {}
    for name in m.index[ok]:
        out.setdefault(m.at[name, "layer"], []).append(name)
    return out
