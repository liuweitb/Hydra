"""Regression: the ported battery reproduces research/signal_probe/feature_values.csv (27 development runs x 99 features).

Tolerance rtol 1e-6, atol 1e-9 (the old csv has 8 significant digits), except the documented exceptions below.
Needs real data: the extraction csv (LA_FEATURES_DEV_CSV or D:/hydra_work/results/features/features_dev.csv) for the full
comparison, cached schema-3 snapshots (LA_SNAPSHOT_CACHE, LA_RUNS_ROOT) for the recompute subset; skipped when absent.
"""

import os
import pathlib

import numpy as np
import pandas as pd
import pytest
import scene_factory as sf

from layer_attribution import features as F
from layer_attribution import snapshot as sn

OLD = sf.ROOT / "research" / "signal_probe" / "feature_values.csv"
RTOL, ATOL = 1e-6, 1e-9
# feature -> (rtol, cause); every other feature must match at RTOL
EXCEPTIONS = {
    "floor_z_time_range": (0.1, "first-seen ties are cut by vertex order (stable sort); the prototype used numpy's default sort, "
                                "whose order among ~180 equal stamps depends on numpy version and CPU (changed tie rule, 0.1-5 % effect)"),
    "floor_time_structure": (0.1, "ratio floor_z_time_range / floor_z_robust_sd, inherits the changed tie rule"),
}


def _new_csv():
    for p in (os.environ.get("LA_FEATURES_DEV_CSV"), "D:/hydra_work/results/features/features_dev.csv",
              "/mnt/d/hydra_work/results/features/features_dev.csv"):
        if p and pathlib.Path(p).is_file():
            return pathlib.Path(p)
    return None


def _old():
    if not OLD.is_file():
        pytest.skip("research/signal_probe/feature_values.csv not present")
    return pd.read_csv(OLD, index_col=0).T  # runs x features


def mismatches(new_rows, old):
    """(run, feature, new, old, cause) beyond tolerance; new_rows: run -> {feature: value}."""
    out = []
    for run in old.index:
        for f in old.columns:
            a, b = new_rows[run].get(f, np.nan), old.at[run, f]
            rtol = EXCEPTIONS.get(f, (RTOL, ""))[0]
            if np.isnan(a) and np.isnan(b):
                continue
            if np.isnan(a) != np.isnan(b) or abs(a - b) > ATOL + rtol * abs(b):
                out.append((run, f, a, b, EXCEPTIONS.get(f, (0, "none"))[1]))
    return out


def test_extraction_csv_reproduces_the_prototype_values():
    path = _new_csv()
    if path is None:
        pytest.skip("features_dev.csv not present")
    old, new = _old(), pd.read_csv(path, index_col="run")
    missing = [r for r in old.index if r not in new.index]
    assert not missing, f"runs missing from {path}: {missing}"
    assert list(old.columns) == [c for c in new.columns if c in old.columns] and len(old.columns) == 99
    bad = mismatches({r: new.loc[r].to_dict() for r in old.index}, old)
    assert not bad, f"{len(bad)} values differ:\n" + "\n".join(map(str, bad[:20]))


def test_documented_exceptions_are_small_changes_not_regressions():
    path = _new_csv()
    if path is None:
        pytest.skip("features_dev.csv not present")
    old, new = _old(), pd.read_csv(path, index_col="run")
    for f in EXCEPTIONS:
        rel = ((new.loc[old.index, f] - old[f]).abs() / old[f].abs()).to_numpy()
        assert np.nanmax(rel) < 0.1 and np.nanmedian(rel) < 0.02, (f, np.nanmax(rel), np.nanmedian(rel))


def test_reference_row_is_at_the_ideal_of_every_reference_based_feature():
    path = _new_csv()
    if path is None:
        pytest.skip("features_dev.csv not present")
    row = pd.read_csv(path, index_col="run").loc["uh2_clean"]
    for n, s in F.REGISTRY.items():
        if s.kind == "reference_based":
            assert row[n] == pytest.approx(s.ideal, abs=1e-9), n


@pytest.mark.parametrize("run", ["uh2_clean_b", "uh2_depth_noise_s03_r0", "uh2_label_confusion_s03_r0", "uh2_pose_drift_s20_r0"])
def test_recomputed_values_match_the_prototype(run):
    old = _old()
    try:
        ref_snap, snap = sn.Snapshot.open("uh2_clean"), sn.Snapshot.open(run)
    except sn.SnapshotError:
        pytest.skip("snapshot caches / run directories not reachable")
    with ref_snap, snap:
        values, errors = F.compute(snap, ref_snap)
    assert errors == {}
    bad = mismatches({run: values}, old.loc[[run]])
    assert not bad, bad
