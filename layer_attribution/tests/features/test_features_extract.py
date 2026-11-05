"""features_extract end to end on synthetic run directories whose snapshot caches are seeded (no spark_dsg needed)."""

import json
import os
import time

import numpy as np
import pandas as pd
import pytest
import scene_factory as sf

from layer_attribution import features as F
from layer_attribution import features_extract as FE
from layer_attribution import runs as R
from layer_attribution import snapshot as sn

SENSOR = dict(corrupted_input="depth", expected_first_layer="mesh", also_hits_directly="places")
BASE = dict(g=40, n_place=60, n_obj=20)


def manifest(fault, severity, seed):
    return dict(SENSOR, fault=fault, severity=severity, seed=seed)


def make_run(root, cache, name, arrays, man=None):
    d = root / name
    (d / "backend").mkdir(parents=True)
    (d / "backend" / "dsg_with_mesh.json").write_text("{}")
    if man is not None:
        (d / "injection_manifest.json").write_text(json.dumps(man))
    snap = sf.snapshot_from_arrays(arrays, name=name)
    snap.meta["src"] = sn.source_key(d)
    snap.save(sn.cache_path(d, cache))
    return d


@pytest.fixture()
def world(tmp_path):
    root, cache = tmp_path / "runs", tmp_path / "cache"
    root.mkdir()
    ref = sf.make_scene(0, **BASE)
    moved = sf.make_scene(0, jitter=0.02, relabel=0.2, drop_objects=2, **BASE)
    make_run(root, cache, "uh2_clean", ref)
    make_run(root, cache, "uh2_clean_b", ref)
    make_run(root, cache, "uh2_label_confusion_s03_r0", moved, manifest("label_confusion", 0.03, 0))
    make_run(root, cache, "uh2_depth_noise_s50", moved, manifest("depth_noise", 0.5, 0))
    make_run(root, cache, "uh2_depth_noise_s50_r1", moved, manifest("depth_noise", 0.5, 1))  # held-out seed
    return dict(root=root, cache=cache, ref=ref, moved=moved, tmp=tmp_path)


def run_cli(w, out_name, *extra):
    out = w["tmp"] / out_name
    code = FE.main(["--root", str(w["root"]), "--out", str(out), "--cache-dir", str(w["cache"]), "--min-age-s", "0", *extra])
    return code, out


def test_development_extraction_writes_the_wide_csv_and_its_side_files(world):
    code, out = run_cli(world, "f.csv")
    assert code == 0
    df = pd.read_csv(out)
    assert list(df.columns) == list(FE.RUN_COLUMNS) + list(F.REGISTRY)
    assert list(df["run"]) == ["uh2_clean", "uh2_clean_b", "uh2_label_confusion_s03_r0", "uh2_depth_noise_s50"]  # no held-out run
    assert list(df["split"]) == ["reference", "repeat", "benign", "dev"]
    assert list(df["fault"]) == ["none", "none", "label_confusion", "depth_noise"] and df["seed"].isna().tolist() == [True, True, False, False]
    ref_row = df.iloc[0]
    for n, s in F.REGISTRY.items():
        if s.kind == "reference_based":
            assert ref_row[n] == pytest.approx(s.ideal, abs=1e-9), n
    meta = pd.read_csv(out.with_name("f_meta.csv"))
    assert list(meta["name"]) == list(F.REGISTRY) and {"evidence_layer", "definition", "ideal", "tol_abs", "count_based"} <= set(meta.columns)
    assert list(pd.read_csv(out.with_name("f_errors.csv")).columns) == ["run", "feature", "error"]
    timing = pd.read_csv(out.with_name("f_timing.csv"))
    assert len(timing) == 4 and timing["snapshot_from_cache"].all() and (timing["features_computed"] == 99).all()


def test_csv_values_equal_a_direct_computation(world):
    _, out = run_cli(world, "f.csv")
    row = pd.read_csv(out).set_index("run").loc["uh2_depth_noise_s50"]
    direct, _ = F.compute(sf.snapshot_from_arrays(world["moved"]), sf.snapshot_from_arrays(world["ref"]))
    got = row[list(F.REGISTRY)].to_numpy(float)
    assert np.allclose(got, list(direct.values()), rtol=1e-12, atol=1e-12, equal_nan=True)


def test_test_split_needs_final_and_nothing_is_written_without_it(world):
    code, out = run_cli(world, "refused.csv", "--splits", "dev,test")
    assert code == 2 and not out.exists()
    code, out = run_cli(world, "final.csv", "--splits", "test", "--final")
    assert code == 0 and list(pd.read_csv(out)["run"]) == ["uh2_depth_noise_s50_r1"]
    assert pd.read_csv(out)["split"].tolist() == ["test"]


def test_bad_arguments_are_rejected(world):
    assert run_cli(world, "a.csv", "--splits", "dev,bogus")[0] == 2
    assert run_cli(world, "b.csv", "--only", "mesh_vertex_count_ratio,not_a_feature")[0] == 2
    assert not (world["tmp"] / "a.csv").exists() and not (world["tmp"] / "b.csv").exists()


def test_only_restricts_the_columns_in_registry_order(world):
    code, out = run_cli(world, "f.csv", "--only", "place_radius_mean,mesh_vertex_count_ratio")
    assert code == 0
    assert list(pd.read_csv(out).columns) == list(FE.RUN_COLUMNS) + ["mesh_vertex_count_ratio", "place_radius_mean"]
    assert len(pd.read_csv(out.with_name("f_meta.csv"))) == 2


def test_second_invocation_is_served_from_the_feature_cache(world):
    _, out1 = run_cli(world, "one.csv")
    _, out2 = run_cli(world, "two.csv")
    t2 = pd.read_csv(out2.with_name("two_timing.csv"))
    assert (t2["features_computed"] == 0).all() and (t2["features_cached"] == 99).all()
    assert out1.read_text() == out2.read_text()
    _, out3 = run_cli(world, "three.csv", "--no-feature-cache")
    assert (pd.read_csv(out3.with_name("three_timing.csv"))["features_computed"] == 99).all()


def test_feature_cache_follows_the_snapshot_content(world):
    run_cli(world, "one.csv")
    d = world["root"] / "uh2_depth_noise_s50"
    make_run_again = sf.snapshot_from_arrays(sf.make_scene(0, jitter=0.05, **BASE), name=d.name)
    (d / "backend" / "dsg_with_mesh.json").write_text("{ }")  # new source stat -> new snapshot key
    make_run_again.meta["src"] = sn.source_key(d)
    make_run_again.save(sn.cache_path(d, world["cache"]))
    _, out = run_cli(world, "two.csv")
    t = pd.read_csv(out.with_name("two_timing.csv")).set_index("run")
    assert t.at["uh2_depth_noise_s50", "features_computed"] == 99 and t.at["uh2_clean_b", "features_cached"] == 99


def test_unusable_run_is_reported_and_the_others_still_run(world):
    d = world["root"] / "uh2_depth_noise_s50"
    (d / "backend" / "dsg_with_mesh.json").write_text("not a scene graph")  # stale cache, cannot rebuild
    code, out = run_cli(world, "f.csv")
    assert code == 1
    assert "uh2_depth_noise_s50" not in set(pd.read_csv(out)["run"]) and len(pd.read_csv(out)) == 3
    err = pd.read_csv(out.with_name("f_errors.csv"))
    assert err["run"].tolist() == ["uh2_depth_noise_s50"] and err["feature"].tolist() == ["(snapshot)"]


def test_two_worker_processes_give_the_same_csv(world):
    _, one = run_cli(world, "j1.csv", "--jobs", "1", "--no-feature-cache")
    _, two = run_cli(world, "j2.csv", "--jobs", "2", "--no-feature-cache")
    assert one.read_text() == two.read_text()


def test_runs_written_moments_ago_are_skipped(world):
    old = time.time() - 3600
    for d in world["root"].iterdir():
        os.utime(d / "backend" / "dsg_with_mesh.json", (old, old))
    fresh = world["root"] / "uh2_depth_noise_s50" / "backend" / "dsg_with_mesh.json"
    os.utime(fresh, None)
    runs, ref, skipped = FE.select_runs(world["root"], R.DEV_SPLITS, min_age_s=600)
    assert [r.name for r in skipped] == ["uh2_depth_noise_s50"] and "uh2_depth_noise_s50" not in [r.name for r in runs] and ref.name == "uh2_clean"
    os.utime(world["root"] / "uh2_clean" / "backend" / "dsg_with_mesh.json", None)
    with pytest.raises(RuntimeError):
        FE.select_runs(world["root"], R.DEV_SPLITS, min_age_s=600)


def test_dry_run_lists_runs_and_writes_nothing(world, capsys):
    code, out = run_cli(world, "f.csv", "--dry-run")
    assert code == 0 and not out.exists()
    text = capsys.readouterr().out
    assert "uh2_depth_noise_s50 " in text and "uh2_depth_noise_s50_r1" not in text
