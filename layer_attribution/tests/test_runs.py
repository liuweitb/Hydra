import json

import pytest

from layer_attribution import runs


def make_run(root, name, manifest=None, complete=True):
    d = root / name
    (d / "backend").mkdir(parents=True)
    if complete:
        (d / "backend" / "dsg_with_mesh.json").write_text("{}")
    if manifest is not None:
        (d / "injection_manifest.json").write_text(json.dumps(manifest))
    return d


def sensor(fault, severity, seed, first="mesh", also=None):
    return {"fault": fault, "severity": severity, "seed": seed, "expected_first_layer": first,
            "also_hits_directly": also, "corrupted_input": "depth"}


@pytest.fixture
def root(tmp_path):
    make_run(tmp_path, "uh2_clean")
    make_run(tmp_path, "uh2_clean_b")
    make_run(tmp_path, "uh2_depth_noise_s03_r0", sensor("depth_noise", 0.03, 0, also="places"))
    make_run(tmp_path, "uh2_depth_noise_s03_r1", sensor("depth_noise", 0.03, 1))
    make_run(tmp_path, "uh2_depth_noise_s50", sensor("depth_noise", 0.5, 0))
    make_run(tmp_path, "uh2_depth_noise_s50_r1", sensor("depth_noise", 0.5, 1))
    make_run(tmp_path, "uh2_depth_noise_s80_r2", sensor("depth_noise", 0.8, 2))
    cfg = {"fault": "rooms_min_size", "severity": 0.8, "seed": None, "expected_first_layer": "rooms",
           "corrupted_input": "config"}
    make_run(tmp_path, "uh2_cfg_rooms_min_size_s80", cfg)
    make_run(tmp_path, "uh2_cfg_rooms_min_size_s100", dict(cfg, severity=1.0))
    make_run(tmp_path, "uh2_pose_drift_s20_r0", None, complete=False)  # still running: no manifest, no dsg
    make_run(tmp_path, "tmp_cfg_ref")
    return tmp_path


def by_name(rs):
    return {r.name: r for r in rs}


def test_split_assignment(root):
    r = by_name(runs.index_runs(root))
    assert r["uh2_clean"].split == "reference" and r["uh2_clean_b"].split == "repeat"
    assert r["uh2_depth_noise_s03_r0"].split == "benign" and r["uh2_depth_noise_s03_r1"].split == "benign"
    assert r["uh2_depth_noise_s50"].split == "dev"  # legacy name, seed 0 from the manifest
    assert r["uh2_depth_noise_s50_r1"].split == "test" and r["uh2_depth_noise_s80_r2"].split == "test"
    assert r["uh2_cfg_rooms_min_size_s80"].split == "dev" and r["uh2_cfg_rooms_min_size_s100"].split == "test"
    assert r["uh2_cfg_rooms_min_size_s80"].kind == "config"
    assert r["uh2_pose_drift_s20_r0"].split == "unknown" and not r["uh2_pose_drift_s20_r0"].complete


def test_tmp_runs_are_skipped_by_default(root):
    assert "tmp_cfg_ref" not in by_name(runs.index_runs(root))
    assert "tmp_cfg_ref" in by_name(runs.index_runs(root, include_tmp=True))


def test_test_split_needs_final_flag(root):
    rs = runs.index_runs(root)
    with pytest.raises(runs.SplitViolation):
        runs.select(rs, splits=("dev", "test"))
    with pytest.raises(runs.SplitViolation):
        runs.select(rs, splits=runs.ALL_SPLITS)
    assert {r.name for r in runs.select(rs, splits=("test",), final=True)} == {
        "uh2_depth_noise_s50_r1", "uh2_depth_noise_s80_r2", "uh2_cfg_rooms_min_size_s100"}


def test_default_selection_is_development_only(root):
    names = {r.name for r in runs.select(runs.index_runs(root))}
    assert "uh2_depth_noise_s50_r1" not in names and "uh2_depth_noise_s80_r2" not in names
    assert {"uh2_clean", "uh2_depth_noise_s50", "uh2_depth_noise_s03_r1"} <= names
    assert "uh2_pose_drift_s20_r0" not in names  # splits exclude 'unknown'


def test_complete_only_and_fault_filter(root):
    rs = runs.index_runs(root)
    assert runs.select(rs, splits=("unknown",)) == []
    assert len(runs.select(rs, splits=("unknown",), complete_only=False)) == 1
    only = runs.select(rs, splits=("benign",), faults=["depth_noise"])
    assert {r.name for r in only} == {"uh2_depth_noise_s03_r0", "uh2_depth_noise_s03_r1"}


def test_manifest_fields(root):
    r = by_name(runs.index_runs(root))["uh2_depth_noise_s03_r0"]
    assert (r.expected_first_layer, r.also_hits_directly, r.seed, r.severity) == ("mesh", ("places",), 0, 0.03)
    assert r.dsg_path == root / "uh2_depth_noise_s03_r0" / "backend" / "dsg_with_mesh.json"


def test_config_dev_severities_match_the_launch_plan():
    from layer_attribution.pipeline import config_faults as cf

    plan = {}
    for row in cf.launch_plan():
        plan.setdefault(row["fault"], []).append(row["severity"])
    assert set(plan) == set(runs.DEV_CONFIG_SEVERITY)
    for fault, sevs in plan.items():
        assert runs.DEV_CONFIG_SEVERITY[fault] in sevs, fault  # one dev run per fault, the rest are test


def test_reference_run(root):
    assert runs.reference_run(runs.index_runs(root)).name == "uh2_clean"
    with pytest.raises(FileNotFoundError):
        runs.reference_run([])
