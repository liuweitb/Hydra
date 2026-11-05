"""Config faults: rule maths, config-dir contents, guard against silently ignored keys, effective-config parser.

Runs without Hydra. Tests that need Hydra's config/ tree skip if it is missing from the checkout.
"""

import json
import math
import re
from pathlib import Path

import pytest
import yaml

from layer_attribution.pipeline import config_faults as cf

BASE = cf.DEFAULT_BASE
HEADER = Path(__file__).parent / "fixtures" / "hydra_info_header_clean.txt"
needs_base = pytest.mark.skipif(not (BASE / "datasets" / "uhumans2.yaml").is_file(), reason="Hydra config/ not found")
RULES = [(n, k, r) for n, f in cf.FAULTS.items() for k, r in f.rules.items()]
SEVERITIES = [i / 20 for i in range(21)]


def load_ds(d):
    return yaml.safe_load((Path(d) / "datasets" / "uhumans2.yaml").read_text(encoding="utf-8"))


def get(tree, dotted):
    for p in dotted.split("."):
        tree = tree[p]
    return tree


def mini_base(tmp_path, dataset_tree):
    base = tmp_path / "base"
    (base / "datasets").mkdir(parents=True)
    (base / "label_spaces").mkdir()
    (base / "datasets" / "uhumans2.yaml").write_text(yaml.safe_dump(dataset_tree))
    (base / "label_spaces" / "uhumans2_office_label_space.yaml").write_text("total_semantic_labels: 21\n")
    return base


# ---------------------------------------------------------------- rules

@pytest.mark.parametrize("name,key,rule", RULES, ids=[f"{n}:{k.split('.')[-1]}" for n, k, _ in RULES])
def test_rule_endpoints_monotone_and_typed(name, key, rule):
    assert rule(0.0) == rule.default and rule(1.0) == rule.extreme
    vals = [rule(s) for s in SEVERITIES]
    lo, hi = sorted((rule.default, rule.extreme))
    assert all(lo <= v <= hi for v in vals)
    steps = [b - a for a, b in zip(vals, vals[1:])]
    assert all(d >= 0 for d in steps) if rule.extreme > rule.default else all(d <= 0 for d in steps)
    assert len(set(vals)) > 3  # the knob actually moves
    assert all(isinstance(v, int) for v in vals) if rule.integer else all(isinstance(v, float) for v in vals)


def test_geo_rule_has_constant_ratio():
    r = cf.Rule(10, 1000, "geo")
    assert r(0.5) == pytest.approx(100.0)
    assert r(0.25) / r(0.0) == pytest.approx(r(0.75) / r(0.5), rel=1e-3)


def test_integer_rule_rounds_half_up():
    assert cf.Rule(0, 5, integer=True)(0.5) == 3


def test_rule_validation():
    with pytest.raises(ValueError):
        cf.Rule(1, 2, "cubic")
    with pytest.raises(ValueError):
        cf.Rule(0, 2, "geo")


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf"), "0.5", None, True])
def test_bad_severity_rejected(bad, tmp_path):
    with pytest.raises(ValueError):
        cf.values_at("rooms_min_size", bad)
    with pytest.raises(ValueError):
        cf.make_config_dir("rooms_min_size", bad, tmp_path / "x")


def test_unknown_fault(tmp_path):
    with pytest.raises(ValueError, match="unknown config fault"):
        cf.make_config_dir("nope", 0.5, tmp_path / "x")


def test_table_shape():
    assert set(cf.LAYER_ORDER) >= {f.layer for f in cf.FAULTS.values()}
    for f in cf.FAULTS.values():
        assert f.layer in cf.LAYER_ORDER and f.rules and f.note
        assert all(cf.LAYER_ORDER.index(d) > cf.LAYER_ORDER.index(f.layer) for d in f.downstream)


def test_recommended_list_covers_objects_and_rooms():
    assert 3 <= len(cf.RECOMMENDED) <= 5
    layers = {cf.FAULTS[n].layer for n in cf.RECOMMENDED}
    assert {"objects", "rooms"} <= layers


@pytest.mark.parametrize("name", cf.RECOMMENDED)
def test_launch_severities_are_three_distinct_nonzero_settings(name):
    spec = cf.FAULTS[name]
    sev = list(spec.launch_severities)
    assert len(sev) == 3 and sev == sorted(set(sev)) and 0 < sev[0] and sev[-1] <= 1
    settings = [tuple(cf.values_at(name, s).values()) for s in sev]
    default = tuple(r.default for r in spec.rules.values())
    assert len(set(settings)) == 3 and default not in settings


def test_launch_plan_matches_the_table():
    plan = cf.launch_plan()
    assert len(plan) == 3 * len(cf.RECOMMENDED)
    names = [p["name"] for p in plan]
    assert len(set(names)) == len(names)
    for p in plan:
        spec = cf.FAULTS[p["fault"]]
        assert re.fullmatch(r"uh2_cfg_[a-z_]+_s\d{2,3}", p["name"])
        assert p["values"] == cf.values_at(p["fault"], p["severity"])
        assert p["layer"] == spec.layer and p["downstream"] == list(spec.downstream)


def test_run_name_scheme():
    assert cf.run_name("rooms_min_size", 0.6) == "uh2_cfg_rooms_min_size_s60"
    assert cf.run_name("rooms_min_size", 1.0) == "uh2_cfg_rooms_min_size_s100"
    assert cf.run_name("rooms_min_size", 0.05, prefix="x") == "x_rooms_min_size_s05"
    with pytest.raises(ValueError):
        cf.run_name("nope", 0.5)
    with pytest.raises(ValueError):
        cf.run_name("rooms_min_size", 1.5)


# ---------------------------------------------------------------- config dir

@needs_base
@pytest.mark.parametrize("name,key,rule", RULES, ids=[f"{n}:{k.split('.')[-1]}" for n, k, _ in RULES])
def test_table_default_matches_repo_config(name, key, rule):
    old = get(load_ds(BASE), key)
    assert math.isclose(old, rule.default) and isinstance(old, int) == rule.integer


@needs_base
@pytest.mark.parametrize("fault", sorted(cf.FAULTS))
def test_severity_zero_reproduces_base(fault, tmp_path):
    dest = cf.make_config_dir(fault, 0.0, tmp_path / "cfg")
    assert load_ds(dest) == load_ds(BASE)
    assert cf.diff_configs(BASE, dest) == []


@needs_base
@pytest.mark.parametrize("fault", sorted(cf.FAULTS))
@pytest.mark.parametrize("severity", [0.5, 1.0])
def test_only_declared_keys_change(fault, severity, tmp_path):
    dest = cf.make_config_dir(fault, severity, tmp_path / "cfg")
    spec = cf.FAULTS[fault]
    assert cf.diff_configs(BASE, dest) == sorted(spec.rules)
    tree = load_ds(dest)
    for key, value in cf.values_at(fault, severity).items():
        assert get(tree, key) == value
        assert get(tree, key) != get(load_ds(BASE), key)


@needs_base
def test_config_dir_is_a_complete_copy(tmp_path):
    dest = cf.make_config_dir("rooms_min_size", 0.5, tmp_path / "cfg")
    base_files = {p.relative_to(BASE) for p in BASE.rglob("*") if p.is_file()}
    dest_files = {p.relative_to(dest) for p in dest.rglob("*") if p.is_file()}
    assert dest_files == base_files | {Path(cf.MANIFEST)}
    for rel in base_files - {Path("datasets/uhumans2.yaml")}:
        assert (dest / rel).read_bytes() == (BASE / rel).read_bytes()
    assert b"\r" not in (dest / "datasets" / "uhumans2.yaml").read_bytes()


@needs_base
def test_manifest(tmp_path):
    dest = cf.make_config_dir("places_compression", 0.5, tmp_path / "cfg")
    m = json.loads((dest / cf.MANIFEST).read_text())
    assert m["fault"] == "places_compression" and m["severity"] == 0.5 and m["corrupted_input"] == "config"
    assert m["expected_first_layer"] == "places" and m["downstream_layers"] == ["rooms", "buildings"]
    key = "frontend.freespace_places.graph.compression_distance_m"
    assert m["params"] == {key: {"default": 1.5, "value": 3.0}}
    assert m["n_keys_changed_vs_base"] == 1 and m["layer_order"] == cf.LAYER_ORDER
    assert m["base_dataset_yaml_sha256"] != m["dataset_yaml_sha256"]


@needs_base
def test_dest_must_be_new_or_empty_and_apart_from_base(tmp_path):
    full = tmp_path / "full"
    full.mkdir()
    (full / "x").write_text("1")
    with pytest.raises(ValueError, match="not empty"):
        cf.make_config_dir("rooms_min_size", 0.5, full)
    empty = tmp_path / "empty"
    empty.mkdir()
    cf.make_config_dir("rooms_min_size", 0.5, empty)
    with pytest.raises(ValueError, match="overlap"):
        cf.make_config_dir("rooms_min_size", 0.5, BASE / "sub")
    with pytest.raises(ValueError, match="overlap"):
        cf.make_config_dir("rooms_min_size", 0.5, BASE)


def test_missing_base_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        cf.make_config_dir("rooms_min_size", 0.5, tmp_path / "o", base_config_dir=tmp_path / "nothing")


def test_key_absent_from_base_is_an_error_not_a_new_key(tmp_path):
    tree = {"backend": {"update_functors": {"rooms": {"room_finder": {"min_dilation_m": 0.5}}}}}
    base = mini_base(tmp_path, tree)
    with pytest.raises(KeyError, match="min_room_size"):
        cf.make_config_dir("rooms_min_size", 0.5, tmp_path / "o", base_config_dir=base)
    with pytest.raises(KeyError, match="frontend"):
        cf.make_config_dir("objects_min_size", 0.5, tmp_path / "o2", base_config_dir=base)


def test_misspelled_table_key_is_caught(tmp_path, monkeypatch):
    tree = {"frontend": {"objects": {"min_cluster_size": 40}}}
    base = mini_base(tmp_path, tree)
    bad = cf.ConfigFault("objects", {"frontend.objects.min_cluster_sizee": cf.Rule(40, 1000, "geo", integer=True)}, note="x")
    monkeypatch.setitem(cf.FAULTS, "typo", bad)
    with pytest.raises(KeyError, match="sizee"):
        cf.make_config_dir("typo", 0.5, tmp_path / "o", base_config_dir=base)


def test_table_default_out_of_sync_or_wrong_type_is_an_error(tmp_path):
    base = mini_base(tmp_path, {"frontend": {"objects": {"min_cluster_size": 41}}})
    with pytest.raises(ValueError, match="!= base value"):
        cf.make_config_dir("objects_min_size", 0.5, tmp_path / "o", base_config_dir=base)
    base2 = mini_base(tmp_path / "b2", {"frontend": {"objects": {"min_cluster_size": 40.0}}})
    with pytest.raises(ValueError, match="!= base value"):
        cf.make_config_dir("objects_min_size", 0.5, tmp_path / "o2", base_config_dir=base2)
    base3 = mini_base(tmp_path / "b3", {"frontend": {"objects": {"min_cluster_size": "forty"}}})
    with pytest.raises(TypeError):
        cf.make_config_dir("objects_min_size", 0.5, tmp_path / "o3", base_config_dir=base3)


def test_cli_plan_prints_one_json_line_per_run(capsys):
    cf.main(["--plan"])
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert rows == cf.launch_plan()


def test_cli(tmp_path, capsys):
    cf.main(["--list"])
    out = capsys.readouterr().out
    assert all(n in out for n in cf.FAULTS)
    if (BASE / "datasets" / "uhumans2.yaml").is_file():
        cf.main(["--fault", "rooms_min_size", "--severity", "1", "--dest", str(tmp_path / "cli")])
        assert "400" in capsys.readouterr().out


# ---------------------------------------------------------------- effective config echo of a real Hydra run

@pytest.fixture(scope="module")
def echo():
    return cf.read_effective_config(HEADER)


def test_echo_parses_known_values(echo):
    assert cf.find_effective(echo, "backend.update_functors.rooms.room_finder.min_dilation_m") == (
        ("backend", "update_functors", "rooms", "room_finder", "min_dilation_m"), ("0.5", False))
    path, (val, is_default) = cf.find_effective(echo, "frontend.objects.min_cluster_size")
    assert path == ("frontend", "objects", "clustering", "min_cluster_size") and (val, is_default) == ("40", True)
    assert cf.find_effective(echo, "frontend.freespace_places.gvd.max_distance_m")[1] == ("4.5", False)
    assert cf.find_effective(echo, "backend.update_functors.rooms.room_finder.dilation_threshold_mode")[1] == ("PLATEAU", False)


def test_every_table_key_resolves_in_the_echo(echo):
    for _, key, rule in RULES:
        _, (val, _) = cf.find_effective(echo, key)
        assert math.isclose(float(val), rule.default, rel_tol=1e-6)


def test_ambiguous_or_unknown_key_raises(echo):
    with pytest.raises(KeyError, match="matches 3"):
        cf.find_effective(echo, "frontend.min_cluster_size")
    with pytest.raises(KeyError, match="matches 0"):
        cf.find_effective(echo, "frontend.objects.min_cluster_sizee")


def test_check_effective_flags_ignored_override(tmp_path):
    expected = {"frontend.objects.min_cluster_size": 200}
    assert cf.check_effective(HEADER, expected)["frontend.objects.min_cluster_size"] == ("40", False)
    text = HEADER.read_text()
    patched = text.replace("      min_cluster_size:       40 (default)", "      min_cluster_size:       200", 1)
    assert patched != text
    log = tmp_path / "hydra_python.INFO"
    log.write_text(patched)
    assert cf.check_effective(log, expected)["frontend.objects.min_cluster_size"] == ("200", True)
    # the other min_cluster_size entries (surface places 50, frontiers 10) are untouched
    cfg = cf.read_effective_config(log)
    assert cfg[("frontend", "surface_places", "clustering", "min_cluster_size")][0] == "50"
    assert cfg[("frontend", "frontier_places", "min_cluster_size")][0] == "10"


def test_check_effective_unknown_key_reports_not_ok():
    res = cf.check_effective(HEADER, {"frontend.objects.nope": 1})
    assert res["frontend.objects.nope"][1] is False


def make_run(tmp_path, text):
    logs = tmp_path / "run" / "logs"
    logs.mkdir(parents=True)
    (logs / "hydra_python.host.user.log.INFO.20261101-000000.1").write_text(text)  # the hydra_python.INFO symlink is absent on purpose
    return tmp_path / "run"


@needs_base
def test_verify_run_reads_the_real_log_and_flags_a_missed_override(tmp_path):
    cfg = cf.make_config_dir("rooms_min_size", 0.5, tmp_path / "cfg")
    text = HEADER.read_text()
    key = "backend.update_functors.rooms.room_finder.min_room_size"
    applied = text.replace("min_room_size:          10 (default)", "min_room_size:          63", 1)
    assert applied != text
    run_ok = make_run(tmp_path / "a", applied)
    assert cf.verify_run(run_ok, cfg) == {key: ("63", True)}
    run_bad = make_run(tmp_path / "b", text)  # Hydra echoed the default: the override never arrived
    assert cf.verify_run(run_bad, cfg) == {key: ("10", False)}
    with pytest.raises(FileNotFoundError):
        cf.find_info_log(tmp_path / "nothing")


@needs_base
def test_cli_verify_exit_code(tmp_path):
    cfg = cf.make_config_dir("rooms_min_size", 0.5, tmp_path / "cfg")
    run = make_run(tmp_path / "r", HEADER.read_text())
    with pytest.raises(SystemExit) as e:
        cf.main(["--verify", str(run), str(cfg)])
    assert e.value.code == 1
