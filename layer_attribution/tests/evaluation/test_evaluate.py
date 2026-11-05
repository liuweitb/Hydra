import json

import numpy as np
import pandas as pd
import pytest

from layer_attribution import diagnosis as D
from layer_attribution import runs as R
from layer_attribution.evaluation import evaluate as E

FEATS = {  # name -> (kind, evidence layer, ideal, tol_abs)
    "mesh_f": ("reference_based", "mesh", 0.0, 0.05),
    "obj_f": ("reference_based", "objects", 0.0, 0.05),
    "obj_g": ("reference_based", "objects", 0.0, 0.05),
    "pl_f": ("reference_based", "places", 0.0, 0.05),
    "room_f": ("reference_based", "rooms", 0.0, 0.05),
    "mesh_rf": ("reference_free", "mesh", np.nan, 0.05),
}
RB1 = {"rb": D.ScoreConfig("rb", k=1)}


def meta():
    rows = [dict(name=n, kind=k, layer=l, ideal=i, tol_abs=t, tol_rel=0.0, count_based=False) for n, (k, l, i, t) in FEATS.items()]
    return pd.DataFrame(rows).set_index("name")


def build(rng):
    """reference, repeat, benign (families label_dropout / depth_noise, seeds 0,1 = validation and 3,4,5 = calibration),
    sensor faults (seed 0 dev, seed 1 test) and config faults (objects_min_size: 0.6 dev, 1.0 test)."""
    rows, qrows = [], []

    def add(run, split, kind, fault, sev, seed, vals, fail):
        full = {**{n: 0.0 for n in FEATS}, **vals}
        rows.append(dict(run=run, split=split, kind=kind, fault=fault, severity=sev, seed=seed, **full))
        qrows.append(dict(run=run, failure_rate=fail, n_room_missing=0, n_wrong_room=0,
                          n_wrong_label=int(fail * 100) if fault.startswith(("label", "objects")) else 0, n_wrong_position=0,
                          n_target_missing=0 if fault.startswith(("label", "objects")) else int(fail * 100)))

    add(R.REFERENCE, "reference", "reference", "none", 0.0, None, {}, 0.0)
    add(R.REPEAT, "repeat", "repeat", "none", 0.0, None, {}, 0.0)
    for seed in (0, 1, 3, 4, 5):
        for fault in ("label_dropout", "depth_noise"):
            noise = {n: float(rng.normal(0, 0.02)) for n in FEATS}
            add(f"b_{fault}_{seed}", "benign", "sensor", fault, 0.03, seed, noise, 0.02 * (seed % 3) + (0.2 if fault == "label_dropout" and seed == 4 else 0.0))
    for seed, split in ((0, "dev"), (1, "test")):
        for sev in (0.2, 0.5):
            add(f"ld_{sev}_{seed}", split, "sensor", "label_dropout", sev, seed, {"obj_f": 4.0 * sev, "obj_g": 3.0 * sev}, 0.5 * sev + 0.1)
            # a geometry fault moves mesh and places, and its downstream layers more strongly
            add(f"dn_{sev}_{seed}", split, "sensor", "depth_noise", sev, seed,
                {"mesh_f": 1.5 * sev, "pl_f": 3.0 * sev, "obj_f": 5.0 * sev, "obj_g": 5.0 * sev, "room_f": 4.0 * sev, "mesh_rf": 1.0 * sev}, 0.5 * sev + 0.1)
    add("cfg_0.6", "dev", "config", "objects_min_size", 0.6, None, {"obj_f": 3.0, "obj_g": 3.0}, 0.7)
    add("cfg_1.0", "test", "config", "objects_min_size", 1.0, None, {"obj_f": 5.0, "obj_g": 5.0}, 0.9)
    runmeta = pd.DataFrame(rows).set_index("run")[["split", "kind", "fault", "severity", "seed"]]
    table = pd.DataFrame(rows).set_index("run")[list(FEATS)]
    return table, runmeta, pd.DataFrame(qrows).set_index("run")


@pytest.fixture
def data():
    return build(np.random.default_rng(0))


def write_inputs(data, tmp_path):
    table, runmeta, queries = data
    pd.concat([runmeta, table], axis=1).rename_axis("run").reset_index().to_csv(tmp_path / "f.csv", index=False)
    meta().reset_index().to_csv(tmp_path / "m.csv", index=False)
    queries.reset_index().to_csv(tmp_path / "q.csv", index=False)
    return ["--features", str(tmp_path / "f.csv"), "--meta", str(tmp_path / "m.csv"), "--queries", str(tmp_path / "q.csv"),
            "--out", str(tmp_path / "out"), "--benign-exclude", ""]


def test_calibration_and_validation_runs_use_disjoint_benign_seeds(data):
    _, runmeta, _ = data
    calib, valid = E.calibration_runs(runmeta, exclude=()), E.validation_runs(runmeta, exclude=())
    assert {runmeta.at[r, "seed"] for r in calib} == {3, 4, 5} and {runmeta.at[r, "seed"] for r in valid} == {0, 1}
    assert not set(calib) & set(valid)
    assert len(E.calibration_runs(runmeta, exclude=("depth_noise",))) == 3  # one family left
    assert len(E.calibration_runs(runmeta, exclude=(), seeds=None)) == 10


def test_records_cover_dev_runs_only_and_carry_ground_truth(data):
    table, runmeta, queries = data
    recs, cals, _ = E.build_records(table, runmeta, meta(), queries, RB1, ("dev",), exclude=())
    assert {r["split"] for r in recs} == {"dev"}
    assert {r["run"] for r in recs} == {"ld_0.2_0", "ld_0.5_0", "dn_0.2_0", "dn_0.5_0", "cfg_0.6"}
    r = next(r for r in recs if r["run"] == "dn_0.5_0" and r["method"] == "dag_rb")
    assert r["accept"] == ["mesh", "places"] and r["fault"] == "depth_noise" and r["seed"] == 0 and r["config"] == "rb-k1-m1.5"
    assert {r["method"] for r in recs} == {"dag_rb", "naive_rb", "downstream_rb", "symptom"}
    assert set(cals) == {1.5}


def test_dag_beats_naive_when_the_downstream_layer_is_stronger(data):
    table, runmeta, queries = data
    recs, _, _ = E.build_records(table, runmeta, meta(), queries, RB1, ("dev",), exclude=())
    pick = lambda m: next(r for r in recs if r["run"] == "dn_0.5_0" and r["method"] == m)  # noqa: E731
    assert pick("dag_rb")["blame"][0] in ("mesh", "places")  # roots of the deviating set
    assert pick("naive_rb")["blame"] == ["objects"]  # the strongest layer is downstream: the naive rule is wrong
    assert pick("downstream_rb")["blame"][0] in ("objects", "rooms")


def test_label_and_config_faults_are_attributed_to_objects(data):
    table, runmeta, queries = data
    recs, _, _ = E.build_records(table, runmeta, meta(), queries, RB1, ("dev",), exclude=())
    for run in ("ld_0.5_0", "cfg_0.6"):
        assert next(r for r in recs if r["run"] == run and r["method"] == "dag_rb")["blame"] == ["objects"]
        assert next(r for r in recs if r["run"] == run and r["method"] == "symptom")["blame"] == ["objects"]


def test_build_records_refuses_the_test_split_without_final(data):
    table, runmeta, queries = data
    with pytest.raises(R.SplitViolation):
        E.build_records(table, runmeta, meta(), queries, RB1, ("test",), exclude=())
    recs, _, _ = E.build_records(table, runmeta, meta(), queries, RB1, ("test",), exclude=(), final=True)
    assert {r["run"] for r in recs} == {"ld_0.2_1", "ld_0.5_1", "dn_0.2_1", "dn_0.5_1", "cfg_1.0"}


def test_masking_uses_the_band_of_the_same_fault_family(data):
    table, runmeta, queries = data
    pooled, fam = E.mask_bands(runmeta, queries)
    assert pooled.hi_rate == pytest.approx(0.22) and fam["depth_noise"].hi_rate == pytest.approx(0.04)
    assert fam["label_dropout"].hi_rate == pytest.approx(0.22)
    recs, _, _ = E.build_records(table, runmeta, meta(), queries, RB1, ("dev",), exclude=())
    dn = next(r for r in recs if r["run"] == "dn_0.2_0" and r["method"] == "dag_rb")  # failure rate 0.2
    assert dn["masked"] is False and dn["masked_pooled"] is True  # above its own family, inside the pooled band
    cfg = next(r for r in recs if r["run"] == "cfg_0.6" and r["method"] == "dag_rb")  # no benign counterpart: pooled band
    assert cfg["masked"] is False and cfg["masked_pooled"] is False


def test_validation_false_alarm_counts_deviating_held_out_benign_runs(data):
    table, runmeta, _ = data
    calib, valid = E.calibration_runs(runmeta, exclude=()), E.validation_runs(runmeta, exclude=())
    cfg = D.ScoreConfig("rb", k=1, margin=2.0)
    ok = E.validation_false_alarm(table, meta(), calib, valid, cfg)
    assert ok["n"] == 4 and ok["rate"] == 0.0
    bad = table.copy()
    bad.loc[valid[0], "mesh_f"] = 9.0
    fa = E.validation_false_alarm(bad, meta(), calib, valid, cfg)
    assert fa["rate"] == pytest.approx(0.25) and fa["per_layer"]["mesh"] == pytest.approx(0.25)
    empty = E.validation_false_alarm(table, meta(), calib, [], cfg)
    assert empty["n"] == 0 and np.isnan(empty["rate"])


def test_loo_false_alarm_runs_on_the_calibration_set(data):
    table, runmeta, _ = data
    calib = E.calibration_runs(runmeta, exclude=())
    cfg = D.ScoreConfig("rb", k=1, margin=3.0)
    assert E.loo_false_alarm(table, meta(), calib, cfg, "run", runmeta)["n"] == 6
    assert E.loo_false_alarm(table, meta(), calib, cfg, "family", runmeta)["n"] == 6


def grid_row(mode, k, ex, margin, acc, fa):
    return dict(mode=mode, k=k, exclude_count_based=ex, margin=margin, dag_macro=acc, valid_false_alarm=fa)


def test_choose_configs_prefers_accuracy_among_configs_meeting_the_cap_then_conservative_ties():
    g = pd.DataFrame([grid_row("rb", 1, False, 1.5, 1.0, 0.2), grid_row("rb", 2, False, 2.0, 0.9, 0.0),
                      grid_row("rb", 3, True, 3.0, 0.9, 0.0), grid_row("rb", 1, True, 3.0, 0.9, 0.04),
                      grid_row("rf", 1, False, 1.5, 0.8, 0.0)])
    c = E.choose_configs(g)
    assert c["rb"]["cap_met"] and c["rb"]["config"]["k"] == 3 and c["rb"]["config"]["margin"] == 3.0  # tie: larger margin, then k
    assert c["rf"]["config"]["k"] == 1


def test_choose_configs_fallback_keeps_every_least_alarming_config_in_the_tie_break():
    g = pd.DataFrame([grid_row("rb", 1, False, 1.5, 0.8, 0.2), grid_row("rb", 2, False, 2.0, 1.0, 0.2),
                      grid_row("rb", 3, False, 3.0, 0.9, 0.5), grid_row("rf", 1, False, 1.5, 0.7, np.nan),
                      grid_row("rf", 2, False, 2.0, 0.6, np.nan)])
    c = E.choose_configs(g)
    assert not c["rb"]["cap_met"] and c["rb"]["config"]["k"] == 2  # accuracy decides among the 0.2 ties, not grid order
    assert not c["rf"]["cap_met"] and c["rf"]["config"]["k"] == 1  # NaN false alarm counts as 1 and ties again by accuracy


def test_test_split_is_refused_by_the_cli_without_final(data, tmp_path):
    argv = ["report"] + write_inputs(data, tmp_path) + ["--splits", "test"]
    with pytest.raises(R.SplitViolation):
        E.main(argv)


def test_select_then_report_end_to_end_with_separate_sensor_and_config_blocks(data, tmp_path):
    common = write_inputs(data, tmp_path)
    assert E.main(["select"] + common) == 0
    chosen = json.loads((tmp_path / "out" / "chosen_config.json").read_text())
    assert set(chosen) == {"rb", "rf"} and {"config", "cap_met", "valid_false_alarm", "dev_macro"} <= set(chosen["rb"])
    assert E.main(["report"] + common + ["--splits", "test", "--final"]) == 0
    summ = json.loads((tmp_path / "out" / "summary_test.json").read_text())
    assert set(summ["blocks"]) == {"sensor", "config"}
    assert summ["blocks"]["sensor"]["n_runs"] == 4 and summ["blocks"]["config"]["n_runs"] == 1
    assert "dag_rb" in summ["blocks"]["sensor"]["accuracy_by_fault"] and "const_mesh" in summ["blocks"]["sensor"]["constants"]
    assert "dag_rb" in summ["blocks"]["sensor"]["permutation_null"] and "false_alarm" in summ
    text = (tmp_path / "out" / "report_test.md").read_text()
    assert text.startswith("# Attribution report") and "## sensor faults" in text and "## config faults" in text
    assert "held-out benign runs" in text and "cap:" in text
    assert "[" not in text.split("### Top-1 accuracy")[1].split("Constant baselines")[0].replace("[2.5", "")  # n < 5: no interval
    assert (tmp_path / "out" / "predictions_test.csv").exists()
    prov = json.loads((tmp_path / "out" / "provenance_test.json").read_text())
    assert prov["final"] is True and prov["splits"] == ["test"] and len(prov["inputs"]["features"]["sha256"]) == 64
    assert prov["calibration_seeds"] == [3, 4, 5] and prov["validation_seeds"] == [0, 1]
    assert "evaluation/evaluate.py" in {k.replace("\\", "/") for k in prov["source_sha256"]}
    assert prov["scored_runs"] == ["cfg_1.0", "dn_0.2_1", "dn_0.5_1", "ld_0.2_1", "ld_0.5_1"]
