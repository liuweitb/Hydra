import numpy as np
import pandas as pd
import pytest

from layer_attribution import calibration as C
from layer_attribution import diagnosis as D


def meta_frame():
    rows = [
        ("a", "reference_based", "mesh", 0.0, 0.1, 0.0, False),
        ("b", "reference_based", "objects", 1.0, 0.0, 0.1, True),
        ("c", "reference_free", "places", np.nan, 0.05, 0.0, False),
        ("d", "reference_based", None, 0.0, 0.1, 0.0, False),  # not layer evidence
        ("e", "reference_based", "rooms", 0.0, 0.0, 0.0, False),
    ]
    return pd.DataFrame(rows, columns=["name", "kind", "layer", "ideal", "tol_abs", "tol_rel", "count_based"]).set_index("name")


def table():
    cols = ["a", "b", "c", "d", "e"]
    data = {
        "ref": [0.0, 1.0, 10.0, 0.0, 0.0],
        "ben1": [0.2, 1.1, 10.2, 0.0, 0.0],
        "ben2": [0.4, 0.9, 9.8, 0.0, 0.0],
        "ben3": [0.1, 1.0, 10.0, 0.0, 0.0],
        "fault": [3.0, 1.05, 12.0, 5.0, 2.0],
    }
    return pd.DataFrame.from_dict(data, orient="index", columns=cols)


def test_fit_threshold_is_margin_times_benign_max_or_floor():
    cal = C.fit(table(), meta_frame(), ["ben1", "ben2", "ben3"], "ref", margin=1.5)
    assert cal.threshold["a"] == pytest.approx(1.5 * 0.4)  # benign max deviation 0.4 beats the floor 0.1
    assert cal.threshold["b"] == pytest.approx(max(1.5 * 0.1, 0.1))  # floor = 0.1 * |1| ties the margin term
    assert cal.threshold["c"] == pytest.approx(1.5 * 0.2)
    assert cal.threshold["e"] > 0 and cal.threshold["e"] < 1e-6  # zero benign deviation and zero floor: tiny positive
    assert cal.threshold["d"] == pytest.approx(0.1)  # floor when the benign runs do not move


def test_reference_free_center_is_the_median_of_reference_and_benign():
    cal = C.fit(table(), meta_frame(), ["ben1", "ben2", "ben3"], "ref")
    assert cal.center["c"] == pytest.approx(np.median([10.0, 10.2, 9.8, 10.0]))
    assert cal.center["a"] == 0.0 and cal.center["b"] == 1.0  # ideals


def test_strengths_and_benign_runs_stay_below_one():
    cal = C.fit(table(), meta_frame(), ["ben1", "ben2", "ben3"], "ref", margin=1.5)
    s = cal.strengths(table())
    assert (s.loc[["ben1", "ben2", "ben3"]].drop(columns=["e"]) < 1.0).all().all()
    assert s.at["fault", "a"] == pytest.approx(3.0 / 0.6)
    assert s.at["fault", "e"] == C.STRENGTH_CAP  # any move of a zero-variance feature is capped, not infinite


def test_nan_benign_value_makes_the_feature_unusable_only_if_all_are_nan():
    t = table()
    t.loc[["ben1", "ben2", "ben3"], "c"] = np.nan
    cal = C.fit(t, meta_frame(), ["ben1", "ben2", "ben3"], "ref")
    assert np.isnan(cal.threshold["c"])
    t2 = table()
    t2.loc["ben2", "a"] = np.nan
    assert C.fit(t2, meta_frame(), ["ben1", "ben2", "ben3"], "ref").threshold["a"] == pytest.approx(1.5 * 0.2)


def test_usable_features_by_mode_and_count_filter():
    cal = C.fit(table(), meta_frame(), ["ben1", "ben2", "ben3"], "ref")
    assert C.usable_features(cal, "rb") == {"mesh": ["a"], "objects": ["b"], "rooms": ["e"]}
    assert C.usable_features(cal, "rf") == {"places": ["c"]}
    assert C.usable_features(cal, "both", exclude_count_based=True) == {"mesh": ["a"], "places": ["c"], "rooms": ["e"]}


def test_bad_inputs():
    with pytest.raises(ValueError):
        C.fit(table(), meta_frame(), [], "ref")
    with pytest.raises(ValueError):
        C.check_meta(meta_frame().drop(columns=["ideal"]))
    bad = meta_frame()
    bad.loc["a", "kind"] = "nope"
    with pytest.raises(ValueError):
        C.check_meta(bad)


def row(**kw):
    return pd.Series(kw, dtype=float)


def test_layer_scores_use_the_kth_largest_strength():
    r = row(m1=5.0, m2=3.0, m3=0.5, o1=0.4, o2=2.0)
    feats = {"mesh": ["m1", "m2", "m3"], "objects": ["o1", "o2"], "places": [], "rooms": ["missing"]}
    s1, s2, s3 = (D.layer_scores(r, feats, k) for k in (1, 2, 3))
    assert (s1["mesh"], s2["mesh"], s3["mesh"]) == (5.0, 3.0, 0.5)
    assert s3["objects"] == 0.4  # fewer than k features uses the smallest
    assert s1["places"] == 0.0 and s1["rooms"] == 0.0  # no usable feature, and a name absent from the row is ignored


def test_an_undefined_usable_feature_counts_as_a_deviation_not_as_silence():
    r = row(o1=np.nan, o2=0.2, m1=np.nan)
    feats = {"objects": ["o1", "o2"], "mesh": ["m1"]}
    s = D.layer_scores(r, feats, k=1)
    assert s["objects"] == D.NAN_STRENGTH and s["mesh"] == D.NAN_STRENGTH  # structure gone: counted as deviating
    assert D.layer_scores(r, feats, k=2)["objects"] == 0.2  # but not stronger than a measured feature
    assert D.nan_counts(r, feats) == {"mesh": 1, "objects": 1, "places": 0, "rooms": 0}
    assert D.dag_rule(D.layer_scores(r, feats, k=1)).deviating == ["mesh", "objects"]


SC = lambda **kw: {"mesh": 0.0, "objects": 0.0, "places": 0.0, "rooms": 0.0, **kw}  # noqa: E731


def test_dag_rule_blames_the_unexplained_layers():
    assert D.dag_rule(SC(objects=7.0)).blame == ["objects"]
    d = D.dag_rule(SC(mesh=1.2, objects=9.0, places=6.0, rooms=4.0))
    assert d.blame == ["places", "mesh"]  # roots of the dataflow DAG, strongest first
    assert d.deviating == ["mesh", "objects", "places", "rooms"]
    assert D.dag_rule(SC(mesh=1.2, objects=9.0, places=6.0, rooms=4.0), "chain").blame == ["mesh"]
    assert D.dag_rule(SC(mesh=0.9, rooms=0.3)).blame == []


def test_naive_blames_the_largest_score_and_abstains_when_nothing_deviates():
    assert D.naive_rule(SC(mesh=1.2, objects=9.0, places=6.0)).blame == ["objects"]
    assert D.naive_rule(SC(mesh=0.9)).blame == []
    assert D.naive_rule(SC(mesh=2.0, places=2.0)).blame == ["mesh"]  # deterministic tie-break by layer order


def test_downstream_rule_prefers_the_deepest_layer():
    assert D.downstream_rule(SC(mesh=9.0, objects=2.0)).blame == ["objects"]
    assert D.downstream_rule(SC(mesh=9.0, places=3.0)).blame == ["mesh"]
    assert D.downstream_rule(SC()).blame == []


def test_symptom_rule_uses_failure_modes_only():
    d = D.symptom_rule({"room_missing": 3, "wrong_room": 1, "wrong_label": 2, "target_missing": 0})
    assert d.blame == ["rooms"] and d.scores["rooms"] == pytest.approx(4 / 6)
    assert D.symptom_rule({"target_missing": 5}).blame == ["objects"]
    assert D.symptom_rule({}).blame == [] and D.symptom_rule({"wrong_room": 0}).blame == []


def test_diagnose_run_end_to_end():
    cal = C.fit(table(), meta_frame(), ["ben1", "ben2", "ben3"], "ref", margin=1.5)
    s = cal.strengths(table()).loc["fault"]
    out = D.diagnose_run(s, cal, D.ScoreConfig("rb", k=1))
    assert set(out) == {"dag_rb", "naive_rb", "downstream_rb"}
    assert out["dag_rb"].blame[0] in ("mesh", "rooms")  # both a and e move, objects b does not
    assert all(d.nan_counts == {l: 0 for l in D.LAYERS} for d in out.values())
    assert D.ScoreConfig("rf").label() == "rf-k2-m1.5"
    assert D.ScoreConfig("rb", 3, True, 2.0).label() == "rb-k3-nocount-m2"
    with pytest.raises(ValueError):
        D.ScoreConfig("nope")
    with pytest.raises(ValueError):
        D.ScoreConfig("rb", margin=0.0)
