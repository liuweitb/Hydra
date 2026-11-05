import math

import numpy as np
import pytest

from layer_attribution.evaluation import metrics as m


def rec(run, fault, sev, accept, blame, method="dag_rb", masked=False, deviating=None, failure_rate=0.4, seed=1):
    return dict(run=run, fault=fault, severity=sev, seed=seed, split="test", kind="sensor", accept=list(accept),
                masked=masked, failure_rate=failure_rate, method=method, blame=list(blame),
                scores={l: 1.0 for l in blame}, deviating=list(deviating if deviating is not None else blame))


GEOM = ("mesh", "places")


def test_correctness_lenient_and_strict():
    r = rec("a", "depth_noise", 0.5, GEOM, ["places"])
    assert m.is_correct(r) and not m.is_correct(r, strict=True)
    r2 = rec("b", "depth_noise", 0.5, GEOM, ["places", "mesh"])
    assert m.is_correct(r2) and m.is_correct(r2, strict=True)
    assert not m.is_correct(rec("c", "depth_noise", 0.5, GEOM, ["rooms"]))
    assert not m.is_correct(rec("d", "depth_noise", 0.5, GEOM, []))


def test_masked_runs_are_not_scored():
    rows = [rec("a", "label_dropout", 0.2, ["objects"], ["objects"]),
            rec("b", "label_dropout", 0.2, ["objects"], ["rooms"], masked=True)]
    acc = m.accuracy(rows, "dag_rb")
    assert acc["label_dropout"]["n"] == 1 and acc["label_dropout"]["acc"] == 1.0


def test_macro_average_weights_groups_equally():
    rows = [rec(f"g{i}", "depth_noise", 0.5, GEOM, ["mesh"]) for i in range(9)]
    rows += [rec("o", "label_dropout", 0.5, ["objects"], ["rooms"])]  # 1 wrong run in the small group
    acc = m.accuracy(rows, "dag_rb")
    assert acc["depth_noise"]["acc"] == 1.0 and acc["label_dropout"]["acc"] == 0.0
    assert acc["macro"]["acc"] == pytest.approx(0.5) and acc["macro"]["n"] == 10


def test_accuracy_by_severity_and_unknown_method():
    rows = [rec("a", "pose_drift", 0.2, GEOM, ["mesh"]), rec("b", "pose_drift", 0.8, GEOM, ["rooms"])]
    acc = m.accuracy(rows, "dag_rb", by="severity")
    assert acc[0.2]["acc"] == 1.0 and acc[0.8]["acc"] == 0.0
    assert m.accuracy(rows, "nope") == {}


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    x = [1, 0, 1, 1, 0, 1, 1, 1]
    a, b = m.bootstrap_ci(x), m.bootstrap_ci(x)
    assert a == b and a[1] <= a[0] <= a[2] and a[0] == pytest.approx(0.75)
    assert all(math.isnan(v) for v in m.bootstrap_ci([]))
    assert m.bootstrap_ci([1, 1, 1]) == (1.0, 1.0, 1.0)


def test_confusion_rows_are_accept_sets():
    rows = [rec("a", "depth_noise", 0.5, GEOM, ["mesh"]), rec("b", "depth_holes", 0.5, GEOM, ["rooms"]),
            rec("c", "label_dropout", 0.5, ["objects"], []), rec("d", "label_dropout", 0.5, ["objects"], ["objects"])]
    t = m.confusion(rows, "dag_rb")
    assert t["mesh+places"]["mesh"] == 1 and t["mesh+places"]["rooms"] == 1
    assert t["objects"]["none"] == 1 and t["objects"]["objects"] == 1


def test_masked_and_propagation_tables():
    rows = [rec("a", "label_dropout", 0.2, ["objects"], ["objects"], masked=True, failure_rate=0.01),
            rec("b", "label_dropout", 0.2, ["objects"], ["objects"], failure_rate=0.3, deviating=["objects", "rooms"]),
            rec("c", "label_dropout", 0.5, ["objects"], ["objects"], failure_rate=0.6)]
    t = m.masked_table(rows)
    assert t[("label_dropout", 0.2)]["n"] == 2 and t[("label_dropout", 0.2)]["masked"] == 0.5
    assert t[("label_dropout", 0.2)]["failure_rate"] == pytest.approx(0.155)
    p = m.propagation_table(rows, "dag_rb")["label_dropout"]
    assert p["objects"] == 1.0 and p["rooms"] == pytest.approx(1 / 3) and p["mesh"] == 0.0


def test_permutation_null_is_below_a_perfect_method_and_reproducible():
    rows = [rec(f"d{i}", "depth_noise", 0.5, GEOM, ["mesh"]) for i in range(6)]
    rows += [rec(f"l{i}", "label_dropout", 0.5, ["objects"], ["objects"]) for i in range(6)]
    a, b = m.permutation_null(rows, "dag_rb", n_perm=500), m.permutation_null(rows, "dag_rb", n_perm=500)
    assert a == b and a["real"] == 1.0 and a["n"] == 12
    assert a["null_mean"] < 0.7 and a["null_lo"] <= a["null_mean"] <= a["null_hi"] and a["p_ge"] < 0.05
    useless = [rec(f"x{i}", "depth_noise" if i % 2 else "label_dropout", 0.5, GEOM if i % 2 else ["objects"], ["rooms"]) for i in range(8)]
    z = m.permutation_null(useless, "dag_rb", n_perm=200)
    assert z["real"] == 0.0 and z["null_mean"] == 0.0 and z["p_ge"] == 1.0
    assert np.isnan(m.permutation_null(rows[:1], "dag_rb")["real"])


def test_permutation_null_exposes_the_chance_level_of_a_method_that_never_blames_some_layers():
    # a method that always says 'mesh' scores 0.5 on a 50/50 mix of geometry (accept mesh) and label (accept objects) faults
    rows = [rec(f"d{i}", "depth_noise", 0.5, GEOM, ["mesh"]) for i in range(5)]
    rows += [rec(f"l{i}", "label_dropout", 0.5, ["objects"], ["mesh"]) for i in range(5)]
    n = m.permutation_null(rows, "dag_rb", n_perm=400)
    assert n["real"] == pytest.approx(0.5) and n["null_mean"] == pytest.approx(0.5, abs=0.1)  # like-for-like chance, not 0.25


def test_named_member_counts_the_blamed_layer_when_the_accept_set_has_two_members():
    rows = [rec("a", "depth_noise", 0.5, GEOM, ["mesh"]), rec("b", "depth_noise", 0.8, GEOM, ["mesh"]),
            rec("c", "depth_noise", 0.2, GEOM, []), rec("d", "label_dropout", 0.5, ["objects"], ["objects"])]
    assert m.named_member(rows, "dag_rb") == {"depth_noise": {"mesh": 2, "none": 1}}


def test_random_and_prior_baselines():
    rows = [rec("a", "depth_noise", 0.5, GEOM, ["mesh"]), rec("b", "label_dropout", 0.5, ["objects"], ["objects"])]
    assert m.expected_random(rows) == pytest.approx((2 / 4 + 1 / 4) / 2)
    dev = [rec(f"d{i}", "depth_noise", 0.5, GEOM, ["mesh"]) for i in range(4)] + \
          [rec(f"e{i}", "pose_drift", 0.5, GEOM, ["mesh"]) for i in range(4)] + \
          [rec("f", "label_dropout", 0.5, ["objects"], ["objects"])]
    layer = m.fit_prior(dev)
    assert layer in GEOM  # two of three fault types accept it
    base = m.prior_records(rows, layer)
    assert [r["method"] for r in base] == ["prior", "prior"] and all(r["blame"] == [layer] for r in base)
    acc = m.accuracy(base, "prior")
    assert acc["depth_noise"]["acc"] == 1.0 and acc["label_dropout"]["acc"] == 0.0
