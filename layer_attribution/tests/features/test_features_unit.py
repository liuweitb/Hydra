"""Feature battery on synthetic scenes: registry, known answers, compute() contract, identity, degenerate inputs."""

import math

import numpy as np
import pytest
import scene_factory as sf

from layer_attribution import dag
from layer_attribution import features as F
from layer_attribution import snapshot as sn

TAGS = {"mesh_geom", "mesh_sem", "objects", "places", "rooms", "edges", "input_pose", "process"}


def view(a, **kw):
    return F.FeatureView(sf.snapshot_from_arrays(a, **kw))


@pytest.fixture(scope="module")
def scene():
    return sf.make_scene(0)


@pytest.fixture(scope="module")
def perturbed():
    return sf.make_scene(0, jitter=0.02, relabel=0.2, drop_objects=3, drop_frames=True)


# ---------------------------------------------------------------- registry and metadata
def test_registry_is_well_formed():
    assert len(F.REGISTRY) == 99
    for name, s in F.REGISTRY.items():
        assert s.name == name and s.kind in ("reference_based", "reference_free") and s.layer in TAGS
        assert s.definition, name
        assert (s.ideal is not None) == (s.kind == "reference_based"), name
        assert s.tol_abs >= 0 and s.tol_rel >= 0, name


def test_meta_lists_every_feature_with_its_evidence_layer():
    m = F.meta()
    assert [x["name"] for x in m] == list(F.REGISTRY)
    assert set(m[0]) == {"name", "kind", "layer", "evidence_layer", "ideal", "tol_abs", "tol_rel", "count_based", "unit", "definition"}
    for x in m:
        assert x["evidence_layer"] == dag.FEATURE_LAYER.get(x["layer"]), x["name"]
    assert {x["evidence_layer"] for x in m} == {"mesh", "objects", "places", "rooms", None}
    assert {x["layer"] for x in m if x["evidence_layer"] is None} == {"edges", "input_pose", "process"}


def test_source_hashes_follow_called_features():
    h = F.source_hashes()
    assert set(h) == set(F.REGISTRY) and all(len(v) == 12 for v in h.values())
    assert F._called_features(F.floor_time_structure.__code__) == {"floor_z_time_range", "floor_z_robust_sd"}
    assert F._called_features(F.mesh_vertex_count_ratio.__code__) == set()


def test_features_never_touch_the_injection_manifest():
    import inspect

    assert "injection_manifest" not in inspect.getsource(F)
    assert not hasattr(view(sf.make_scene(0, g=8, n_place=20)), "manifest")


# ---------------------------------------------------------------- helpers with known answers
def test_distribution_and_box_helpers():
    assert F.js_divergence([1, 2, 3], [2, 4, 6]) < 1e-12
    assert abs(F.js_divergence([1, 0], [0, 1]) - 1) < 1e-12
    c, d = np.zeros((1, 3)), np.ones((1, 3))
    assert abs(F.aabb_iou(c, d, c, d)[0] - 1) < 1e-12
    assert abs(F.aabb_iou(c, d, c + [0.5, 0, 0], d)[0] - 1 / 3) < 1e-12


def test_label_histogram_ignores_unset_and_out_of_range_labels():
    h = F._label_hist(np.array([-1, 0, 0, 3, 20, 21, 65535]))
    assert len(h) == F.N_LABELS and h.sum() == 4 and h[0] == 2 and h[3] == 1 and h[20] == 1


def _places(pos, parent):
    pos = np.asarray(pos, float)
    return view(dict(place_pos=pos, place_parent=np.asarray(parent, np.int32)))


def test_room_partition_agreement():
    pos = [[0, 0, 0], [0.1, 0, 0], [5, 0, 0], [5.1, 0, 0]]
    same = F.room_partition_ari(_places(pos, [0, 0, 1, 1]), _places(pos, [0, 0, 1, 1]))
    crossed = F.room_partition_ari(_places(pos, [0, 1, 0, 1]), _places(pos, [0, 0, 1, 1]))
    assert abs(same - 1) < 1e-12 and abs(crossed + 0.5) < 1e-12


def test_free_space_overlap_and_volume():
    pp = np.array([[0, 0, 0], [10, 0, 0]], float)
    a = view(dict(place_pos=pp, place_radius=np.array([1.0, 1.0])))
    b = view(dict(place_pos=pp + [100, 0, 0], place_radius=np.array([1.0, 1.0])))
    big = view(dict(place_pos=pp, place_radius=np.array([2.0, 2.0])))
    assert abs(F.place_freespace_iou(a, a) - 1) < 1e-12 and F.place_freespace_iou(a, b) == 0
    assert abs(F.place_freevol_ratio(a, a) - 1) < 1e-12
    assert 5 < F.place_freevol_ratio(big, a) < 11  # radius x2 gives about x8


def test_object_duplicate_and_overlap_fractions():
    pos = np.array([[0, 0, 0], [0.3, 0, 0], [5, 0, 0]], float)
    base = dict(obj_pos=pos, obj_bbox_center=pos, obj_bbox_dim=np.ones((3, 3)))
    assert abs(F.obj_dup_same_class_frac_050(view(dict(base, obj_label=np.array([5, 5, 7])))) - 2 / 3) < 1e-12
    assert abs(F.obj_overlap_diffclass_frac(view(dict(base, obj_label=np.array([5, 7, 7])))) - 2 / 3) < 1e-12


def _plane(n, noise=0.0, z0=1.0, seed=0, label=F.LABEL_FLOOR, size=10.0):
    r = np.random.default_rng(seed)
    z = z0 + r.normal(0, noise, n) if noise else np.full(n, z0)
    return np.c_[r.uniform(0, size, n), r.uniform(0, size, n), z].astype(np.float32), np.full(n, label, np.int16)


@pytest.mark.parametrize("noise,lo,hi", [(0.0, 0, 1e-6), (0.02, 1e-3, 1)])
def test_roughness_separates_flat_from_noisy_surfaces(noise, lo, hi):
    xyz, lab = _plane(20000, noise)
    assert lo <= F.mesh_roughness_median(view(dict(mesh_xyz=xyz, mesh_label=lab))) <= hi


@pytest.mark.parametrize("drift,lo,hi", [(0.0, 0, 0.005), (0.3, 0.2, 0.35)])
def test_floor_height_range_over_time_sees_z_drift(drift, lo, hi):
    n = 40000
    t = np.random.default_rng(3).uniform(0, 500, n)
    xyz, lab = _plane(n)
    xyz[:, 2] += (drift * t / 500).astype(np.float32)
    assert lo <= F.floor_z_time_range(view(dict(mesh_xyz=xyz, mesh_label=lab, mesh_t_first=(t * 1e9).astype(np.int64)))) <= hi


def test_floor_time_structure_is_low_when_noise_limited_and_high_when_ordered_in_time():
    n = 40000
    t = np.random.default_rng(6).uniform(0, 500, n)
    xyz, lab = _plane(n, noise=0.002)
    tf = (t * 1e9).astype(np.int64)
    low = F.floor_time_structure(view(dict(mesh_xyz=xyz, mesh_label=lab, mesh_t_first=tf)))
    xyz2 = xyz.copy()
    xyz2[:, 2] += (0.3 * t / 500).astype(np.float32)
    high = F.floor_time_structure(view(dict(mesh_xyz=xyz2, mesh_label=lab, mesh_t_first=tf)))
    assert low < 0.5 and high > 1.5


def test_floor_time_range_breaks_ties_in_first_seen_time_by_vertex_order():
    n = 30000
    stamps = np.repeat(np.arange(100), n // 100).astype(np.int64) * 10**9  # 300 vertices share each stamp, bins cut through blocks
    xyz, lab = _plane(n, noise=0.002)
    xyz[:, 2] += np.random.default_rng(8).normal(0, 0.01, n).astype(np.float32)
    order = np.lexsort((np.arange(n), stamps / 1e9))
    med = [np.median(xyz[p, 2].astype(float)) for p in np.array_split(order, 8)]
    got = F.floor_z_time_range(view(dict(mesh_xyz=xyz, mesh_label=lab, mesh_t_first=stamps)))
    assert got == pytest.approx(max(med) - min(med), rel=1e-12)


def test_revisit_offset_recovers_a_doubled_surface():
    n2 = 6000
    r = np.random.default_rng(5)
    base = np.c_[r.uniform(0, 4, n2), r.uniform(0, 4, n2), np.zeros(n2)]
    second = r.random(n2) < 0.5
    t = np.where(second, 200e9, 100e9).astype(np.int64)
    est = []
    for off in (0.0, 0.05, 0.2):
        xyz = base.copy()
        xyz[second, 2] = off
        est.append(F.mesh_revisit_offset_median_k96(view(dict(mesh_xyz=xyz.astype(np.float32), mesh_label=np.full(n2, 3, np.int16), mesh_t_first=t))))
    assert est[0] < 0.01 and 0.025 < est[1] < 0.055 and 0.08 < est[2] < 0.21
    same_time = view(dict(mesh_xyz=base.astype(np.float32), mesh_label=np.full(n2, 3, np.int16), mesh_t_first=np.full(n2, 100e9, np.int64)))
    assert np.isnan(F.mesh_revisit_offset_median(same_time))


# ---------------------------------------------------------------- compute() contract
def test_compute_returns_every_feature_finite_on_a_rich_scene(scene, perturbed):
    values, errors = F.compute(view(perturbed), view(scene))
    assert list(values) == list(F.REGISTRY) and errors == {}
    assert all(math.isfinite(v) for v in values.values()), {k: v for k, v in values.items() if not math.isfinite(v)}


def test_reference_based_features_equal_their_ideal_for_the_reference_itself(scene):
    values, errors = F.compute(view(scene), view(scene))
    assert errors == {}
    for n, s in F.REGISTRY.items():
        if s.kind == "reference_based":
            assert values[n] == pytest.approx(s.ideal, abs=1e-9), n


def test_perturbed_run_moves_the_features_of_the_layers_it_touches(scene, perturbed):
    values, _ = F.compute(view(perturbed), view(scene))
    moved = {F.REGISTRY[n].layer for n, s in F.REGISTRY.items() if s.kind == "reference_based" and abs(values[n] - s.ideal) > 1e-6}
    assert {"mesh_geom", "mesh_sem", "objects", "input_pose"} <= moved
    assert not moved & {"places", "rooms"}  # same places and rooms in both scenes


def test_compute_is_deterministic_and_survives_the_npz_round_trip(scene, tmp_path):
    snap = sf.snapshot_from_arrays(scene)
    first, _ = F.compute(snap, snap)
    snap.save(tmp_path / "s.npz")
    with sn.Snapshot.load(tmp_path / "s.npz") as loaded:
        second, _ = F.compute(loaded, loaded)
    assert list(first) == list(second) and np.array_equal(list(first.values()), list(second.values()), equal_nan=True)


def test_compute_filters_names_and_skips_reference_based_without_reference(scene):
    v = view(scene)
    free = {n for n, s in F.REGISTRY.items() if s.kind == "reference_free"}
    values, _ = F.compute(v, None)
    assert set(values) == free
    values, _ = F.compute(v, v, names=["place_radius_mean", "obj_count_ratio"])
    assert list(values) == ["obj_count_ratio", "place_radius_mean"]  # registry order
    with pytest.raises(KeyError):
        F.compute(v, None, names=["no_such_feature"])


def test_compute_isolates_failing_and_infinite_features(scene, monkeypatch):
    def boom(run, ref=None):
        raise ValueError("boom")

    def inf(run, ref=None):
        return float("inf")

    for name, fn in (("t_boom", boom), ("t_inf", inf)):
        monkeypatch.setitem(F.REGISTRY, name, F.Spec(name, "reference_free", "places", fn, None, 0.0, 0.0, False, ""))
    values, errors = F.compute(view(scene), None, names=["t_boom", "t_inf", "place_radius_mean"])
    assert math.isnan(values["t_boom"]) and errors["t_boom"] == "ValueError: boom"
    assert math.isnan(values["t_inf"]) and "inf" in errors["t_inf"]
    assert math.isfinite(values["place_radius_mean"]) and "place_radius_mean" not in errors


# ---------------------------------------------------------------- Snapshot -> FeatureView derivations
def test_nodes_with_two_parents_count_as_unattached(scene):
    p2 = int(scene["obj_parent"][2])
    snap = sf.snapshot_from_arrays(scene, extra_parents={2: (p2 + 1) % len(scene["place_pos"]), 0: 9})
    assert snap.objects.n_parents[2] == 2 and p2 >= 0 and scene["obj_parent"][0] == -1
    v = F.FeatureView(snap)
    assert v.obj_parent[2] == -1 and v.obj_room[2] == -1  # two parents
    assert v.obj_parent[0] == 9  # orphan in the arrays, one parent in the snapshot
    assert v.obj_parent[3] == scene["obj_parent"][3]
    snap.places.n_room_parents[4] = 2
    assert F.FeatureView(snap).place_parent[4] == -1 and F.FeatureView(snap).place_parent[5] == scene["place_parent"][5]


def test_unset_mesh_label_and_invalid_box_do_not_break_the_battery(scene):
    snap = sf.snapshot_from_arrays(scene)
    snap.mesh.label[:50] = sn.MESH_NO_LABEL
    snap.objects.bbox_dim[1] = np.nan
    snap.objects.label[2] = sn.NO_LABEL
    v = F.FeatureView(snap)
    assert (v.mesh_label[:50] == -1).all()
    values, errors = F.compute(v, view(scene))
    assert math.isfinite(values["mesh_label_hist_js"]) and math.isfinite(values["obj_class_hist_js"])
    assert math.isnan(values["obj_bbox_iou_median"])  # a NaN box propagates, by design
    assert set(errors) <= {"obj_floor_gap_p90"} and math.isnan(values["obj_floor_gap_p90"])  # the KD-tree refuses a NaN query: NaN plus error text


def test_view_exposes_prototype_attribute_contract(scene):
    snap = sf.snapshot_from_arrays(scene)
    v = F.FeatureView(snap)
    for k, want in scene.items():
        if k.startswith("stats_layer"):
            got = v.stats_last(int(k[len("stats_layer")]))
        else:
            got = getattr(v, k)
        assert np.shape(got) == np.shape(want), k
        assert np.allclose(got, want, equal_nan=True), k
    assert v.name == "syn" and v.t_first_s.shape == scene["mesh_t_first"].shape
    assert snap.check() == []


# ---------------------------------------------------------------- degenerate inputs
@pytest.mark.parametrize("kind", sf.DEGENERATE)
def test_degenerate_scenes_give_finite_or_nan_and_never_raise(kind, scene):
    deg = sf.degenerate(scene, kind)
    assert sf.snapshot_from_arrays(deg).check() == []
    for run_a, ref_a in ((deg, scene), (scene, deg), (deg, deg)):
        values, errors = F.compute(view(run_a), view(ref_a))
        assert list(values) == list(F.REGISTRY)
        for n, v in values.items():
            assert not math.isinf(v), n
            assert n not in errors or math.isnan(v), n
        values_free, _ = F.compute(view(run_a), None)
        assert all(not math.isinf(v) for v in values_free.values())


def test_missing_layer_gives_the_expected_ratios(scene):
    ref = view(scene)
    cases = {"no_objects": ("obj_count_ratio", 0.0), "no_places": ("place_count_ratio", 0.0), "no_rooms": ("room_count_ratio", 0.0),
             "empty_mesh": ("mesh_vertex_count_ratio", 0.0)}
    for kind, (name, want) in cases.items():
        values, _ = F.compute(view(sf.degenerate(scene, kind)), ref, names=[name, "mesh_vertex_count_ratio"])
        assert values[name] == want, kind
        if kind != "empty_mesh":
            assert values["mesh_vertex_count_ratio"] == 1.0
