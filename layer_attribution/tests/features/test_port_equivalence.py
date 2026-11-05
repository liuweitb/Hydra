"""The ported battery (Snapshot -> FeatureView) gives the prototype's numbers on the same arrays.

Runs research/signal_probe/features_proto.py on synthetic scenes next to features.py; only differences that are documented in
features.py are allowed (labels outside 0..20 are not counted in histograms).
"""

import math

import pytest
import scene_factory as sf

from layer_attribution import features as F

proto = sf.load_prototype()
pytestmark = pytest.mark.skipif(proto is None, reason="research/signal_probe/features_proto.py not present")

VARIANTS = {
    "jitter_relabel_drop": dict(jitter=0.02, relabel=0.25, drop_objects=4, drop_frames=True),
    "z_drift_other_places": dict(drift=0.3, n_place=90, n_obj=30),
}


def _same(a, b):
    return (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-12 + 1e-10 * abs(b)


def test_registry_matches_the_prototype():
    assert list(F.REGISTRY) == list(proto.REGISTRY)
    for n, s in F.REGISTRY.items():
        o = proto.REGISTRY[n]
        assert (s.kind, s.layer, s.ideal, s.tol_abs, s.tol_rel, s.count_based, s.unit) == (o.kind, o.layer, o.ideal, o.tol_abs, o.tol_rel, o.count_based, o.unit), n


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_feature_equals_the_prototype(variant):
    ref_arrays = sf.make_scene(0)
    run_arrays = sf.make_scene(0, **VARIANTS[variant])
    old = proto.compute_all(sf.prototype_snap(proto, run_arrays), sf.prototype_snap(proto, ref_arrays))
    new, errors = F.compute(sf.snapshot_from_arrays(run_arrays), sf.snapshot_from_arrays(ref_arrays))
    assert errors == {} and list(new) == list(old)
    bad = {n: (new[n], old[n]) for n in old if not _same(new[n], old[n])}
    assert not bad, bad


def test_two_parent_objects_match_the_prototype_convention():
    # the prototype read has_parent()/get_parent(): an object with two parent places had none
    ref_arrays = sf.make_scene(1)
    run_arrays = sf.make_scene(1, jitter=0.01)
    n_place = len(run_arrays["place_pos"])
    two = {i: (int(run_arrays["obj_parent"][i]) + 1) % n_place for i in (2, 5)}  # both objects have a parent in this scene
    proto_arrays = dict(run_arrays)
    parent = run_arrays["obj_parent"].copy()
    parent[list(two)] = -1
    proto_arrays["obj_parent"] = parent
    old = proto.compute_all(sf.prototype_snap(proto, proto_arrays), sf.prototype_snap(proto, ref_arrays))
    new, errors = F.compute(sf.snapshot_from_arrays(run_arrays, extra_parents=two), sf.snapshot_from_arrays(ref_arrays))
    assert errors == {}
    bad = {n: (new[n], old[n]) for n in old if not _same(new[n], old[n])}
    assert not bad, bad
