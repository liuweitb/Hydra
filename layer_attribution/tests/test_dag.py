import pytest

from layer_attribution import dag
from layer_attribution.pipeline import config_faults as cf


def test_ancestors_follow_the_dataflow_dag():
    assert dag.ancestors("mesh") == frozenset() and dag.ancestors("places") == frozenset()
    assert dag.ancestors("objects") == {"mesh"} and dag.ancestors("rooms") == {"places"}


def test_chain_dag_makes_places_and_rooms_downstream_of_mesh():
    assert dag.ancestors("places", "chain") == {"mesh"} and dag.ancestors("rooms", "chain") == {"mesh", "places"}


def test_roots_keep_only_unexplained_layers():
    assert dag.roots({"mesh", "objects", "places", "rooms"}) == ["mesh", "places"]
    assert dag.roots({"objects"}) == ["objects"]  # objects deviate on their own: a label or object-config fault
    assert dag.roots({"objects", "rooms"}) == ["objects", "rooms"]  # two unexplained layers
    assert dag.roots({"mesh", "objects", "places", "rooms"}, "chain") == ["mesh"]
    assert dag.roots(set()) == []


def test_every_injectable_fault_has_an_accept_set_inside_the_layers():
    for fault, layers in dag.ACCEPT.items():
        assert layers and set(layers) <= set(dag.LAYERS), fault
    with pytest.raises(KeyError):
        dag.accept_set("not_a_fault")


def test_config_fault_accept_sets_match_the_layer_each_fault_targets():
    for name, spec in cf.FAULTS.items():
        assert dag.accept_set(name) == (spec.layer,), name


def test_feature_layer_mapping():
    assert dag.layer_of_feature("mesh_sem") == "objects" and dag.layer_of_feature("mesh_geom") == "mesh"
    assert dag.layer_of_feature("process") is None and dag.layer_of_feature("input_pose") is None
