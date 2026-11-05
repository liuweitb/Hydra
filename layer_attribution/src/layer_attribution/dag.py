"""Layer dependency structure, fault ground truth and feature-group mapping (single source of truth).

Sources: research/dag_knobs/hydra_dependency_dag.md (code-cited). Evidence layers are mesh (geometry only), objects (object
nodes and the mesh per-vertex label channel, which is the object layer's input), places (GVD free-space places) and rooms.
"""

LAYERS = ("mesh", "objects", "places", "rooms")

# feature layer tag (features.py) -> evidence layer; other tags (edges, input_pose, process) are not layer evidence
FEATURE_LAYER = {"mesh_geom": "mesh", "mesh_sem": "objects", "objects": "objects", "places": "places", "rooms": "rooms"}

# data-flow DAG: layer -> layers whose corruption can change it. mesh and places are siblings under the TSDF.
DAGS = {
    "dataflow": {"mesh": (), "places": (), "objects": ("mesh",), "rooms": ("places",)},
    # the chain the proposal assumed, kept for the sensitivity analysis
    "chain": {"mesh": (), "objects": ("mesh",), "places": ("mesh",), "rooms": ("places", "mesh")},
    # no dependency knowledge: every layer is a root (the naive baselines ignore it anyway)
    "flat": {"mesh": (), "objects": (), "places": (), "rooms": ()},
}

SENSOR_ACCEPT = {
    "label_confusion": ("objects",),
    "label_dropout": ("objects",),
    "depth_noise": ("mesh", "places"),
    "depth_holes": ("mesh", "places"),
    "pose_drift": ("mesh", "places"),
    "frame_drop": ("mesh", "places"),
}
CONFIG_ACCEPT = {
    "objects_min_size": ("objects",), "objects_tol_merge": ("objects",), "objects_tol_split": ("objects",),
    "rooms_min_size": ("rooms",), "rooms_min_component": ("rooms",), "rooms_plateau_ratio": ("rooms",),
    "rooms_max_dilation": ("rooms",), "rooms_min_dilation": ("rooms",),
    "places_compression": ("places",), "places_min_node_dist": ("places",), "places_max_dist": ("places",),
}
ACCEPT = {**SENSOR_ACCEPT, **CONFIG_ACCEPT}


def accept_set(fault: str) -> tuple:
    """Layers that count as a correct top-1 for this injected fault."""
    try:
        return ACCEPT[fault]
    except KeyError:
        raise KeyError(f"no ground-truth layer for fault {fault!r}") from None


def ancestors(layer: str, dag: str = "dataflow") -> frozenset:
    """All layers upstream of `layer` in the named DAG."""
    parents, seen, todo = DAGS[dag], set(), list(DAGS[dag][layer])
    while todo:
        p = todo.pop()
        if p not in seen:
            seen.add(p)
            todo.extend(parents[p])
    return frozenset(seen)


def roots(deviating, dag: str = "dataflow") -> list:
    """Deviating layers with no deviating ancestor, in LAYERS order."""
    dev = set(deviating)
    return [l for l in LAYERS if l in dev and not (ancestors(l, dag) & dev)]


def layer_of_feature(tag: str):
    """Evidence layer of a feature layer tag, None when the tag is not layer evidence."""
    return FEATURE_LAYER.get(tag)
