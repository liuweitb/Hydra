"""Second batch of config dirs for truncated knob-validation runs (one knob per dir, kv2_*).

Keys that do not exist in the shipped uhumans2.yaml are added (nested dicts created); they are real C++ fields.
usage: python make_knob_val2_configs.py OUT_DIR
"""
import copy
import pathlib
import shutil
import sys

import yaml

REPO_CFG = pathlib.Path("D:/Hydra/config")
OUT = pathlib.Path(sys.argv[1])


def write(name, changes):
    root = OUT / name
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    (root / "label_spaces").mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(yaml.safe_load((REPO_CFG / "datasets" / "uhumans2.yaml").read_text()))
    for path, value in changes.items():
        node = cfg
        parts = path.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    (root / "datasets" / "uhumans2.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    shutil.copy(REPO_CFG / "label_spaces" / "uhumans2_office_label_space.yaml", root / "label_spaces")


write("kv2_base", {})
write("kv2_meshres3", {"frontend.pgmo.mesh_resolution": 0.3})
write("kv2_meshw20", {"active_window.mesh.min_weight": 20.0})
write("kv2_meshw200", {"active_window.mesh.min_weight": 200.0})
write("kv2_maxclu", {"frontend.objects.max_cluster_size": 200})
write("kv2_mergedist", {"frontend.freespace_places.graph.node_merge_distance_m": 2.5})
write("kv2_mindiff", {"frontend.freespace_places.gvd.min_diff_m": 0.25})
write("kv2_dsratio", {"frontend.freespace_places.tsdf_interpolator.ratio": 4})
write(
    "kv2_inert",
    {
        "backend.pgmo.covariance.odom": 0.5,
        "backend.pgmo.optimizer.gnc.max_iterations": 7,
        "backend.add_places_to_deformation_graph": False,
        "backend.optimize_on_lc": False,
        "frontend.pgmo.d_graph_resolution": 5.0,
        "frontend.pgmo.time_horizon": 3.0,
    },
)
print("written", sorted(p.name for p in OUT.iterdir() if p.name.startswith("kv2_")))
