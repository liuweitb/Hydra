"""Config dirs for the truncated knob-validation runs (one key changed per dir).

kv_base   : repo config unchanged
kv_inval  : frontend.objects.min_cluster_size above max_cluster_size (violates a check(), expect termination)
kv_gvdmin : gvd.min_distance_m above the TSDF truncation distance (expect no places, no error exit)
kv_meshres: frontend.pgmo.mesh_resolution 0.005 -> 0.1 (DSG mesh vertex merging only)
kv_minedge: graph.min_edge_distance_m 0.25 -> 1.0 (place edges below this clearance dropped)
usage: python make_knob_val_configs.py OUT_DIR
"""
import copy
import pathlib
import shutil
import sys

import yaml

REPO_CFG = pathlib.Path("D:/Hydra/config")
OUT = pathlib.Path(sys.argv[1])


def write(name, mutate):
    root = OUT / name
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    (root / "label_spaces").mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(yaml.safe_load((REPO_CFG / "datasets" / "uhumans2.yaml").read_text()))
    mutate(cfg)
    (root / "datasets" / "uhumans2.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    shutil.copy(REPO_CFG / "label_spaces" / "uhumans2_office_label_space.yaml", root / "label_spaces")


def setk(path, value):
    def f(cfg):
        node = cfg
        for p in path.split(".")[:-1]:
            node = node[p]  # KeyError if the key is not in the shipped yaml: no silent new keys
        last = path.split(".")[-1]
        assert last in node, path
        node[last] = value

    return f


write("kv_base", lambda c: None)
write("kv_inval", setk("frontend.objects.min_cluster_size", 200000))
write("kv_gvdmin", setk("frontend.freespace_places.gvd.min_distance_m", 0.35))
write("kv_meshres", setk("frontend.pgmo.mesh_resolution", 0.1))
write("kv_minedge", setk("frontend.freespace_places.graph.min_edge_distance_m", 1.0))
print("written", sorted(p.name for p in OUT.iterdir() if p.name.startswith("kv_")))
