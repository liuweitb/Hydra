"""Build three tiny config dirs (datasets/ + label_spaces/) for the apply-vs-ignore test.

base : repo config unchanged
mis  : base + keys that match no C++ field (misspelled / wrong namespace) + one type error
set  : base + two real keys changed
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
    cfg = yaml.safe_load((REPO_CFG / "datasets" / "uhumans2.yaml").read_text())
    cfg = copy.deepcopy(cfg)
    mutate(cfg)
    (root / "datasets" / "uhumans2.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    shutil.copy(REPO_CFG / "label_spaces" / "uhumans2_office_label_space.yaml", root / "label_spaces")


def mis(cfg):
    cfg["frontend"]["objects"]["min_cluster_sizee"] = 400  # misspelled
    cfg["frontend"]["min_cluster_size"] = 400  # right name, wrong namespace
    cfg["objects"] = {"min_cluster_size": 400}  # right name, top level
    cfg["frontend"]["objects"]["cluster_tolerance"] = "abc"  # type error on a real key


def setv(cfg):
    cfg["frontend"]["objects"]["min_cluster_size"] = 400
    cfg["backend"]["update_functors"]["rooms"]["room_finder"]["min_room_size"] = 30


write("base", lambda c: None)
write("mis", mis)
write("set", setv)
print("written", [p.name for p in OUT.iterdir()])


def bad_type_a(cfg):
    cfg["frontend"]["surface_places"]["type"] = "place_2dd"
    cfg["backend"]["update_functors"]["rooms"]["type"] = "UpdateRoomsFunctorr"


def bad_type_b(cfg):
    cfg["active_window"]["tsdf"]["semantic_integrator"]["type"] = "SingleLabelIntegratorr"


write("typea", bad_type_a)
write("typeb", bad_type_b)
