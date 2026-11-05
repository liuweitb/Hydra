"""Two config dirs for the functor-order test.

ordbase : repo config unchanged
ordtest : backend.update_functors key 'buildings' renamed to 'zz_buildings' (same functor type), so that the
          alphabetically ordered std::map in DsgUpdater runs it after 'rooms'.
usage: python make_order_test_configs.py OUT_DIR
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


def rename_buildings(cfg):
    uf = cfg["backend"]["update_functors"]
    uf["zz_buildings"] = uf.pop("buildings")


write("ordbase", lambda c: None)
write("ordtest", rename_buildings)
print("written", sorted(p.name for p in OUT.iterdir()))
