#!/usr/bin/env python3
"""Print the config dir hydra_python uses by default, and diff uhumans2 files against D:/Hydra/config."""

import hashlib
import pathlib

import hydra_python
from hydra_python.pipeline import get_config_path, load_configs

p = get_config_path()
print("hydra_python module:", pathlib.Path(hydra_python.__file__).resolve())
print("default config path:", p, "->", p.resolve())
other = pathlib.Path("/mnt/d/Hydra/config")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


for rel in ("datasets/uhumans2.yaml", "label_spaces/uhumans2_office_label_space.yaml"):
    a, b = p / rel, other / rel
    print(rel, "default:", sha(a), "D:/Hydra/config:", sha(b), "SAME" if sha(a) == sha(b) else "DIFFERENT")

c = load_configs("uhumans2", "uhumans2_office")
print("top-level keys of composed config:", list(c))
print("frontend keys:", list(c["frontend"]))
print("backend keys:", list(c["backend"]))
