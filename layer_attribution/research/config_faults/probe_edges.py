#!/usr/bin/env python3
import pathlib
import sys

import numpy as np
import spark_dsg as dsg

g = dsg.DynamicSceneGraph.load(str(pathlib.Path(sys.argv[1]) / "backend" / "dsg_with_mesh.json"))
layer = g.get_layer(dsg.DsgLayers.PLACES)
print("layer attrs:", [x for x in dir(layer) if not x.startswith("_")])
e = next(iter(layer.edges))
print("edge attrs:", [x for x in dir(e) if not x.startswith("_")])
print("info attrs:", [x for x in dir(e.info) if not x.startswith("_")])
print(e.source, e.target, e.info.weight, e.info.weighted)
w = np.array([x.info.weight for x in layer.edges])
print("n edges", len(w), "weight quantiles", np.quantile(w, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]).round(3))
print("unique weights", len(np.unique(w.round(9))))
n = next(iter(layer.nodes))
print("node siblings API:", [x for x in dir(n) if not x.startswith("_")])
print("num_siblings of first", len(list(n.siblings())))
