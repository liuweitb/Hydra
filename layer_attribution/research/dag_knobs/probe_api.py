"""Probe spark_dsg python API on a reference run (no mesh load)."""
import sys
import spark_dsg as dsg

path = sys.argv[1]
G = dsg.DynamicSceneGraph.load(path)
print("version", getattr(dsg, "__version__", "?"))
print("layer_keys", list(G.layer_keys))
print("layer_names", list(G.layer_names))
print("has_mesh", G.has_mesh())
for layer in G.layers:
    print("layer", layer.id, "num_nodes", layer.num_nodes() if hasattr(layer, "num_nodes") else "?")
    break
print([a for a in dir(G) if not a.startswith("_")])
n = next(iter(G.nodes))
print([a for a in dir(n) if not a.startswith("_")])
print("layer attr", n.layer, type(n.layer))
print([a for a in dir(n.layer) if not a.startswith("_")])
