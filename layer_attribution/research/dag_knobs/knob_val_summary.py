"""Per-run summary of the truncated knob-validation runs, compared with tmp_dk_kv_base.

usage: python knob_val_summary.py RUNS_ROOT OUT_JSON
Reads <run>/backend/dsg_with_mesh.json; a run without it is reported as missing.
"""
import collections
import json
import os
import sys

import numpy as np
import spark_dsg as dsg

NAMES = ["kv_base", "kv_inval", "kv_gvdmin", "kv_meshres", "kv_minedge"]  # overridden by argv[3:]; first name is the base


def nid(x):
    return int(x.value) if hasattr(x, "value") else int(x)


def key_str(k):
    return f"{k.layer}" if k.partition == 0 else f"{k.layer}[{k.partition}]"


def summarize(path):
    G = dsg.DynamicSceneGraph.load(path)
    layers = collections.defaultdict(list)
    for n in G.nodes:
        layers[key_str(n.layer)].append(n)
    out = {"layer_counts": {k: len(v) for k, v in sorted(layers.items())}}
    V = np.asarray(G.mesh.get_vertices())
    out["mesh_vertices"] = int(G.mesh.num_vertices())
    out["mesh_faces"] = int(G.mesh.num_faces())
    pl = G.get_layer("PLACES")
    out["places_edges"] = len(list(pl.edges))
    out["parent_child_edges"] = collections.Counter(
        f"{key_str(G.get_node(p).layer)}->{key_str(n.layer)}" for ns in layers.values() for n in ns for p in n.parents()
    )
    out["parent_child_edges"] = dict(out["parent_child_edges"])
    out["room_sizes"] = sorted(len(list(r.children())) for r in G.get_layer("ROOMS").nodes)
    out["mesh_pos"] = V[:3].T if V.shape[0] in (3, 6) else V[:, :3]
    pos = {k: np.array([np.asarray(n.attributes.position, dtype=float) for n in v]) for k, v in layers.items()}
    out["pos"] = pos
    return out


def main(root, out_path, names=None):
    names = names or NAMES
    S = {}
    for n in names:
        p = f"{root}/tmp_dk_{n}/backend/dsg_with_mesh.json"
        S[n] = summarize(p) if os.path.exists(p) else None
    res = {}
    base = S[names[0]]
    for n in names:
        s = S[n]
        if s is None:
            res[n] = "missing (run did not produce a saved graph)"
            continue
        r = {k: s[k] for k in ("layer_counts", "mesh_vertices", "mesh_faces", "places_edges", "parent_child_edges", "room_sizes")}
        if base is not None and n != names[0]:
            r["mesh_vertices_ratio_vs_base"] = s["mesh_vertices"] / base["mesh_vertices"]
            r["mesh_positions_identical_to_base"] = bool(s["mesh_pos"].shape == base["mesh_pos"].shape and np.allclose(s["mesh_pos"], base["mesh_pos"], atol=0))
            ident = {}
            for k in sorted(set(s["pos"]) | set(base["pos"])):
                a, b = s["pos"].get(k), base["pos"].get(k)
                if a is None or b is None:
                    ident[k] = "layer missing in one run"
                elif a.shape != b.shape:
                    ident[k] = f"count {len(b)} -> {len(a)}"
                elif len(a) == 0:
                    ident[k] = "empty"
                else:
                    ia, ib = np.lexsort(a.T[::-1]), np.lexsort(b.T[::-1])
                    ident[k] = f"same count {len(a)}, max |dpos| = {float(np.abs(a[ia] - b[ib]).max()):.4g} m"
            r["layers_vs_base"] = ident
        res[n] = r
    json.dump(res, open(out_path, "w"), indent=1, sort_keys=True)
    print(json.dumps(res, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or None)
