"""Structural facts of a saved scene graph (no mesh load): layers, interlayer edges, node categories.

Reads only a reference run (uh2_clean). Output: json to argv[2].
"""
import collections
import json
import sys

import spark_dsg as dsg


def key_str(k):
    return f"{k.layer}" if k.partition == 0 else f"{k.layer}[{k.partition}]"


def describe(G):
    out = {"layer_keys": [key_str(k) for k in G.layer_keys], "layer_names": list(G.layer_names)}
    lk = {}
    for name in G.layer_names:
        try:
            lk[name] = key_str(G.get_layer_key(name))
        except Exception as e:  # noqa: BLE001
            lk[name] = f"ERR {e}"
    out["name_to_key"] = lk

    nodes_by_key = collections.defaultdict(list)
    for n in G.nodes:
        nodes_by_key[key_str(n.layer)].append(n)

    layers = {}
    for k, nodes in sorted(nodes_by_key.items()):
        cats = collections.Counter(str(n.id)[0] for n in nodes)
        labels = collections.Counter()
        for n in nodes:
            a = n.attributes
            if hasattr(a, "semantic_label"):
                labels[int(a.semantic_label)] += 1
        attr_types = collections.Counter(type(n.attributes).__name__ for n in nodes)
        n_sib = sum(len(list(n.siblings())) for n in nodes)
        layers[k] = {
            "num_nodes": len(nodes),
            "id_categories": dict(cats),
            "attr_types": dict(attr_types),
            "semantic_labels": dict(sorted(labels.items())),
            "intra_layer_edge_endpoints": n_sib,
        }
    out["layers"] = layers

    # parent -> child counts, number of parents per node
    pc = collections.Counter()
    parents_per_node = collections.defaultdict(collections.Counter)
    sib_cross = collections.Counter()
    for k, nodes in nodes_by_key.items():
        for n in nodes:
            ps = list(n.parents())
            parents_per_node[k][len(ps)] += 1
            for pid in ps:
                pk = key_str(G.get_node(pid).layer)
                pc[f"{pk} -> {k}"] += 1
            for sid in n.siblings():
                sk = key_str(G.get_node(sid).layer)
                if sk != k:
                    sib_cross[f"{k} ~ {sk}"] += 1
    out["parent_child_edge_counts"] = dict(pc)
    out["num_parents_histogram_per_layer"] = {k: dict(v) for k, v in parents_per_node.items()}
    out["cross_partition_sibling_endpoints"] = dict(sib_cross)

    # attribute stats
    stats = {}
    objs = nodes_by_key.get("2", [])
    if objs:
        mc = [len(a.attributes.mesh_connections) for a in objs]
        stats["objects"] = {
            "mesh_connections_min_med_max": [min(mc), sorted(mc)[len(mc) // 2], max(mc)],
            "attr_fields": [a for a in dir(objs[0].attributes) if not a.startswith("_")],
        }
    pl = nodes_by_key.get("3", [])
    if pl:
        d = [float(n.attributes.distance) for n in pl]
        stats["places"] = {
            "radius_min_med_max": [min(d), sorted(d)[len(d) // 2], max(d)],
            "real_place_count": sum(1 for n in pl if getattr(n.attributes, "real_place", True)),
            "attr_fields": [a for a in dir(pl[0].attributes) if not a.startswith("_")],
        }
    mp = nodes_by_key.get("3[1]", [])
    if mp:
        stats["mesh_places"] = {
            "pcl_mesh_connections_min_med_max": [
                min(len(n.attributes.pcl_mesh_connections) for n in mp),
                sorted(len(n.attributes.pcl_mesh_connections) for n in mp)[len(mp) // 2],
                max(len(n.attributes.pcl_mesh_connections) for n in mp),
            ],
            "attr_fields": [a for a in dir(mp[0].attributes) if not a.startswith("_")],
        }
    rooms = nodes_by_key.get("4", [])
    if rooms:
        stats["rooms"] = {"children_per_room": sorted(len(list(r.children())) for r in rooms)}
    out["attr_stats"] = stats

    # edges within a layer, edge attributes of the places layer
    try:
        pl_layer = G.get_layer("PLACES")
        ws = [float(e.info.weight) for e in pl_layer.edges]
        out["places_edge_weight_min_med_max"] = [min(ws), sorted(ws)[len(ws) // 2], max(ws)] if ws else None
        out["places_num_edges"] = len(ws)
    except Exception as e:  # noqa: BLE001
        out["places_edge_err"] = str(e)
    return out


if __name__ == "__main__":
    G = dsg.DynamicSceneGraph.load(sys.argv[1])
    res = describe(G)
    with open(sys.argv[2], "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print(json.dumps(res, indent=1, sort_keys=True)[:6000])
