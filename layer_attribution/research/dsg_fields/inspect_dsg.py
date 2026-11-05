#!/usr/bin/env python3
"""Inventory of a saved Hydra scene graph through the spark_dsg Python API.

Reports per layer: attribute class, which fields are populated (differ from a default-constructed
attribute object), value ranges and real examples; edge classes (intra/inter layer, parent/child,
sibling) with counts, degrees and weights; mesh arrays; label spaces; JSON-level node schemas.

usage (WSL, via research/dsg_fields/wsl_py.sh):
    inspect_dsg.py RUN_DIR_OR_JSON [--out-dir DIR] [--no-json-schema]
"""

import argparse
import datetime
import json
import pathlib
import resource
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import spark_dsg as dsg

SKIP_FIELDS = {"NO_SEMANTIC_LABEL", "metadata"}
LABEL_CATS = {"O": "OBJECTS", "a": "AGENTS", "p": "PLACES", "P": "MESH_PLACES", "R": "ROOMS", "B": "BUILDINGS"}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, *parts):
        line = " ".join(str(p) for p in parts)
        self.lines.append(line)
        print(line, flush=True)

    def section(self, title):
        self("")
        self("=" * 100)
        self(title)
        self("=" * 100)


def canon(v):
    """Hashable, exactly comparable form of an attribute value (for default / distinct tests)."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, np.ndarray):
        return ("nd", v.shape, v.dtype.str, v.tobytes())
    if isinstance(v, (list, tuple)):
        return ("l", tuple(canon(x) for x in v))
    if isinstance(v, (set, frozenset)):
        return ("s", tuple(sorted(canon(x) for x in v)))
    if isinstance(v, dict):
        return ("d", tuple(sorted((str(k), canon(x)) for k, x in v.items())))
    if isinstance(v, datetime.timedelta):
        return ("td", v.total_seconds())
    if isinstance(v, dsg.BoundingBox):
        return ("bb", int(v.type), tuple(np.asarray(v.dimensions)), tuple(np.asarray(v.world_P_center)),
                tuple(np.asarray(v.world_R_center).ravel()))
    if isinstance(v, dsg.Quaternion):
        return ("q", v.w, v.x, v.y, v.z)
    if isinstance(v, dsg.Color):
        return ("c", v.r, v.g, v.b, v.a)
    if isinstance(v, dsg.NearestVertexInfo):
        return ("nvi", repr(v))
    return ("repr", repr(v))


def short(v, n=110):
    s = repr(v).replace("\n", " ")
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


def public_fields(attrs):
    out = []
    for name in dir(type(attrs)):
        if name.startswith("_") or name in SKIP_FIELDS:
            continue
        try:
            v = getattr(attrs, name)
        except Exception:
            continue
        if callable(v):
            continue
        out.append(name)
    return out


def describe_field(name, values, default):
    """One summary row for a field across the nodes of a layer."""
    n = len(values)
    cd = canon(default)
    non_default = [i for i, v in enumerate(values) if canon(v) != cd]
    row = {"field": name, "n": n, "n_populated": len(non_default), "kind": type(values[0]).__name__}
    ex_i = non_default[0] if non_default else 0
    row["example"] = short(values[ex_i])
    sample = values[0]
    if isinstance(sample, (bool, int, float, np.integer, np.floating)):
        a = np.asarray(values, dtype=float)
        row.update(min=float(a.min()), max=float(a.max()), mean=float(a.mean()), n_distinct=int(len(np.unique(a))))
    elif isinstance(sample, str):
        c = Counter(values)
        row["n_distinct"] = len(c)
        row["top"] = c.most_common(4)
    elif isinstance(sample, np.ndarray):
        shapes = Counter(v.shape for v in values)
        row["shapes"] = dict(shapes)
        if sample.ndim == 1 and sample.size and sample.dtype.kind in "fiu" and len({v.shape for v in values}) == 1:
            a = np.stack(values).astype(float)
            row["min_per_axis"] = np.round(a.min(0), 4).tolist()
            row["max_per_axis"] = np.round(a.max(0), 4).tolist()
    elif isinstance(sample, (list, tuple)):
        lens = np.array([len(v) for v in values])
        row.update(len_min=int(lens.min()), len_mean=float(lens.mean()), len_max=int(lens.max()), len_total=int(lens.sum()))
        flat = [x for v in values for x in v]
        if flat and isinstance(flat[0], (int, np.integer)):
            row["elem_min"], row["elem_max"] = int(min(flat)), int(max(flat))
    elif isinstance(sample, dsg.BoundingBox):
        row["types"] = dict(Counter(v.type.name for v in values))
        valid = [v for v in values if v.type.name != "INVALID"]
        if valid:
            d = np.stack([np.asarray(v.dimensions) for v in valid])
            row["dims_min"], row["dims_max"] = np.round(d.min(0), 3).tolist(), np.round(d.max(0), 3).tolist()
            row["volume_min_max_median"] = [float(f(np.prod(d, 1))) for f in (np.min, np.max, np.median)]
            row["n_rotated"] = int(sum(v.has_rotation() for v in valid))
    elif isinstance(sample, dsg.Quaternion):
        row["n_identity"] = int(sum(canon(v) == ("q", 1.0, 0.0, 0.0, 0.0) for v in values))
    elif isinstance(sample, datetime.timedelta):
        s = np.array([v.total_seconds() for v in values])
        row.update(min=float(s.min()), max=float(s.max()), n_distinct=int(len(np.unique(s))))
    elif isinstance(sample, dict):
        row["len_max"] = max(len(v) for v in values)
    return row


def metadata_nonempty(nodes):
    cnt = 0
    for n in nodes:
        try:
            md = n.attributes.metadata.get()
            cnt += bool(md)
        except Exception:
            pass
    return cnt


def layer_name_map(g, nodes_by_key):
    """Name by node id category: layer_names maps AGENTS onto the object key, but agents sit in partition 97."""
    return {k: "/".join(sorted({LABEL_CATS.get(n.id.category, n.id.category) for n in nodes})) for k, nodes in nodes_by_key.items() if nodes}


def default_instance(cls):
    try:
        return cls()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=pathlib.Path, help="run dir (contains backend/) or a dsg json file")
    ap.add_argument("--out-dir", type=pathlib.Path, default=None)
    ap.add_argument("--no-json-schema", action="store_true", help="skip the raw-JSON schema pass (extra ~3 s, ~0.5 GB)")
    args = ap.parse_args()

    path = args.path.expanduser()
    if path.is_dir():
        run_dir = path
        path = path / "backend" / "dsg_with_mesh.json"
    else:
        run_dir = path.parent.parent
    R = Report()
    summary = {"path": str(path), "layers": {}, "edges": {}, "mesh": {}}

    R.section(f"FILE {path}")
    t0 = time.time()
    g = dsg.DynamicSceneGraph.load(str(path))
    summary["load_seconds"] = round(time.time() - t0, 2)
    summary["load_peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    R("size bytes", path.stat().st_size, "| spark_dsg.DynamicSceneGraph.load seconds", summary["load_seconds"],
      "| peak RSS MB", summary["load_peak_rss_mb"])
    R("num_nodes", g.num_nodes(), "(unpartitioned only:", g.num_nodes(False), ") num_edges", g.num_edges(),
      "(unpartitioned only:", g.num_edges(False), ")  has_mesh", g.has_mesh())
    R("layer_keys", [str(k) for k in g.layer_keys], " layer_ids", list(g.layer_ids))
    R("layer_names", {k: str(v) for k, v in g.layer_names.items()})
    R("graph metadata keys", list(g.metadata.get().keys()))

    # ---- layers
    R.section("LAYERS AND NODE ATTRIBUTES")
    node_layer = {}
    node_index = {}
    nodes_by_key = {}
    for key in g.layer_keys:
        lv = g.get_layer(key.layer, key.partition)
        nodes = list(lv.nodes)
        nodes_by_key[str(key)] = nodes
        for i, n in enumerate(nodes):
            node_layer[int(n.id.value)] = str(key)
            node_index[int(n.id.value)] = i
    lname = layer_name_map(g, nodes_by_key)

    for key in g.layer_keys:
        ks = str(key)
        nodes = nodes_by_key[ks]
        R("")
        R(f"--- layer {ks} ({lname.get(ks, '?')}): {len(nodes)} nodes")
        if not nodes:
            continue
        classes = Counter(type(n.attributes).__name__ for n in nodes)
        cats = Counter(n.id.category for n in nodes)
        R("attribute class(es):", dict(classes), "| id categories:", dict(cats), "| first ids:", [n.id.str() for n in nodes[:4]])
        cls = type(nodes[0].attributes)
        dflt = default_instance(cls)
        fields = public_fields(nodes[0].attributes)
        rows = []
        for f in fields:
            vals = [getattr(n.attributes, f) for n in nodes]
            d = getattr(dflt, f) if dflt is not None else vals[0]
            rows.append(describe_field(f, vals, d))
        md_n = metadata_nonempty(nodes)
        rows.append({"field": "metadata", "n": len(nodes), "n_populated": md_n, "kind": "dict", "example": "{}" if not md_n else "non-empty"})
        summary["layers"][ks] = {"name": lname.get(ks), "n_nodes": len(nodes), "class": dict(classes), "fields": rows}
        R(f"{'field':26s} {'populated':>13s}  details")
        for r in rows:
            pop = f"{r['n_populated']}/{r['n']}"
            det = []
            for k in ("min", "max", "mean", "n_distinct", "types", "dims_min", "dims_max", "volume_min_max_median", "n_rotated",
                      "n_identity", "len_min", "len_mean", "len_max", "len_total", "elem_min", "elem_max", "shapes",
                      "min_per_axis", "max_per_axis", "top"):
                if k in r:
                    v = r[k]
                    det.append(f"{k}={round(v, 4) if isinstance(v, float) else v}")
            R(f"{r['field']:26s} {pop:>13s}  {' '.join(det)}")
            R(f"{'':26s} {'':>13s}  e.g. {r['example']}")

    # ---- edges
    R.section("EDGES")
    edges = list(g.edges)
    inter = {(e.source, e.target) for e in g.interlayer_edges}
    R("g.edges", len(edges), "| g.interlayer_edges", len(inter), "| g.unpartitioned_edges", sum(1 for _ in g.unpartitioned_edges),
      "| directed flag in file: false (each edge stored once, source/target order is arbitrary)")
    cls_cnt = Counter()
    wstats = defaultdict(list)
    geo = defaultdict(list)
    for e in edges:
        s, t = int(e.source), int(e.target)
        a, b = node_layer[s], node_layer[t]
        cls_cnt[(a, b)] += 1
        wstats[(a, b)].append((e.info.weight, e.info.weighted))
    # canonical unordered pair counts
    pair_cnt = Counter()
    for (a, b), c in cls_cnt.items():
        pair_cnt[tuple(sorted((a, b)))] += c
    R(f"{'layer pair (unordered)':28s} {'count':>6s}  kind               stored source->target orders")
    for pair, c in sorted(pair_cnt.items()):
        a, b = pair
        kind = "intralayer (sibling)" if a == b else "interlayer (parent/child)"
        orders = {f"{x}->{y}": v for (x, y), v in cls_cnt.items() if tuple(sorted((x, y))) == pair}
        R(f"{a + ' - ' + b:28s} {c:6d}  {kind:26s} {orders}")
        summary["edges"][f"{a} - {b}"] = {"count": c, "kind": kind, "stored_orders": orders}
    R("")
    R("edge attributes (EdgeAttributes: weight, weighted, metadata):")
    for (a, b), ws in sorted(wstats.items()):
        arr = np.array([w for w, _ in ws])
        weighted = Counter(x for _, x in ws)
        R(f"  {a}->{b}: n={len(ws)} weighted={dict(weighted)} weight min/mean/max = {arr.min():.4g}/{arr.mean():.4g}/{arr.max():.4g}")

    # weight semantics: node-to-node length or something else
    R("")
    R("edge weight semantics (weighted edge classes only; GVD voxel size 0.2 m assumed for the lattice test):")
    for pair in sorted({tuple(sorted(k)) for k, v in wstats.items() if v[0][1]}):
        w, d, dmin = [], [], []
        for e in edges:
            s, t = int(e.source), int(e.target)
            if tuple(sorted((node_layer[s], node_layer[t]))) != pair:
                continue
            w.append(e.info.weight)
            d.append(float(np.linalg.norm(g.get_position(s) - g.get_position(t))))
            if pair == ("3", "3"):
                dmin.append(min(g.get_node(s).attributes.distance, g.get_node(t).attributes.distance))
        w, d = np.array(w), np.array(d)
        lat = np.mean(np.abs((w / 0.2) ** 2 - np.round((w / 0.2) ** 2)) < 1e-3)
        line = f"  {pair}: corr(weight, node-to-node length) = {np.corrcoef(w, d)[0, 1]:.3f}; weight on 0.2*sqrt(int) lattice: {lat:.1%}"
        if dmin:
            dmin = np.array(dmin)
            line += (f"; corr(weight, min endpoint distance attr) = {np.corrcoef(w, dmin)[0, 1]:.3f}; "
                     f"weight <= min endpoint distance: {np.mean(w <= dmin + 1e-6):.1%}")
        R(line)

    # parent / child / sibling structure per layer
    R("")
    R("per-layer node relations from SceneGraphNode.parents()/children()/siblings():")
    for key in g.layer_keys:
        ks = str(key)
        nodes = nodes_by_key[ks]
        if not nodes:
            continue
        npar = np.array([len(n.parents()) for n in nodes])
        nchi = np.array([len(n.children()) for n in nodes])
        nsib = np.array([len(n.siblings()) for n in nodes])
        par_layers = Counter(node_layer[p] for n in nodes for p in n.parents())
        chi_layers = Counter(node_layer[c] for n in nodes for c in n.children())
        R(f"  {ks} ({lname.get(ks)}): parents/node hist {dict(Counter(npar.tolist()))} -> {dict(par_layers)}; "
          f"children/node min/mean/max {nchi.min()}/{nchi.mean():.2f}/{nchi.max()} -> {dict(chi_layers)}; "
          f"siblings/node min/mean/max {nsib.min()}/{nsib.mean():.2f}/{nsib.max()}")
        summary["layers"][ks]["parents_hist"] = {int(k): v for k, v in Counter(npar.tolist()).items()}
        summary["layers"][ks]["children_minmeanmax"] = [int(nchi.min()), float(nchi.mean()), int(nchi.max())]
        summary["layers"][ks]["siblings_minmeanmax"] = [int(nsib.min()), float(nsib.mean()), int(nsib.max())]
    # odd edges: object stored as source of an object->place edge
    odd = [e for e in edges if node_layer[int(e.source)] == "2" and node_layer[int(e.target)] == "3"]
    if odd:
        R("")
        R(f"{len(odd)} OBJECTS->PLACES stored-order edges (the other {cls_cnt[('3', '2')]} are PLACES->OBJECTS); same parent/child relation, only storage order differs:")
        for e in odd[:4]:
            o = g.get_node(e.source)
            R("  ", dsg.NodeSymbol(e.source).str(), "label", o.attributes.semantic_label, "->", dsg.NodeSymbol(e.target).str(),
              "parents of object:", sorted(dsg.NodeSymbol(p).str() for p in o.parents()))

    # ---- mesh
    R.section("MESH (g.mesh)")
    m = g.mesh
    V = m.get_vertices()
    F = m.get_faces()
    labels = np.array(m.get_labels())
    nv = m.num_vertices()
    R("num_vertices", nv, "num_faces", m.num_faces(), "total_bytes", m.total_bytes())
    R("get_vertices() shape/dtype:", V.shape, V.dtype, "-> rows 0..2 = x,y,z (m); rows 3..5 = r,g,b scaled to 0..1")
    R("  xyz min", np.round(V[:3].min(1), 3).tolist(), "max", np.round(V[:3].max(1), 3).tolist())
    R("  rgb row0..2 min", np.round(V[3:6].min(1), 4).tolist(), "max", np.round(V[3:6].max(1), 4).tolist(),
      "| vertex 0 rgb*255 =", np.round(V[3:6, 0] * 255, 1).tolist(), "vs m.color(0) =", (m.color(0).r, m.color(0).g, m.color(0).b))
    alpha = np.array([m.color(i).a for i in range(0, nv, 97)])
    R("  alpha (every 97th vertex) distinct:", np.unique(alpha).tolist())
    R("get_faces() shape/dtype:", F.shape, F.dtype, "-> 3 x n_faces vertex indices; index range", int(F.min()), int(F.max()))
    ftri = F.T
    degenerate = int(((ftri[:, 0] == ftri[:, 1]) | (ftri[:, 1] == ftri[:, 2]) | (ftri[:, 0] == ftri[:, 2])).sum())
    R("  degenerate faces (repeated index):", degenerate)
    names = dict(g.get_labelspace(2, 0).labels_to_names) if g.get_labelspace(2, 0) else {}
    lc = Counter(labels.tolist())
    R("per-vertex labels (m.get_labels()): list[int] of length", len(labels), "range", int(labels.min()), int(labels.max()))
    R("  label histogram:", {f"{k}:{names.get(k, '?')}": v for k, v in sorted(lc.items(), key=lambda kv: -kv[1])})
    ts = np.array([m.timestamp(i) for i in range(nv)], dtype=np.int64)
    fs = np.array([m.first_seen_timestamp(i) for i in range(nv)], dtype=np.int64)
    R("timestamp(i) ns (last update of the vertex block): min", int(ts.min()), "max", int(ts.max()), "distinct", len(np.unique(ts)),
      "-> seconds", round(ts.min() / 1e9, 2), "..", round(ts.max() / 1e9, 2))
    R("first_seen_timestamp(i) ns: nonzero", int((fs > 0).sum()), "of", nv, f"({(fs > 0).mean():.1%}); min>0", int(fs[fs > 0].min()) if (fs > 0).any() else None,
      "max", int(fs.max()))
    R("example vertex 0:", "pos", m.pos(0), "color", (m.color(0).r, m.color(0).g, m.color(0).b, m.color(0).a), "label", m.label(0),
      "timestamp", m.timestamp(0), "first_seen", m.first_seen_timestamp(0), "face0", F[:, 0].tolist())
    summary["mesh"] = {"num_vertices": nv, "num_faces": int(F.shape[1]), "label_hist": {int(k): v for k, v in lc.items()},
                       "n_first_seen_nonzero": int((fs > 0).sum()), "ts_min": int(ts.min()), "ts_max": int(ts.max()),
                       "n_degenerate_faces": degenerate}

    # mesh_connections consistency (object -> mesh vertices)
    R("")
    R("object mesh_connections index into the mesh (vertex ids); label of the connected vertices vs object semantic_label:")
    for n in nodes_by_key.get("2", [])[:5]:
        mc = np.array(n.attributes.mesh_connections, dtype=np.int64)
        vl = labels[mc]
        R(f"  {n.id.str()}: n_conn={len(mc)} idx range {mc.min()}..{mc.max()} (<{nv}: {mc.max() < nv}) object label {n.attributes.semantic_label}"
          f" -> fraction of connected vertices with that label {np.mean(vl == n.attributes.semantic_label):.3f}")
    # 2D place mesh connections
    p2 = nodes_by_key.get("3[1]", [])
    if p2:
        mc = np.array([x for n in p2[:50] for x in n.attributes.pcl_mesh_connections], dtype=np.int64)
        R(f"  first 50 MESH_PLACES nodes: pcl_mesh_connections total {len(mc)} idx range {mc.min()}..{mc.max()};"
          f" labels of those vertices: {dict(Counter(labels[mc].tolist()).most_common(3))} (node semantic_label {Counter(n.attributes.semantic_label for n in p2[:50])})")
    vp = nodes_by_key.get("3", [])
    if vp:
        n = vp[0]
        R("  PLACES voxblox_mesh_connections of", n.id.str(), ":", [short(v, 160) for v in n.attributes.voxblox_mesh_connections[:2]])
        dists = [float(np.linalg.norm(n.attributes.position - np.asarray(v.voxel_pos))) for v in n.attributes.voxblox_mesh_connections]
        R("   place distance attr", round(n.attributes.distance, 4), "| |place pos - basis voxel_pos|", np.round(dists, 4).tolist())
        res = np.array([n.attributes.distance - min(np.linalg.norm(n.attributes.position - np.asarray(v.voxel_pos))
                                                   for v in n.attributes.voxblox_mesh_connections) for n in vp])
        R(f"   all {len(vp)} places: distance - min_k |pos - basis_k| : |max| {np.abs(res).max():.4g}, within 1e-4 for {np.mean(np.abs(res) < 1e-4):.1%}"
          " (basis voxel_pos = nearest-obstacle voxels; 'block' and 'vertex' of these records are uninitialized memory, see determinism_diff.py)")

    # ---- label spaces
    R.section("LABEL SPACES")
    for lk in g.layer_keys:
        try:
            ls = g.get_labelspace(lk.layer, lk.partition)
        except Exception:
            continue
        if ls:
            R(f"labelspace layer {lk}: {len(ls.labels_to_names)} labels", dict(list(ls.labels_to_names.items())[:21]))
    md = g.metadata.get()
    R("graph.metadata:", {k: (list(v.keys()) if isinstance(v, dict) else v) for k, v in md.items()})

    # ---- raw JSON schema
    if not args.no_json_schema:
        R.section("RAW JSON SCHEMA (what is serialized, node attribute keys per class)")
        t0 = time.time()
        with open(path) as f:
            raw = json.load(f)
        R("top-level keys:", list(raw.keys()), "| parse seconds", round(time.time() - t0, 2))
        R("header:", raw["SPARK_DSG_header"], "| directed", raw["directed"], "| multigraph", raw["multigraph"])
        schema = defaultdict(set)
        cnt = Counter()
        for nd in raw["nodes"]:
            ty = nd["attributes"]["type"]
            cnt[(ty, nd["layer"], nd["partition"])] += 1
            schema[ty].update(nd["attributes"].keys())
        for ty, keys in schema.items():
            R(f"  {ty}: {sorted(keys)}")
        R("  node counts by (type, layer, partition):", {f"{a}/{b}/{c}": v for (a, b, c), v in cnt.items()})
        R("  mesh keys:", sorted(raw["mesh"].keys()))
        R("  node dict keys:", sorted(raw["nodes"][0].keys()), "| edge dict keys:", sorted(raw["edges"][0].keys()), "| edge info keys:", sorted(raw["edges"][0]["info"].keys()))
        summary["json_schema"] = {ty: sorted(keys) for ty, keys in schema.items()}

    out_dir = args.out_dir
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = run_dir.name
        (out_dir / f"inventory_{tag}.txt").write_text("\n".join(R.lines) + "\n", encoding="utf8")
        (out_dir / f"inventory_{tag}.json").write_text(json.dumps(jsonable(summary), indent=1, default=str), encoding="utf8")
        print(f"\nwrote {out_dir / f'inventory_{tag}.txt'}")
    print("peak RSS MB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))


if __name__ == "__main__":
    sys.exit(main())
