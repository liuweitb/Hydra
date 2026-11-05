#!/usr/bin/env python3
"""Why do two repeated Hydra runs hash differently, and does anything semantic differ?

Levels: (1) artefact hashes, (2) first differing byte, (3) byte equality after masking the
uninitialised-memory keys, (4) parsed-JSON structural diff (paths, counts, node/edge order),
(5) semantic diff through spark_dsg (per layer and field, edges, mesh), (6) pgmo status csv.

usage (WSL, via research/dsg_fields/wsl_py.sh):
    determinism_diff.py RUN_A RUN_B [--out-dir DIR] [--skip-frontend]
RUN_x = run directory (contains backend/ and frontend/).
"""

import argparse
import gc
import hashlib
import json
import pathlib
import re
import resource
import sys
import time
from collections import Counter, defaultdict

import numpy as np

# fields serialised from uninitialised memory in NearestVertexInfo (spark_dsg node_attributes.h)
MASK_BLOCK = re.compile(rb'"block":\[-?\d+,-?\d+,-?\d+\]')
MASK_VERTEX = re.compile(rb'"vertex":\d+')
TIMING_COLS = {"run_time", "optimize_time", "mesh_update_time"}


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


def rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def sha(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def mask(raw):
    return MASK_VERTEX.sub(b'"vertex":V', MASK_BLOCK.sub(b'"block":[X]', raw))


def first_diff(a, b):
    n = min(len(a), len(b))
    i, step = 0, 1 << 16
    while i < n and a[i:i + step] == b[i:i + step]:
        i += step
    while i < n and a[i] == b[i]:
        i += 1
    return i


# ---------------------------------------------------------------- level 4: parsed JSON diff
def pattern(path):
    return re.sub(r"\[\d+\]", "[*]", path)


def walk(a, b, path, acc, limit=3):
    """Collect differing leaves by path pattern; equal subtrees are skipped with one C-level compare."""
    if a == b and type(a) is type(b):
        return
    if type(a) is not type(b):
        acc[(pattern(path), "type")].append((path, repr(a)[:60], repr(b)[:60]))
        return
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                acc[(pattern(f"{path}.{k}"), "missing key")].append((f"{path}.{k}", k in a, k in b))
            else:
                walk(a[k], b[k], f"{path}.{k}", acc, limit)
    elif isinstance(a, list):
        if len(a) != len(b):
            acc[(pattern(path), "length")].append((path, len(a), len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", acc, limit)
    else:
        acc[(pattern(path), "value")].append((path, a, b))


def json_structural_diff(R, pa, pb):
    t0 = time.time()
    with open(pa) as f:
        A = json.load(f)
    with open(pb) as f:
        B = json.load(f)
    R(f"parsed both json files in {time.time() - t0:.1f}s, peak RSS {rss_mb()} MB")
    out = {}
    R("top-level keys equal:", sorted(A) == sorted(B), "| per-key equality:", {k: A[k] == B[k] for k in sorted(A)})
    ida, idb = [n["id"] for n in A["nodes"]], [n["id"] for n in B["nodes"]]
    ea, eb = [(e["source"], e["target"]) for e in A["edges"]], [(e["source"], e["target"]) for e in B["edges"]]
    R("node id sequence identical (same order):", ida == idb, "| same id set:", set(ida) == set(idb), "| n nodes", len(ida), len(idb))
    R("edge (source,target) sequence identical (same order):", ea == eb, "| same pair set:", set(ea) == set(eb), "| n edges", len(ea), len(eb))
    acc = defaultdict(list)
    walk(A, B, "$", acc)
    R("")
    R("differing leaves by JSON path pattern (everything not listed is exactly equal, including float text/ordering/timestamps):")
    if not acc:
        R("  none")
    for (pat, kind), items in sorted(acc.items(), key=lambda kv: -len(kv[1])):
        R(f"  {len(items):6d} x {kind:11s} {pat}")
        for it in items[:2]:
            R(f"{'':16s}e.g. {it}")
        out[f"{pat} [{kind}]"] = len(items)
    del A, B
    gc.collect()
    return out


# ---------------------------------------------------------------- level 5: semantic diff via spark_dsg
def canon(v):
    import datetime
    import spark_dsg as dsg
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
    if isinstance(v, dsg.NearestVertexInfo):
        return ("nvi", tuple(v.voxel_pos), v.label)  # block / vertex excluded: uninitialised memory
    if hasattr(v, "get") and type(v).__name__ == "Metadata":
        return ("md", canon(dict(v.get())))
    return ("repr", repr(v))


def fingerprint(path):
    import spark_dsg as dsg
    g = dsg.DynamicSceneGraph.load(str(path))
    fp = {"nodes": {}, "edges": {}, "mesh": {}, "raw_nvi": []}
    for key in g.layer_keys:
        for n in g.get_layer(key.layer, key.partition).nodes:
            a = n.attributes
            row = {"_layer": str(key), "_cls": type(a).__name__}
            for name in dir(type(a)):
                if name.startswith("_") or name == "NO_SEMANTIC_LABEL":
                    continue
                v = getattr(a, name)
                if callable(v):
                    continue
                row[name] = canon(v)
            row["_parents"] = canon(sorted(n.parents()))
            row["_children"] = canon(sorted(n.children()))
            row["_siblings"] = canon(sorted(n.siblings()))
            fp["nodes"][int(n.id.value)] = row
            if type(a).__name__ == "PlaceNodeAttributes":
                for v in a.voxblox_mesh_connections:
                    fp["raw_nvi"].append((int(n.id.value), tuple(int(x) for x in v.block), tuple(float(x) for x in v.voxel_pos)))
    for e in g.edges:
        k = (min(int(e.source), int(e.target)), max(int(e.source), int(e.target)))
        fp["edges"][k] = (e.info.weight, e.info.weighted, canon(dict(e.info.metadata.get())))
    m = g.mesh
    fp["mesh"] = {"vertices": m.get_vertices(), "faces": m.get_faces(), "labels": np.array(m.get_labels()),
                  "timestamps": np.array([m.timestamp(i) for i in range(m.num_vertices())], dtype=np.int64),
                  "first_seen": np.array([m.first_seen_timestamp(i) for i in range(m.num_vertices())], dtype=np.int64)}
    fp["meta"] = canon(dict(g.metadata.get()))
    fp["layer_keys"] = [str(k) for k in g.layer_keys]
    return fp


def semantic_diff(R, pa, pb):
    fa = fingerprint(pa)
    R(f"fingerprinted A, peak RSS {rss_mb()} MB")
    fb = fingerprint(pb)
    R(f"fingerprinted B, peak RSS {rss_mb()} MB")
    out = {}
    R("layer keys equal:", fa["layer_keys"] == fb["layer_keys"], "| graph metadata equal:", fa["meta"] == fb["meta"])
    ia, ib = set(fa["nodes"]), set(fb["nodes"])
    R("node id sets equal:", ia == ib, f"(A {len(ia)}, B {len(ib)}, only-A {len(ia - ib)}, only-B {len(ib - ia)})")
    diff_fields = Counter()
    n_nodes_diff = Counter()
    per_layer_nodes = Counter()
    for nid in sorted(ia & ib):
        ra, rb = fa["nodes"][nid], fb["nodes"][nid]
        per_layer_nodes[ra["_layer"]] += 1
        bad = [k for k in ra if ra[k] != rb.get(k)]
        for k in bad:
            diff_fields[(ra["_layer"], k)] += 1
        if bad:
            n_nodes_diff[ra["_layer"]] += 1
    R("nodes compared per layer:", dict(per_layer_nodes))
    R("nodes with any differing field (voxblox block/vertex excluded):", dict(n_nodes_diff) or "none")
    R("differing (layer, field) counts:", dict(diff_fields) or "none")
    out["node_field_diffs"] = {f"{a}:{b}": c for (a, b), c in diff_fields.items()}
    pa_, pb_ = set(fa["edges"]), set(fb["edges"])
    wdiff = sum(1 for k in pa_ & pb_ if fa["edges"][k] != fb["edges"][k])
    R("edge sets equal:", pa_ == pb_, f"(A {len(pa_)}, B {len(pb_)}) | edges with differing weight/weighted/metadata: {wdiff}")
    out["edges_equal"] = pa_ == pb_ and wdiff == 0
    for k in fa["mesh"]:
        x, y = fa["mesh"][k], fb["mesh"][k]
        same = x.shape == y.shape and np.array_equal(x, y)
        mad = float(np.abs(x.astype(float) - y.astype(float)).max()) if x.shape == y.shape else None
        R(f"mesh {k:12s} shape {x.shape} equal: {same} max|diff| {mad}")
        out[f"mesh_{k}_equal"] = bool(same)
    # the uninitialised records
    ra = sorted(fa["raw_nvi"])
    rb = sorted(fb["raw_nvi"])
    n = len(ra)
    R("")
    R(f"PlaceNodeAttributes.voxblox_mesh_connections records: A {len(ra)}, B {len(rb)}; same (place, voxel_pos) multiset:",
      sorted((p, v) for p, _, v in ra) == sorted((p, v) for p, _, v in rb))
    blk_diff = sum(1 for x, y in zip(ra, rb) if x[1] != y[1])
    R(f"records whose 'block' triple differs between runs: {blk_diff}/{n}")
    out["nvi_records"] = n
    out["nvi_block_differs"] = blk_diff
    out["fingerprint_peak_rss_mb"] = rss_mb()
    return out, fa, fb


# ---------------------------------------------------------------- pointer evidence
def pointer_evidence(R, raw, label):
    blocks = re.findall(rb'"block":\[(-?\d+),(-?\d+),(-?\d+)\],"label":(?:null|\d+),"vertex":(\d+)', raw)
    if not blocks:
        blocks = re.findall(rb'"block":\[(-?\d+),(-?\d+),(-?\d+)\][^}]*?"vertex":(\d+)', raw)
    b0 = np.array([int(x[0]) for x in blocks], dtype=np.int64)
    b1 = np.array([int(x[1]) for x in blocks], dtype=np.int64)
    b2 = np.array([int(x[2]) for x in blocks], dtype=np.int64)
    vx = np.array([int(x[3]) for x in blocks], dtype=np.uint64)
    hi = (vx >> np.uint64(32)).astype(np.int64)
    R(f"run {label}: {len(vx)} records | block[1] == vertex>>32 in {np.mean(b1 == hi):.1%} | vertex % 8 == 0 in {np.mean(vx % 8 == 0):.1%} | "
      f"vertex range 0x{int(vx.min()):x}..0x{int(vx.max()):x} | distinct vertex {len(np.unique(vx))} | distinct block[0..1] pairs {len(set(zip(b0.tolist(), b1.tolist())))} | "
      f"block[2] range {int(b2.min())}..{int(b2.max())} ({len(np.unique(b2))} distinct)")
    return b2


def show_source(R):
    import os
    base = os.path.expanduser("~/hydra_ws/src")
    for rel, a, b in (("spark_dsg/include/spark_dsg/node_attributes.h", 77, 86), ("hydra/src/places/graph_extractor.cpp", 210, 216)):
        p = os.path.join(base, rel)
        if os.path.exists(p):
            lines = open(p, encoding="utf8", errors="replace").read().splitlines()
            R(f"--- {rel}:{a}-{b}")
            for i in range(a - 1, b):
                R(f"{i + 1:5d}  {lines[i]}")


# ---------------------------------------------------------------- level 6: pgmo status csv
def pgmo_status(R, pa, pb):
    """Rows are backend spin iterations; the last two fields are glued ('nan0'), so state = first 7 columns."""
    def load(p):
        lines = pathlib.Path(p).read_text().splitlines()
        return lines[0].split(","), [ln.split(",")[:7] for ln in lines[1:] if ln.count(",") >= 7]

    ha, ra = load(pa)
    hb, rb = load(pb)
    R("columns:", ha, "(state = first 7; run_time/optimize_time/mesh_update_time are wall-clock)")
    R("rows (backend spin iterations) A", len(ra), "B", len(rb))

    def changes(rows):
        seq = []
        for r in rows:
            if not seq or seq[-1] != r:
                seq.append(r)
        return seq

    sa, sb = changes(ra), changes(rb)
    R(f"distinct consecutive states: A {len(sa)}, B {len(sb)}; sequences identical: {sa == sb}; "
      f"idle (repeated-state) iterations: A {len(ra) - len(sa)}, B {len(rb) - len(sb)}")
    R("final state A:", ra[-1], "B:", rb[-1])
    return {"rows_a": len(ra), "rows_b": len(rb), "state_sequence_equal": sa == sb, "states_a": len(sa), "states_b": len(sb)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_a", type=pathlib.Path)
    ap.add_argument("run_b", type=pathlib.Path)
    ap.add_argument("--out-dir", type=pathlib.Path, default=None)
    ap.add_argument("--skip-frontend", action="store_true")
    ap.add_argument("--skip-semantic", action="store_true", help="skip the spark_dsg fingerprint level")
    args = ap.parse_args()
    ra, rb = args.run_a.expanduser(), args.run_b.expanduser()
    R = Report()
    res = {"run_a": str(ra), "run_b": str(rb)}

    R.section("LEVEL 1  artefact hashes")
    files = ["backend/dsg_with_mesh.json", "backend/dsg.json", "backend/mesh.ply", "backend/trajectory.csv",
             "backend/layer_2_statistics.csv", "backend/layer_3_statistics.csv", "backend/layer_4_statistics.csv",
             "backend/layer_5_statistics.csv", "backend/loop_closures.csv", "backend/deformation_graph.dgrf",
             "backend/pgmo/dsg_pgmo_status.csv", "hydra_config.yaml", "timing_stats.csv"]
    if not args.skip_frontend:
        files += ["frontend/dsg_with_mesh.json", "frontend/dsg.json", "frontend/mesh.ply", "frontend/layer_2_statistics.csv",
                  "frontend/layer_3_statistics.csv", "frontend/layer_4_statistics.csv", "frontend/layer_5_statistics.csv"]
    hashes = {}
    for fn in files:
        pa, pb = ra / fn, rb / fn
        if not (pa.exists() and pb.exists()):
            R(f"{fn:36s} missing in {'A' if not pa.exists() else 'B'}")
            continue
        ha, hb = sha(pa), sha(pb)
        hashes[fn] = ha == hb
        R(f"{fn:36s} {pa.stat().st_size:>10d} {pb.stat().st_size:>10d} bytes  sha256[:16] {ha} {hb}  {'IDENTICAL' if ha == hb else 'DIFFERENT'}")
    res["identical_files"] = [k for k, v in hashes.items() if v]
    res["different_files"] = [k for k, v in hashes.items() if not v]

    R.section("LEVEL 2/3  raw bytes: first difference, then equality after masking the 'block' and 'vertex' keys")
    for fn in [f for f in ("backend/dsg_with_mesh.json", "backend/dsg.json", "frontend/dsg_with_mesh.json", "frontend/dsg.json")
               if f in hashes and not hashes[f]]:
        A, B = (ra / fn).read_bytes(), (rb / fn).read_bytes()
        i = first_diff(A, B)
        R(f"{fn}: sizes {len(A)} vs {len(B)}; first differing byte at offset {i}")
        R("   A ...", A[max(0, i - 90):i + 70])
        R("   B ...", B[max(0, i - 90):i + 70])
        nb = (len(MASK_BLOCK.findall(A)), len(MASK_VERTEX.findall(A)))
        A2, B2 = mask(A), mask(B)
        R(f"   masked keys found in A: block x{nb[0]}, vertex x{nb[1]}; after masking sizes {len(A2)} vs {len(B2)}; byte-identical: {A2 == B2}")
        res[f"{fn}_identical_after_masking"] = A2 == B2
        if fn == "backend/dsg_with_mesh.json":
            b2a = pointer_evidence(R, A, "A")
            b2b = pointer_evidence(R, B, "B")
            R("   block[2] sequence identical between runs:", bool(len(b2a) == len(b2b) and np.array_equal(b2a, b2b)))
        del A, B, A2, B2
        gc.collect()
    R("")
    R("source of the two keys (read from the WSL source tree):")
    show_source(R)

    R.section("LEVEL 4  parsed JSON structural diff, backend/dsg_with_mesh.json")
    res["json_diff_paths"] = json_structural_diff(R, ra / "backend/dsg_with_mesh.json", rb / "backend/dsg_with_mesh.json")

    if not args.skip_semantic:
        R.section("LEVEL 5  semantic diff through spark_dsg (voxblox block/vertex excluded from the comparison)")
        sem, _, _ = semantic_diff(R, ra / "backend/dsg_with_mesh.json", rb / "backend/dsg_with_mesh.json")
        res["semantic"] = sem

    R.section("LEVEL 6  backend/pgmo/dsg_pgmo_status.csv (one row per backend spin; wall-clock columns excluded)")
    if (ra / "backend/pgmo/dsg_pgmo_status.csv").exists() and (rb / "backend/pgmo/dsg_pgmo_status.csv").exists():
        res["pgmo"] = pgmo_status(R, ra / "backend/pgmo/dsg_pgmo_status.csv", rb / "backend/pgmo/dsg_pgmo_status.csv")

    R.section("VERDICT")
    R("peak RSS MB", rss_mb())
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{ra.name}_vs_{rb.name}"
        (args.out_dir / f"determinism_{tag}.txt").write_text("\n".join(R.lines) + "\n", encoding="utf8")
        (args.out_dir / f"determinism_{tag}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf8")
        print("wrote", args.out_dir / f"determinism_{tag}.txt")


if __name__ == "__main__":
    sys.exit(main())
