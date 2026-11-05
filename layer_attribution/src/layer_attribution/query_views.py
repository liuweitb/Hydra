#!/usr/bin/env python3
"""Builders of queries.SceneView: from a saved Hydra run (raw spark_dsg, WSL only) or from a snapshot object.

usage (WSL venv, spark_dsg needed):
    python -m layer_attribution.query_views build RUN [RUN ...] --out DIR
RUN is a run name under ~/runs/uh2 or a run directory; one DIR/<run>.npz view per run is written.
The injection manifest is never read, so a view cannot leak the injected fault.
"""

import argparse
import pathlib
import sys

import numpy as np

from layer_attribution.queries import SceneView

RUNS_ROOT = pathlib.Path("~/runs/uh2").expanduser()


def _csr(rows):
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum([len(r) for r in rows], out=indptr[1:])
    idx = np.fromiter((v for r in rows for v in r), dtype=np.int32, count=int(indptr[-1]))
    return indptr, idx


def view_from_dsg(path, name=None) -> SceneView:
    """Load backend/dsg_with_mesh.json (or dsg.json) with spark_dsg and extract the query-relevant layers."""
    import spark_dsg as dsg

    path = pathlib.Path(path)
    g = dsg.DynamicSceneGraph.load(str(path))
    by_id = lambda layer: sorted(g.get_layer(layer).nodes, key=lambda n: int(n.id.value))  # noqa: E731
    objects, places, rooms = (by_id(x) for x in (dsg.DsgLayers.OBJECTS, dsg.DsgLayers.PLACES, dsg.DsgLayers.ROOMS))
    place_row = {int(n.id.value): i for i, n in enumerate(places)}
    room_row = {int(n.id.value): i for i, n in enumerate(rooms)}

    place_room = np.full(len(places), -1, dtype=np.int32)
    for i, n in enumerate(places):
        rr = sorted(room_row[p] for p in n.parents() if p in room_row)
        if rr:
            place_room[i] = rr[0]
    rows = [sorted(place_row[p] for p in n.parents() if p in place_row) for n in objects]
    indptr, idx = _csr(rows)

    def xyz(nodes):
        return np.array([np.asarray(n.attributes.position, dtype=np.float64).reshape(3) for n in nodes],
                        dtype=np.float64).reshape(-1, 3)

    bbox = np.full((len(objects), 3), np.nan, dtype=np.float32)
    for i, n in enumerate(objects):
        b = n.attributes.bounding_box
        if b.type.name != "INVALID":
            bbox[i] = np.asarray(b.dimensions, dtype=np.float32)
    try:
        names = {int(k): v for k, v in dict(g.get_labelspace(2, 0).labels_to_names).items()}
    except Exception:
        names = {}
    return SceneView(
        name=name or path.parent.parent.name,
        obj_id=np.array([int(n.id.value) for n in objects], dtype=np.int64), obj_pos=xyz(objects),
        obj_label=np.array([int(n.attributes.semantic_label) for n in objects], dtype=np.int32),
        obj_par_indptr=indptr, obj_par_idx=idx,
        place_id=np.array([int(n.id.value) for n in places], dtype=np.int64), place_pos=xyz(places),
        place_radius=np.array([n.attributes.distance for n in places], dtype=np.float32), place_room=place_room,
        room_id=np.array([int(n.id.value) for n in rooms], dtype=np.int64), room_pos=xyz(rooms),
        label_names=names, obj_bbox_dim=bbox)


def view_from_snapshot(snap, name=None) -> SceneView:
    """Adapter for research/dsg_fields/snapshot_prototype.Snapshot (duck-typed, no import).

    The snapshot keeps one parent place per object, so a second parent is lost; use view_from_dsg for exact support.
    """
    o, p, r = snap.objects, snap.places, snap.rooms
    par = np.asarray(o.parent_place, dtype=np.int32)
    indptr = np.concatenate([[0], np.cumsum(par >= 0)]).astype(np.int64)
    bbox = np.asarray(o.bbox_dim, dtype=np.float32) if "bbox_dim" in o.fields() else None
    return SceneView(
        name=name or snap.meta["run"], obj_id=np.asarray(o.id, dtype=np.int64), obj_pos=np.asarray(o.pos, dtype=np.float64),
        obj_label=np.asarray(o.label, dtype=np.int32), obj_par_indptr=indptr, obj_par_idx=par[par >= 0],
        place_id=np.asarray(p.id, dtype=np.int64), place_pos=np.asarray(p.pos, dtype=np.float64),
        place_radius=np.asarray(p.dist, dtype=np.float32), place_room=np.asarray(p.room, dtype=np.int32),
        room_id=np.asarray(r.id, dtype=np.int64), room_pos=np.asarray(r.pos, dtype=np.float64),
        label_names=snap.label_names, obj_bbox_dim=bbox)


def resolve_run_dir(run) -> pathlib.Path:
    p = pathlib.Path(str(run)).expanduser()
    return p if (p / "backend").is_dir() else RUNS_ROOT / str(run)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("runs", nargs="+")
    b.add_argument("--out", required=True, type=pathlib.Path)
    b.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        d = resolve_run_dir(run)
        out = args.out / f"{d.name}.npz"
        src = d / "backend" / "dsg_with_mesh.json"
        if not src.is_file():
            print(f"skip {d.name}: no dsg_with_mesh.json", flush=True)
            continue
        if out.exists() and out.stat().st_mtime_ns >= src.stat().st_mtime_ns and not args.rebuild:
            print(f"cached {d.name}", flush=True)
            continue
        v = view_from_dsg(src, name=d.name)
        v.save(out)
        print(f"built {d.name}: objects={v.n_obj} places={v.n_place} rooms={v.n_room}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
