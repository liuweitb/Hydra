"""spark_dsg scene graphs for the Snapshot tests (every function imports spark_dsg itself; WSL only)."""

import numpy as np

from layer_attribution import snapshot as sn


def _symbol(dsg, ch, i):
    return dsg.NodeSymbol(ch, int(i))


def synthetic_json(path):
    """Small hand-checkable graph; nodes inserted unsorted on purpose."""
    import spark_dsg as dsg

    g = dsg.DynamicSceneGraph()
    for sym, pos, lab, conn in (("O1", [4.0, 1.0, 0.4], 7, [3]), ("O0", [1.0, 2.0, 0.5], 5, [0, 1, 2])):
        a = dsg.ObjectNodeAttributes()
        a.position, a.semantic_label, a.mesh_connections = np.array(pos), lab, conn
        a.bounding_box = dsg.BoundingBox(np.array([0.5, 0.5, 0.5], dtype=np.float32), np.array(pos, dtype=np.float32))
        g.add_node(dsg.DsgLayers.OBJECTS, dsg.NodeSymbol("O", int(sym[1])), a)
    for i, (pos, d) in enumerate((([1.0, 2.0, 1.0], 0.6), ([4.0, 1.0, 1.0], 0.9), ([6.0, 1.0, 1.0], 0.7))):
        p = dsg.PlaceNodeAttributes()
        p.position, p.distance, p.num_basis_points = np.array(pos), d, 2
        g.add_node(dsg.DsgLayers.PLACES, dsg.NodeSymbol("p", i), p)
    for i, pos in enumerate(([2.0, 2.0, 1.0], [6.0, 1.0, 1.0])):
        r = dsg.RoomNodeAttributes()
        r.position = np.array(pos)
        g.add_node(dsg.DsgLayers.ROOMS, dsg.NodeSymbol("R", i), r)
    for c1, i1, c2, i2 in (("p", 0, "O", 0), ("p", 1, "O", 1), ("R", 0, "p", 0), ("R", 0, "p", 1), ("R", 1, "p", 2), ("p", 0, "p", 1),
                           ("p", 1, "p", 2), ("R", 0, "R", 1)):
        g.insert_edge(dsg.NodeSymbol(c1, i1), dsg.NodeSymbol(c2, i2))
    g.add_layer(3, dsg.PartitionId(1), "MESH_PLACES")
    q = dsg.Place2dNodeAttributes()
    q.position, q.semantic_label, q.pcl_mesh_connections = np.array([3.0, 3.0, 0.0]), 3, [0, 1]
    q.ellipse_centroid, q.ellipse_matrix_compress, q.ellipse_matrix_expand = np.zeros(3), np.eye(2), np.eye(2)
    g.add_node(dsg.LayerKey(3, 1), dsg.NodeSymbol("P", 7), q)
    g.add_layer(2, dsg.PartitionId(97), "AGENTS")
    for i in range(2):
        ag = dsg.AgentNodeAttributes()
        ag.position = np.array([float(i), 0.0, 1.0])
        ag.world_R_body = dsg.Quaternion(1.0, 0.0, 0.0, 0.0)
        g.add_node(dsg.LayerKey(2, 97), dsg.NodeSymbol("a", i), ag)
        g.insert_edge(dsg.NodeSymbol("p", i), dsg.NodeSymbol("a", i))
    m = dsg.Mesh(True, True, True, True)
    m.resize_vertices(4)
    for i, xyz in enumerate([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 1)]):
        m.set_pos(i, np.array(xyz, dtype=np.float32))
        c = dsg.Color()
        c.r, c.g, c.b, c.a = 10 * i, 20, 30, 255
        m.set_color(i, c)
        m.set_label(i, i % 3)
        m.set_timestamp(i, 1000 + i)
        m.set_first_seen_timestamp(i, 500 + i)
    m.resize_faces(2)
    m.set_face(0, [0, 1, 2])
    m.set_face(1, [1, 2, 3])
    g.mesh = m
    g.save(str(path))


def random_graph(dsg, rng):
    """Random degenerate graph: any layer may be empty, nodes may be orphans or multi-parent, odd edge types appear."""
    g = dsg.DynamicSceneGraph()
    n = {k: int(rng.integers(0, hi)) for k, hi in (("O", 5), ("p", 7), ("R", 4), ("P", 4), ("a", 4), ("B", 2))}
    ids = {k: [int(i) for i in rng.choice(40, size=v, replace=False)] for k, v in n.items()}
    mesh_kind = ["none", "empty", "full", "labels_only", "stamps_only", "bare"][int(rng.integers(0, 6))]
    nv = int(rng.integers(1, 7))
    hi = nv if mesh_kind not in ("none", "empty") else 1
    g.add_layer(3, dsg.PartitionId(1), "MESH_PLACES")
    g.add_layer(2, dsg.PartitionId(97), "AGENTS")
    for i in ids["O"]:
        a = dsg.ObjectNodeAttributes()
        a.position = rng.normal(size=3)
        a.semantic_label = int(rng.choice([5, 7, 4294967295]))
        a.mesh_connections = [int(x) for x in rng.integers(0, hi, size=int(rng.integers(0, 4)))] if mesh_kind not in ("none", "empty") else []
        if rng.random() < 0.7:
            a.bounding_box = dsg.BoundingBox(np.abs(rng.normal(size=3)).astype(np.float32), rng.normal(size=3).astype(np.float32))
        g.add_node(dsg.DsgLayers.OBJECTS, _symbol(dsg, "O", i), a)
    for i in ids["p"]:
        a = dsg.PlaceNodeAttributes()
        a.position, a.distance, a.num_basis_points = rng.normal(size=3), float(rng.random()), int(rng.integers(0, 6))
        g.add_node(dsg.DsgLayers.PLACES, _symbol(dsg, "p", i), a)
    for i in ids["R"]:
        a = dsg.RoomNodeAttributes()
        a.position = rng.normal(size=3)
        g.add_node(dsg.DsgLayers.ROOMS, _symbol(dsg, "R", i), a)
    for i in ids["P"]:
        a = dsg.Place2dNodeAttributes()
        a.position, a.semantic_label = rng.normal(size=3), int(rng.choice([3, 15, 4294967295]))
        a.ellipse_centroid, a.ellipse_matrix_compress, a.ellipse_matrix_expand = np.zeros(3), np.eye(2), np.eye(2)
        g.add_node(dsg.LayerKey(3, 1), _symbol(dsg, "P", i), a)
    for i in ids["a"]:
        a = dsg.AgentNodeAttributes()
        a.position = rng.normal(size=3)
        a.world_R_body = dsg.Quaternion(1.0, 0.0, 0.0, 0.0)  # default is uninitialised; NaN is written as null and cannot be reloaded
        g.add_node(dsg.LayerKey(2, 97), _symbol(dsg, "a", i), a)
    for i in ids["B"]:
        g.add_node(dsg.DsgLayers.BUILDINGS, _symbol(dsg, "B", i), dsg.SemanticNodeAttributes())

    def link(c1, i1, c2, i2):
        try:
            g.insert_edge(_symbol(dsg, c1, i1), _symbol(dsg, c2, i2))
        except Exception:
            pass  # generator-side: duplicate or rejected edge

    for i in ids["O"]:
        if ids["p"] and rng.random() < 0.8:
            for j in rng.choice(ids["p"], size=min(len(ids["p"]), int(rng.integers(1, 4))), replace=False):
                link("p", j, "O", i)
        if ids["R"] and rng.random() < 0.2:
            link("R", rng.choice(ids["R"]), "O", i)
    for i in ids["p"]:
        if ids["R"] and rng.random() < 0.85:
            for j in rng.choice(ids["R"], size=min(len(ids["R"]), 1 + int(rng.random() < 0.15)), replace=False):
                link("R", j, "p", i)
        for j in rng.choice(ids["p"], size=min(len(ids["p"]), 2), replace=False):
            if j != i:
                link("p", i, "p", j)
    for i in ids["a"]:
        if ids["p"] and rng.random() < 0.7:
            link("p", rng.choice(ids["p"]), "a", i)
    if len(ids["R"]) > 1:
        link("R", ids["R"][0], "R", ids["R"][1])
    if len(ids["P"]) > 1:
        link("P", ids["P"][0], "P", ids["P"][1])
    if ids["B"] and ids["R"]:
        link("B", ids["B"][0], "R", ids["R"][0])
    if mesh_kind != "none":
        # Mesh(has_colors, has_timestamps, has_labels, has_first_seen)
        flags = {"empty": (True,) * 4, "full": (True,) * 4, "labels_only": (False, False, True, False),
                 "stamps_only": (False, True, False, True), "bare": (False,) * 4}[mesh_kind]
        m = dsg.Mesh(*flags)
        if mesh_kind != "empty":
            m.resize_vertices(nv)
            for i in range(nv):
                m.set_pos(i, rng.normal(size=3).astype(np.float32))
            if nv >= 3:
                m.resize_faces(1)
                m.set_face(0, [0, 1, 2])
            if mesh_kind in ("full", "labels_only"):
                for i in range(nv):
                    m.set_label(i, int(rng.choice([0, 3, 19, 4294967295])))
            if mesh_kind in ("full", "stamps_only"):
                for i in range(nv):
                    m.set_timestamp(i, int(i))
                    m.set_first_seen_timestamp(i, int(i))
        g.mesh = m
    return g, len(ids["a"]), mesh_kind


def nonfinite_graph_json(path, what):
    import spark_dsg as dsg

    g = dsg.DynamicSceneGraph()
    if what == "position":
        a = dsg.ObjectNodeAttributes()
        a.position = np.array([np.nan, 0.0, 0.0])
        g.add_node(dsg.DsgLayers.OBJECTS, dsg.NodeSymbol("O", 0), a)
    else:
        a = dsg.AgentNodeAttributes()
        a.position = np.zeros(3)
        a.world_R_body = dsg.Quaternion(np.nan, 0.0, 0.0, 0.0)
        g.add_layer(2, dsg.PartitionId(97), "AGENTS")
        g.add_node(dsg.LayerKey(2, 97), dsg.NodeSymbol("a", 0), a)
    g.save(str(path))


def write_run(run, g, csv_text=None, stats=None):
    """Run directory with backend/dsg_with_mesh.json, optional trajectory.csv and {layer: csv text} statistics files."""
    (run / "backend").mkdir(parents=True, exist_ok=True)
    g.save(str(run / "backend" / "dsg_with_mesh.json"))
    if csv_text is not None:
        (run / "backend" / "trajectory.csv").write_text(csv_text)
    for layer, text in (stats or {}).items():
        (run / "backend" / f"layer_{layer}_statistics.csv").write_text(text)


def csv_variants(n_agents, rng):
    hdr = "timestamp[ns],x,y,z,qw,qx,qy,qz\n"
    rows = lambda k: "".join(f"{i * 10**9},{i},0,0,1,0,0,0\n" for i in range(k))  # noqa: E731
    return [None, "", hdr, hdr + rows(n_agents), hdr + rows(n_agents + 2), "t,x,y\n1,2,3\n", "garbage\nnot,a,number\n"][int(rng.integers(0, 7))]


def verify_against_dsg(run_dir, snap, agent_stamps=False):
    """Every stored value against the spark_dsg API (all nodes, edges, mesh vertices); returns the list of mismatches.

    agent_stamps: compare trajectory.csv times with the agent node timestamps (true for Hydra runs, not for hand-made graphs).
    """
    import spark_dsg as dsg

    g = dsg.DynamicSceneGraph.load(str(run_dir / "backend" / "dsg_with_mesh.json"))
    snap.materialize()
    nodes = {c: [] for c in sn.CAT}
    for key in g.layer_keys:
        for n in g.get_layer(key.layer, key.partition).nodes:
            nodes.setdefault(n.id.category, []).append(n)
    for v in nodes.values():
        v.sort(key=lambda n: int(n.id.value))
    O, P, M, R, B, A = (nodes[c] for c in "OpPRBa")
    o, p, r, mp, b, tr, mesh = snap.objects, snap.places, snap.rooms, snap.mplaces, snap.building, snap.traj, snap.mesh
    bad = []

    def expect(cond, msg):
        if not cond:
            bad.append(msg)

    pid = {int(n.id.value): i for i, n in enumerate(P)}
    rid = {int(n.id.value): i for i, n in enumerate(R)}

    def room_row(parents):
        rp = sorted(int(x) for x in parents if int(x) in rid)
        return len(rp), (rid[rp[0]] if rp else -1)

    for tab, ns, cats in ((o, O, "objects"), (p, P, "places"), (r, R, "rooms"), (mp, M, "mplaces"), (b, B, "building")):
        expect([int(n.id.value) for n in ns] == tab.id.tolist(), f"{cats} ids")
        expect(np.array_equal(np.array([n.attributes.position for n in ns]).reshape(-1, 3), tab.pos), f"{cats} pos")
    for i, n in enumerate(O):
        a = n.attributes
        bb = a.bounding_box
        invalid = bb.type.name == "INVALID"
        expect(sn._label(a.semantic_label) == o.label[i], f"object {i} label")
        expect(invalid and np.isnan(o.bbox_dim[i]).all() and np.isnan(o.bbox_rot[i]).all() or
               np.allclose(bb.dimensions, o.bbox_dim[i]) and np.allclose(bb.world_P_center, o.bbox_center[i]) and np.allclose(bb.world_R_center, o.bbox_rot[i]),
               f"object {i} bbox")
        expect(list(a.mesh_connections) == o.csr("mesh_conn_indptr", "mesh_conn_idx", i).tolist(), f"object {i} mesh_connections")
        par = sorted(int(x) for x in n.parents())
        pp = [x for x in par if x in pid]
        expect([int(p.id[j]) for j in o.csr("parent_indptr", "parent_idx", i)] == pp and o.n_parents[i] == len(par), f"object {i} parents")
        expect(o.parent_place[i] == (pid[pp[0]] if pp else -1), f"object {i} parent_place")
        expect(o.room[i] == (room_row(g.get_node(pp[0]).parents())[1] if pp else -1), f"object {i} room")
        expect(bool(a.is_active) == o.is_active[i] and a.last_update_time_ns == o.last_update_ns[i], f"object {i} flags")
    for i, n in enumerate(P):
        a = n.attributes
        nrp, room = room_row(n.parents())
        ch = [int(x) for x in n.children()]
        basis = np.array([tuple(v.voxel_pos) for v in a.voxblox_mesh_connections], dtype=np.float32).reshape(-1, 3)
        expect(np.float32(a.distance) == p.dist[i] and a.num_basis_points == p.n_basis[i], f"place {i} distance / basis count")
        expect(np.array_equal(basis, p.csr("basis_indptr", "basis_pos", i)), f"place {i} basis points")
        expect(p.n_room_parents[i] == nrp and p.room[i] == room, f"place {i} room")
        expect(p.deg[i] == len(n.siblings()), f"place {i} degree")
        expect(p.n_obj[i] == sum(chr(x >> 56) == "O" for x in ch) and p.n_agent[i] == sum(chr(x >> 56) == "a" for x in ch), f"place {i} children")
        expect(bool(a.is_active) == p.is_active[i] and a.last_update_time_ns == p.last_update_ns[i], f"place {i} flags")
    for i, n in enumerate(R):
        expect(sorted(int(x) for x in n.children() if int(x) in pid) == sorted(int(p.id[j]) for j in r.csr("child_indptr", "child_idx", i)), f"room {i} children")
        expect(r.deg[i] == len(n.siblings()), f"room {i} degree")
        expect(bool(n.has_parent()) == bool(r.has_parent[i]), f"room {i} has_parent")
    for i, n in enumerate(M):
        a = n.attributes
        expect(sn._label(a.semantic_label) == mp.label[i] and list(a.pcl_mesh_connections) == mp.csr("mesh_conn_indptr", "mesh_conn_idx", i).tolist(), f"mplace {i} label / mesh connections")
        expect(np.array_equal(np.array(a.boundary, dtype=np.float32).reshape(-1, 3), mp.csr("boundary_indptr", "boundary_xyz", i)), f"mplace {i} boundary")
        expect(np.allclose(a.ellipse_centroid, mp.ellipse_centroid[i]) and np.allclose(np.asarray(a.ellipse_matrix_expand), mp.ellipse_expand[i]), f"mplace {i} ellipse")
        expect(mp.deg[i] == len(n.siblings()), f"mplace {i} degree")
    for i, n in enumerate(B):
        expect(sn._label(n.attributes.semantic_label) == b.label[i], f"building {i} label")
    for tab, ns, ch in ((snap.place_edges, P, "p"), (snap.mplace_edges, M, "P"), (snap.room_edges, R, "R")):
        ids = np.array([int(n.id.value) for n in ns], dtype=np.int64)
        want = sorted((int(e.source), int(e.target), float(np.float32(e.info.weight))) for e in g.edges
                      if chr(int(e.source) >> 56) == ch and chr(int(e.target) >> 56) == ch)
        have = sorted((int(ids[i]), int(ids[j]), float(w)) for (i, j), w in zip(tab.ij, tab.w))
        expect(want == have, f"{ch}-{ch} edges")
    if snap.meta["traj_source"] == "trajectory.csv" and snap.meta["traj_matches_agents"]:
        if agent_stamps:
            ts = np.array([int(round(n.attributes.timestamp.total_seconds() * 1e9)) for n in A], dtype=np.int64)
            expect(len(ts) == len(tr.t_ns) and (len(ts) == 0 or np.abs(ts - tr.t_ns).max() <= 2000), "trajectory timestamps")
        expect(len(A) == 0 or np.abs(np.array([n.attributes.position for n in A]) - tr.pos).max() < 1e-3, "trajectory positions")
        expect([(pid[int(x)] if int(x) in pid else -1) for n in A for x in sorted(n.parents())[:1]] == [int(v) for v in tr.place[[bool(n.parents()) for n in A]]], "trajectory places")
    m = g.mesh
    if m is not None and m.num_vertices():
        V, nv, fl = m.get_vertices(), m.num_vertices(), snap.meta["mesh_flags"]
        expect(np.array_equal(mesh.xyz, V[:3].T.astype(np.float32)) and np.array_equal(mesh.faces, m.get_faces().T), "mesh vertices / faces")
        want = np.minimum(np.array(m.get_labels(), dtype=np.int64), sn.MESH_NO_LABEL) if fl["labels"] else np.full(nv, sn.MESH_NO_LABEL)
        expect(np.array_equal(mesh.label, want), "mesh labels")
        expect(not fl["colors"] or np.abs(mesh.rgb.astype(float) - V[3:6].T * 255).max() < 0.51, "mesh colours")
        expect(np.array_equal(mesh.stamp_ns, [m.timestamp(i) for i in range(nv)] if fl["stamps"] else np.zeros(nv)), "mesh stamps")
        expect(np.array_equal(mesh.first_seen_ns, [m.first_seen_timestamp(i) for i in range(nv)] if fl["first_seen"] else np.zeros(nv)), "mesh first seen")
    expect(snap.meta["graph_counts"] == {"nodes": g.num_nodes(), "edges": g.num_edges()}, "graph_counts")
    return bad


def opened_paths_during(fn):
    """Paths opened for reading while fn runs (audit hook; the hook stays installed but only records while armed)."""
    import sys

    seen = []

    def hook(event, args):
        if event == "open" and armed[0]:
            seen.append(str(args[0]))

    armed = [False]
    sys.addaudithook(hook)
    armed[0] = True
    try:
        fn()
    finally:
        armed[0] = False
    return seen
