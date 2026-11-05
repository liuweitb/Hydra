"""Snapshot built from real spark_dsg graphs; skipped where spark_dsg is missing (Windows python), runs in the WSL venv."""

import os

import numpy as np
import pytest

dsg = pytest.importorskip("spark_dsg")

import graph_factory as gf  # noqa: E402

from layer_attribution import snapshot as sn  # noqa: E402

HDR = "nodes_active,nodes_removed,nodes_merged,nodes_w_parents,nodes_w_children,edges\n"
TRAJ = "timestamp[ns],x,y,z,qw,qx,qy,qz\n499300019999,0,0,1,1,0,0,0\n501100019999,1,0,1,1,0,0,0\n"


@pytest.fixture()
def synthetic(tmp_path):
    run = tmp_path / "run"
    (run / "backend").mkdir(parents=True)
    gf.synthetic_json(run / "backend" / "dsg_with_mesh.json")
    (run / "backend" / "trajectory.csv").write_text(TRAJ)
    (run / "backend" / "layer_2_statistics.csv").write_text(HDR + "0,0,0,0,0,0\n3,1,2,3,0,0\n")
    (run / "backend" / "layer_3_statistics.csv").write_text(HDR + "7,4,5,6,1,9\n")
    return run, tmp_path / "cache"


def test_synthetic_graph_values(synthetic):
    run, cache = synthetic
    s = sn.Snapshot.open(run, cache_root=cache)
    assert not s.from_cache and s.check() == []
    o, p, r, mp, mesh, tr = s.objects, s.places, s.rooms, s.mplaces, s.mesh, s.traj
    assert list(o.id) == sorted(o.id) and list(o.label) == [5, 7] and sn.symbol(o.id[0]) == "O0" and sn.symbol(p.id[2]) == "p2"
    assert list(o.parent_place) == [0, 1] and list(o.room) == [0, 0] and list(o.n_parents) == [1, 1]
    assert list(o.csr("parent_indptr", "parent_idx", 1)) == [1] and list(o.n_mesh_conn) == [3, 1]
    assert np.allclose(o.bbox_rot, np.eye(3)) and o.bbox_rot.shape == (2, 3, 3) and np.allclose(o.bbox_dim, 0.5)
    assert len(p) == 3 and np.allclose(p.dist, [0.6, 0.9, 0.7]) and list(p.room) == [0, 0, 1] and list(p.deg) == [1, 2, 1]
    assert list(p.n_obj) == [1, 1, 0] and list(p.n_agent) == [1, 1, 0] and list(p.n_room_parents) == [1, 1, 1]
    assert list(r.n_children) == [2, 1] and list(r.deg) == [1, 1] and list(r.has_parent) == [False, False]
    assert np.allclose(r.ext_min[0], [1, 1, 1]) and np.allclose(r.ext_max[0], [4, 2, 1])
    assert len(mp) == 1 and mp.label[0] == 3 and list(mp.csr("mesh_conn_indptr", "mesh_conn_idx", 0)) == [0, 1]
    assert mesh.xyz.shape == (4, 3) and mesh.faces.shape == (2, 3) and list(mesh.label) == [0, 1, 2, 0]
    assert list(mesh.stamp_ns) == [1000, 1001, 1002, 1003] and list(mesh.first_seen_ns) == [500, 501, 502, 503]
    assert list(tr.t_ns) == [499300019999, 501100019999] and list(tr.place) == [0, 1] and s.meta["traj_matches_agents"] is True
    assert s.meta["edge_counts"] == {"O-p": 2, "R-p": 3, "R-R": 1, "p-p": 2, "a-p": 2} and s.meta["edges_unrepresented"] == 0
    g = dsg.DynamicSceneGraph.load(str(run / "backend" / "dsg_with_mesh.json"))
    assert s.meta["graph_counts"] == {"nodes": g.num_nodes(), "edges": g.num_edges()}
    assert list(s.stats.layer_2_last) == [3, 1, 2, 3, 0, 0] and list(s.stats.layer_3_last) == [7, 4, 5, 6, 1, 9]
    assert np.isnan(s.stats.layer_4_last).all() and s.stats.layer_4_last.shape == (6,)  # file absent
    assert gf.verify_against_dsg(run, s) == []
    s.close()


def test_cache_is_keyed_by_json_and_side_files(synthetic):
    run, cache = synthetic
    first = sn.Snapshot.open(run, cache_root=cache)
    first.materialize()
    again = sn.Snapshot.open(run, cache_root=cache)
    assert again.from_cache and again.meta["build_seconds"] == first.meta["build_seconds"]
    for tn in sn.TABLES:
        for f in first._tables[tn].fields():
            assert np.array_equal(getattr(first._tables[tn], f), getattr(again._tables[tn], f), equal_nan=True), (tn, f)
    again.close()
    side = run / "backend" / "layer_2_statistics.csv"
    side.write_text(HDR + "0,0,0,0,0,0\n9,9,9,9,9,9\n")  # a side file changed: stale
    changed = sn.Snapshot.open(run, cache_root=cache)
    assert not changed.from_cache and list(changed.stats.layer_2_last) == [9] * 6
    changed.close()
    src = run / "backend" / "dsg_with_mesh.json"
    os.utime(src, ns=(src.stat().st_atime_ns, src.stat().st_mtime_ns + 10**9))
    assert not sn.Snapshot.open(run, cache_root=cache).from_cache
    assert sn.Snapshot.open(run, cache_root=cache).from_cache


def test_statistics_files_that_are_unusable_give_nan_rows(synthetic):
    run, cache = synthetic
    b = run / "backend"
    (b / "layer_2_statistics.csv").write_text(HDR)  # header only
    (b / "layer_3_statistics.csv").write_text("a,b\n1,2\n")  # too few columns
    (b / "layer_4_statistics.csv").write_text("nodes_active,x,y,z,u,v\n5,1,2,3,4,5\n")  # unexpected header: values kept, warned
    with sn.Snapshot.open(run, cache_root=cache) as s:
        assert np.isnan(s.stats.layer_2_last).all() and np.isnan(s.stats.layer_3_last).all()
        assert list(s.stats.layer_4_last) == [5, 1, 2, 3, 4, 5] and any("differ" in w for w in s.meta["warnings"])
        assert s.check() == []


def test_random_degenerate_graphs_build_pass_invariants_and_equal_the_api(tmp_path):
    rng = np.random.default_rng(0)
    cache, kinds = tmp_path / "cache", {}
    for k in range(41):
        g, na, mk = (dsg.DynamicSceneGraph(), 0, "none") if k == 0 else gf.random_graph(dsg, rng)
        run = tmp_path / f"r{k}"
        gf.write_run(run, g, gf.csv_variants(na, rng))
        s = sn.Snapshot.open(run, cache_root=cache)
        assert s.check() == [], (k, mk, s.check())
        assert gf.verify_against_dsg(run, s) == [], (k, mk)
        s2 = sn.Snapshot.open(run, cache_root=cache)
        assert s2.from_cache
        for tn in sn.TABLES:
            for f in s._tables[tn].fields():
                assert np.array_equal(getattr(s._tables[tn], f), getattr(s2._tables[tn], f), equal_nan=True), (k, tn, f)
        s2.close()
        c = s.meta["counts"]
        for name, val in (("zero_objects", c["objects"] == 0), ("zero_places", c["places"] == 0), ("zero_rooms", c["rooms"] == 0),
                          ("orphan_objects", bool(len(s.objects) and (s.objects.parent_place < 0).any())),
                          ("multi_parent_objects", bool(len(s.objects) and (s.objects.n_parents > 1).any())),
                          ("unset_object_label", bool(len(s.objects) and (s.objects.label < 0).any())),
                          ("invalid_bbox", bool(len(s.objects) and np.isnan(s.objects.bbox_dim).any())),
                          ("rooms_with_parent", bool(len(s.rooms) and s.rooms.has_parent.any())),
                          ("mesh_" + mk, True), ("traj_" + s.meta["traj_source"], True)):
            kinds[name] = kinds.get(name, 0) + int(val)
        s.close()
    for needed in ("zero_objects", "zero_places", "zero_rooms", "orphan_objects", "multi_parent_objects", "unset_object_label", "invalid_bbox",
                   "mesh_full", "mesh_none", "traj_agents", "traj_trajectory.csv"):
        assert kinds.get(needed, 0) > 0, (needed, kinds)


def test_unusable_inputs_raise_a_typed_error(tmp_path):
    bad = tmp_path / "bad"
    (bad / "backend").mkdir(parents=True)
    cache = tmp_path / "cache"
    with pytest.raises(sn.SnapshotError):
        sn.Snapshot.open(bad, cache_root=cache)  # json missing
    for label, write in (("truncated", lambda p: p.write_text('{"nodes": [')), ("empty", lambda p: p.write_text("")),
                         ("binary", lambda p: p.write_bytes(os.urandom(512)))):
        write(bad / "backend" / "dsg_with_mesh.json")
        with pytest.raises(sn.SnapshotError):
            sn.Snapshot.open(bad, cache_root=cache)
        with pytest.raises(sn.SnapshotError):
            sn.Snapshot.build(bad)


@pytest.mark.parametrize("what", ["position", "quaternion"])
def test_nan_attributes_are_a_typed_error_or_flagged(tmp_path, what):
    run = tmp_path / what
    (run / "backend").mkdir(parents=True)
    gf.nonfinite_graph_json(run / "backend" / "dsg_with_mesh.json", what)
    try:
        s = sn.Snapshot.open(run, cache_root=tmp_path / "cache")
    except sn.SnapshotError:
        return
    assert any("non-finite" in b for b in s.check())
    s.close()


def test_damaged_cache_file_is_rebuilt(synthetic):
    run, cache = synthetic
    sn.Snapshot.open(run, cache_root=cache).close()
    cp = sn.cache_path(run, cache)
    cp.write_bytes(cp.read_bytes()[:1000])
    s = sn.Snapshot.open(run, cache_root=cache)
    assert not s.from_cache and s.check() == []
    s.close()


def test_injection_manifest_is_never_read(synthetic):
    run, cache = synthetic
    (run / "injection_manifest.json").write_text('{"fault": "label_confusion", "severity": 0.5, "seed": 0}')
    seen = gf.opened_paths_during(lambda: [sn.Snapshot.open(run, cache_root=cache).close() for _ in range(2)])
    assert seen and not any(os.path.basename(p) == "injection_manifest.json" for p in seen)


def test_real_reference_run_matches_the_known_counts():
    run = sn.RUNS_ROOT / "uh2_clean"
    if not (run / "backend" / "dsg_with_mesh.json").is_file():
        pytest.skip("reference run not available")
    with sn.Snapshot.open(run) as s:
        assert s.check() == []
        assert s.meta["counts"] == {"objects": 109, "places": 1068, "mplaces": 2774, "rooms": 8, "agents": 628, "mesh_vertices": 448062,
                                    "mesh_faces": 662823, "place_edges": 4128, "mplace_edges": 7007, "room_edges": 8}
        assert s.meta["graph_counts"] == {"nodes": 4588, "edges": 13578} and s.meta["edges_unrepresented"] == 0
        assert list(s.stats.layer_2_last) == [109, 105, 80, 105, 0, 0] and not s.rooms.has_parent.any()
        assert sorted(np.flatnonzero(s.objects.n_parents > 1).tolist()) == [9, 31, 58, 78]  # the four objects with two parent places
        assert gf.verify_against_dsg(run, s, agent_stamps=True) == []
