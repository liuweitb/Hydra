"""Snapshot without spark_dsg: tables, invariants, npz round trip, cache key, errors, no manifest access."""

import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import scene_factory as sf

from layer_attribution import snapshot as sn

ROOT = pathlib.Path(__file__).resolve().parents[2]


def snap(**kw):
    return sf.snapshot_from_arrays(sf.make_scene(0, g=24, n_place=40, n_obj=12), **kw)


def seeded_run(tmp_path, name="uh2_x"):
    """Run directory with a dummy json and a cache entry whose source key matches it."""
    run, cache = tmp_path / name, tmp_path / "cache"
    (run / "backend").mkdir(parents=True)
    (run / "backend" / "dsg_with_mesh.json").write_text("{}")
    s = snap()
    s.meta["src"] = sn.source_key(run)
    s.meta["run"] = name
    s.save(sn.cache_path(run, cache))
    return run, cache, s


# ---------------------------------------------------------------- tables and invariants
def test_table_access_csr_and_length():
    s = snap()
    o = s.objects
    assert len(o) == 12 and len(s.places) == 40 and len(s.rooms) == 4 and len(s.stats) == 0
    assert np.array_equal(o.csr("mesh_conn_indptr", "mesh_conn_idx", 3), o.mesh_conn_idx[o.mesh_conn_indptr[3]:o.mesh_conn_indptr[4]])
    assert "bbox_rot" in o.fields() and "has_parent" in s.rooms.fields()
    with pytest.raises(AttributeError):
        o.no_such_field
    with pytest.raises(AttributeError):
        s.no_such_table
    assert sn.symbol((ord("p") << 56) + 17) == "p17"


def test_hand_built_snapshot_satisfies_the_invariants():
    assert snap().check() == []


@pytest.mark.parametrize("damage", [
    lambda s: s.objects.mesh_conn_indptr.__setitem__(-1, s.objects.mesh_conn_indptr[-1] + 1),
    lambda s: s.mesh.faces.__setitem__((0, 0), 10**6),
    lambda s: s.places.id.__setitem__(1, s.places.id[0]),
    lambda s: s.objects.pos.__setitem__((0, 0), np.nan),
    lambda s: s.place_edges.ij.__setitem__((0, 0), 10**6),
    lambda s: s.places.n_room_parents.__setitem__(0, 3),
    lambda s: s.places.room.__setitem__(0, 99),
    lambda s: s.rooms._arrays.pop("has_parent"),
    lambda s: s.rooms._arrays.__setitem__("deg", s.rooms.deg[:-1]),
    lambda s: s.stats._arrays.__setitem__("layer_2_last", np.zeros(3)),
], ids=["csr", "faces", "ids", "nonfinite", "edge", "room_parents", "room_row", "missing_field", "row_count", "stats_shape"])
def test_check_flags_damage(damage):
    s = snap()
    damage(s)
    assert s.check()


# ---------------------------------------------------------------- npz round trip
def test_save_load_round_trip_is_lossless_and_lazy(tmp_path):
    s = snap()
    s.save(tmp_path / "a" / "s.npz")
    assert not list((tmp_path / "a").glob("*.tmp"))
    with sn.Snapshot.load(tmp_path / "a" / "s.npz") as t:
        assert t.objects._arrays == {}  # nothing read yet
        assert t.meta == json.loads(json.dumps(s.meta)) and t.check() == []
        for tn in sn.TABLES:
            assert t._tables[tn].fields() == s._tables[tn].fields()
            for f in s._tables[tn].fields():
                a, b = getattr(s._tables[tn], f), getattr(t._tables[tn], f)
                assert a.dtype == b.dtype and np.array_equal(a, b, equal_nan=True), (tn, f)
    t.close()  # closing twice is fine
    m = sn.Snapshot.load(tmp_path / "a" / "s.npz").materialize()
    assert m._npz is None and m.objects.pos.shape == (12, 3)
    assert snap().mesh_subsample(1000).max() < len(snap().mesh.xyz)


def test_label_names_keys_are_ints():
    s = snap()
    s.meta["label_names"] = {"5": "chair", "7": "couch"}
    assert s.label_names == {5: "chair", 7: "couch"}


# ---------------------------------------------------------------- cache key and errors
def test_cache_path_carries_the_schema(tmp_path):
    assert sn.cache_path(tmp_path / "uh2_a", tmp_path / "c").name == f"uh2_a.snap{sn.SCHEMA}.npz" and sn.SCHEMA == 3


def test_open_serves_the_cache_while_the_source_stat_is_unchanged(tmp_path):
    run, cache, s = seeded_run(tmp_path)
    t = sn.Snapshot.open(run, cache_root=cache)
    assert t.from_cache and np.array_equal(t.objects.pos, s.objects.pos)
    t.close()
    with sn.Snapshot.open(str(run), cache_root=cache) as u:  # a string path works too
        assert u.from_cache


@pytest.mark.parametrize("change", ["json_size", "json_mtime", "trajectory_new", "stats_new"])
def test_open_rejects_the_cache_when_a_source_file_changed(tmp_path, change):
    run, cache, _ = seeded_run(tmp_path)
    b, js = run / "backend", run / "backend" / "dsg_with_mesh.json"
    if change == "json_size":
        js.write_text("{ }")
    elif change == "json_mtime":
        os.utime(js, ns=(js.stat().st_atime_ns, js.stat().st_mtime_ns + 5 * 10**9))
    elif change == "trajectory_new":
        (b / "trajectory.csv").write_text("timestamp[ns],x,y,z,qw,qx,qy,qz\n")
    else:
        (b / "layer_3_statistics.csv").write_text("x\n")
    with pytest.raises(sn.SnapshotError):  # stale -> rebuild is attempted and cannot succeed on a dummy json
        sn.Snapshot.open(run, cache_root=cache)


def test_open_rebuilds_instead_of_trusting_a_foreign_schema_or_a_damaged_file(tmp_path):
    run, cache, s = seeded_run(tmp_path)
    s.meta["schema"] = sn.SCHEMA - 1
    s.save(sn.cache_path(run, cache))
    with pytest.raises(sn.SnapshotError):
        sn.Snapshot.open(run, cache_root=cache)
    cp = sn.cache_path(run, cache)
    s.meta["schema"] = sn.SCHEMA
    s.save(cp)
    cp.write_bytes(cp.read_bytes()[:500])
    with pytest.raises(sn.SnapshotError):
        sn.Snapshot.open(run, cache_root=cache)


def test_open_flags_and_cache_switches(tmp_path):
    run, cache, _ = seeded_run(tmp_path)
    with pytest.raises(sn.SnapshotError):
        sn.Snapshot.open(run, cache_root=cache, rebuild=True)
    with pytest.raises(sn.SnapshotError):
        sn.Snapshot.open(run, cache_root=cache, cache=False)
    empty = tmp_path / "empty"
    (empty / "backend").mkdir(parents=True)
    with pytest.raises(sn.SnapshotError, match="missing"):
        sn.Snapshot.open(empty, cache_root=cache)
    with pytest.raises(sn.SnapshotError):
        sn.build_snapshot(empty)


def test_source_freshness_tolerates_unc_mtime_rounding_only(tmp_path):
    run, _, _ = seeded_run(tmp_path)
    now = sn.source_key(run)
    near = json.loads(json.dumps(now))
    near["mtime_ns"] += 500
    far = json.loads(json.dumps(now))
    far["mtime_ns"] += 5000
    assert sn.source_fresh(near, now) and not sn.source_fresh(far, now)
    assert set(now["side"]) == set(sn.SIDE_FILES) and all(v is None for v in now["side"].values())


def test_cache_hit_never_opens_the_injection_manifest(tmp_path):
    import graph_factory as gf

    run, cache, _ = seeded_run(tmp_path)
    (run / "injection_manifest.json").write_text('{"fault": "label_confusion"}')
    seen = gf.opened_paths_during(lambda: sn.Snapshot.open(run, cache_root=cache).close())
    assert seen and not any(os.path.basename(p) == "injection_manifest.json" for p in seen)


# ---------------------------------------------------------------- modules and command line
def test_importing_the_data_layer_does_not_import_spark_dsg():
    code = "import sys; import layer_attribution.snapshot, layer_attribution.features, layer_attribution.features_extract; print('spark_dsg' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_check_command_reports_ok_for_a_cached_run(tmp_path, capsys):
    run, cache, _ = seeded_run(tmp_path)
    assert sn.main(["check", str(run), "--cache-dir", str(cache)]) == 0
    assert "uh2_x: OK" in capsys.readouterr().out
    assert sn.main(["build", str(run), "--cache-dir", str(cache)]) == 0
    assert "Snapshot(uh2_x)" in capsys.readouterr().out
