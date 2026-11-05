"""Query-level failure on synthetic scenes: every failure mode, room matching, support, band, builders.

Scene: room A = [0,6]x[0,6] and room B = [6.5,12.5]x[0,6], places on a 0.5 m grid, objects attached to the nearest place.
"""

import json

import numpy as np
import pytest

from layer_attribution import queries as Q
from layer_attribution.queries import QueryConfig, SceneView
from layer_attribution.query_views import view_from_snapshot

CHAIR, BENCH, COUCH, COMPUTER, PLANT = 5, 6, 7, 10, 13
NAMES = {CHAIR: "chair", BENCH: "bench", COUCH: "couch", COMPUTER: "computer", PLANT: "plant"}
A = (0.0, 0.0, 6.0, 6.0)
B = (6.5, 0.0, 12.5, 6.0)


def obj(label, x, y, z=1.0, **kw):
    return {"label": label, "pos": (x, y, z), **kw}


def make_view(name, rooms, objects):
    """rooms {key: rectangle}; objects with optional attach=<room key>, parents=<place rows>, orphan=True."""
    keys = sorted(rooms)
    place_pos, place_room = [], []
    for k, key in enumerate(keys):
        x0, y0, x1, y1 = rooms[key]
        for x in np.arange(x0, x1 + 1e-9, 0.5):
            for y in np.arange(y0, y1 + 1e-9, 0.5):
                place_pos.append((x, y, 1.0))
                place_room.append(k)
    place_pos, place_room = np.array(place_pos, dtype=float).reshape(-1, 3), np.array(place_room, dtype=np.int32)
    indptr, idx = [0], []
    for o in objects:
        if o.get("orphan"):
            rows = []
        elif "parents" in o:
            rows = list(o["parents"])
        else:
            cand = np.arange(len(place_pos))
            if "attach" in o:
                cand = cand[place_room == keys.index(o["attach"])]
            d = np.linalg.norm(place_pos[cand] - np.array(o["pos"]), axis=1)
            rows = [int(cand[int(np.argmin(d))])]
        idx += rows
        indptr.append(len(idx))
    centers = [((rooms[k][0] + rooms[k][2]) / 2, (rooms[k][1] + rooms[k][3]) / 2, 1.0) for k in keys]
    return SceneView(
        name=name, obj_id=np.array([(ord("O") << 56) | i for i in range(len(objects))], dtype=np.int64),
        obj_pos=np.array([o["pos"] for o in objects], dtype=float).reshape(-1, 3),
        obj_label=np.array([o["label"] for o in objects], dtype=np.int32), obj_par_indptr=np.array(indptr, dtype=np.int64),
        obj_par_idx=np.array(idx, dtype=np.int32),
        place_id=np.array([(ord("p") << 56) | i for i in range(len(place_pos))], dtype=np.int64), place_pos=place_pos,
        place_radius=np.full(len(place_pos), 0.5, dtype=np.float32), place_room=place_room,
        room_id=np.array([(ord("R") << 56) | i for i in range(len(keys))], dtype=np.int64),
        room_pos=np.array(centers, dtype=float).reshape(-1, 3), label_names=dict(NAMES))


REF_OBJECTS = [
    obj(CHAIR, 2, 2),  # 0 chair in A
    obj(CHAIR, 10, 2),  # 1 chair in B
    obj(COUCH, 3, 5),  # 2 only couch in A
    obj(COMPUTER, 4, 4.5),  # 3 computer in A
    obj(PLANT, 9, 5),  # 4 plant in B
]


@pytest.fixture(scope="module")
def ref():
    return make_view("ref", {"A": A, "B": B}, REF_OBJECTS)


def run_with(objects, rooms=None, name="run"):
    return make_view(name, rooms or {"A": A, "B": B}, objects)


def modes(ref, run, cfg=QueryConfig(), strict=False):
    ev = Q.evaluate(ref, run, cfg)
    return [r.mode_strict if strict else r.mode for r in ev.results]


def replace(i, new):
    objs = list(REF_OBJECTS)
    objs[i:i + 1] = new if isinstance(new, list) else [new]
    return objs


# ---------------------------------------------------------------- queries and support
def test_one_query_per_reference_object_with_support(ref):
    qs = Q.build_queries(ref)
    assert [q.ref_row for q in qs] == [0, 1, 2, 3, 4] and [q.qid for q in qs] == [0, 1, 2, 3, 4]
    q = qs[0]
    assert q.label_name == "chair" and np.allclose(q.pos, (2, 2, 1))
    assert q.ref_room_row == 0 and q.ref_room_id == ref.room_id[0] and q.ref_room_rows == (0,)
    assert len(q.ref_place_rows) == 1 and q.ref_place_ids == (int(ref.place_id[q.ref_place_rows[0]]),)
    assert np.allclose(ref.place_pos[q.ref_place_rows[0]], (2, 2, 1))
    assert qs[1].ref_room_row == 1


def test_reference_answer_set_size_marks_ambiguous_pairs():
    objs = [obj(CHAIR, 1, 1), obj(CHAIR, 4, 4), obj(CHAIR, 10, 2), obj(COUCH, 3, 5)]
    v = make_view("r", {"A": A, "B": B}, objs)
    qs = Q.build_queries(v)
    assert [q.n_same_class_in_room for q in qs] == [2, 2, 1, 1]
    groups = Q.pair_groups(qs)
    assert groups[(CHAIR, 0)] == [0, 1] and groups[(CHAIR, 1)] == [2] and groups[(COUCH, 0)] == [3]


def test_object_without_ancestor_room_is_not_queried():
    objs = REF_OBJECTS + [obj(CHAIR, 5, 5, orphan=True)]
    v = make_view("r", {"A": A, "B": B}, objs)
    assert Q.objects_without_room(v) == [5]
    assert len(Q.build_queries(v)) == 5


def test_two_parent_object_keeps_both_places_and_nearest_decides_the_room(ref):
    # parents: a place of room A near (6,3) and a place of room B near (6.5,3); the object sits closer to B
    rows_a = np.flatnonzero((ref.place_room == 0) & (np.isclose(ref.place_pos[:, 0], 6.0)) & (np.isclose(ref.place_pos[:, 1], 3.0)))
    rows_b = np.flatnonzero((ref.place_room == 1) & (np.isclose(ref.place_pos[:, 0], 6.5)) & (np.isclose(ref.place_pos[:, 1], 3.0)))
    v = make_view("r", {"A": A, "B": B}, [obj(CHAIR, 6.4, 3, parents=[int(rows_a[0]), int(rows_b[0])])])
    (q,) = Q.build_queries(v)
    assert len(q.ref_place_ids) == 2 and set(q.ref_room_rows) == {0, 1}
    assert q.ref_room_row == 1  # nearest parent place wins
    assert Q.node_name(q.ref_id) == "O(0)" and Q.node_name(v.place_id[0]) == "p(0)"


# ---------------------------------------------------------------- identity and failure modes
def test_identical_graph_all_ok_and_strict_ok(ref):
    run = run_with(REF_OBJECTS)
    ev = Q.evaluate(ref, run)
    assert all(r.ok and r.mode == Q.OK and r.mode_strict == Q.OK for r in ev.results)
    assert all(r.hit_err_m == 0 and r.n_near == 1 and r.room_structure == "one_to_one" for r in ev.results)
    s = ev.summary()
    assert s["n_queries"] == 5 and s["n_fail"] == 0 and s["failure_rate"] == 0
    assert ev.results[0].hit_id == run.obj_id[0] and ev.results[0].hit_place_ids == (int(run.place_id[run.parents(0)[0]]),)
    assert ev.results[0].rep_room_ids == (int(run.room_id[0]),)


def test_small_position_error_inside_tolerance_is_ok(ref):
    r = Q.evaluate(ref, run_with(replace(0, obj(CHAIR, 2.6, 2.5)))).results[0]
    assert r.ok and r.hit_err_m == pytest.approx(np.hypot(0.6, 0.5))


def test_wrong_position_same_class_in_room_but_too_far(ref):
    r = Q.evaluate(ref, run_with(replace(0, obj(CHAIR, 4.0, 2.0)))).results[0]
    assert r.mode == Q.WRONG_POSITION and not r.ok
    assert r.n_same_class_in_rooms == 1 and r.err_in_rooms_m == pytest.approx(2.0) and r.hit_row == -1


def test_target_missing_no_same_class_node_in_matched_room(ref):
    objs = [o for i, o in enumerate(REF_OBJECTS) if i != 0]  # chair of room A removed, chair of room B stays
    ev = Q.evaluate(ref, run_with(objs))
    assert ev.results[0].mode == Q.TARGET_MISSING
    assert ev.results[0].err_in_rooms_m == np.inf and ev.results[0].err_any_m == pytest.approx(np.hypot(8, 0))
    assert ev.results[1].ok  # the chair of room B is unaffected


def test_wrong_label_node_at_target_carries_another_class(ref):
    r = Q.evaluate(ref, run_with(replace(0, obj(BENCH, 2, 2)))).results[0]
    assert r.mode == Q.WRONG_LABEL and r.other_class_label == BENCH and r.other_class_id != -1


def test_label_error_takes_precedence_over_a_far_same_class_node(ref):
    objs = replace(0, [obj(BENCH, 2, 2), obj(CHAIR, 5, 5)])  # a chair exists in the room, far from the target
    assert Q.evaluate(ref, run_with(objs)).results[0].mode == Q.WRONG_LABEL


def test_neighbour_of_another_class_explained_by_the_reference_is_not_a_label_error():
    ref_objs = [obj(CHAIR, 2, 2), obj(COMPUTER, 2.5, 2)]
    ref = make_view("ref", {"A": A}, ref_objs)
    run = make_view("run", {"A": A}, [obj(COMPUTER, 2.5, 2)])  # chair lost, computer intact 0.5 m away
    assert Q.evaluate(ref, run).results[0].mode == Q.TARGET_MISSING
    run2 = make_view("run2", {"A": A}, [obj(COMPUTER, 2.0, 2), obj(COMPUTER, 2.5, 2)])  # chair relabelled computer
    assert Q.evaluate(ref, run2).results[0].mode == Q.WRONG_LABEL


def test_wrong_room_target_found_under_another_room(ref):
    r = Q.evaluate(ref, run_with(replace(0, obj(CHAIR, 2, 2, attach="B")))).results[0]
    assert r.mode == Q.WRONG_ROOM and r.elsewhere_row >= 0 and r.elsewhere_room_ids == (int(ref.room_id[1]),)


def test_orphan_target_node_counts_as_wrong_room(ref):
    r = Q.evaluate(ref, run_with(replace(0, obj(CHAIR, 2, 2, orphan=True)))).results[0]
    assert r.mode == Q.WRONG_ROOM and r.elsewhere_room_ids == ()


def test_room_missing_when_no_run_room_overlaps_the_reference_room(ref):
    objs = [o for i, o in enumerate(REF_OBJECTS) if i in (0, 2, 3)]  # everything in room B is gone
    ev = Q.evaluate(ref, run_with(objs, rooms={"A": A}))
    assert [r.mode for r in ev.results] == [Q.OK, Q.ROOM_MISSING, Q.OK, Q.OK, Q.ROOM_MISSING]
    assert ev.results[1].rep_room_rows == () and ev.results[1].room_structure == "missing"
    assert ev.match.coverage[1] < 0.2 < 0.99 < ev.match.coverage[0]  # only the border strip of B is near a run place


def test_empty_run_graph_fails_every_query_as_room_missing(ref):
    empty = make_view("empty", {}, [])
    ev = Q.evaluate(ref, empty)
    assert [r.mode for r in ev.results] == [Q.ROOM_MISSING] * 5
    assert ev.summary()["failure_rate"] == 1.0


def test_run_without_objects_but_with_rooms_is_target_missing(ref):
    ev = Q.evaluate(ref, make_view("noobj", {"A": A, "B": B}, []))
    assert [r.mode for r in ev.results] == [Q.TARGET_MISSING] * 5


# ---------------------------------------------------------------- merged / split rooms
def test_merged_rooms_do_not_fail_the_query_but_are_flagged(ref):
    run = run_with(REF_OBJECTS, rooms={"M": (0.0, 0.0, 12.5, 6.0)})
    ev = Q.evaluate(ref, run)
    assert all(r.ok for r in ev.results) and {r.room_structure for r in ev.results} == {"merged"}
    assert [r.mode_strict for r in ev.results] == [Q.ROOM_MERGED] * 5
    assert ev.summary(strict=True)["failure_rate"] == 1.0 and ev.summary()["failure_rate"] == 0.0


def test_centroid_hungarian_turns_a_merge_into_room_missing(ref):
    run = run_with(REF_OBJECTS, rooms={"M": (0.0, 0.0, 12.5, 6.0)})
    cfg = QueryConfig(room_method="centroid", centroid_gate_m=4.0)
    got = modes(ref, run, cfg)
    assert got.count(Q.ROOM_MISSING) in (2, 3) and Q.OK in got  # one of the two merged rooms loses the 1-1 assignment
    assert all(m == Q.ROOM_MISSING for m in modes(ref, run, QueryConfig(room_method="centroid")))  # both farther than 3 m


def test_split_room_keeps_every_fragment_as_representing(ref):
    rooms = {"A": A, "B1": (6.5, 0.0, 9.5, 6.0), "B2": (10.0, 0.0, 12.5, 6.0)}
    ev = Q.evaluate(ref, run_with(REF_OBJECTS, rooms=rooms))
    assert all(r.ok for r in ev.results)
    assert ev.results[1].room_structure == "split" and len(ev.results[1].rep_room_rows) == 2
    assert ev.results[4].mode_strict == Q.ROOM_SPLIT and ev.results[0].mode_strict == Q.OK


def test_small_fragment_is_not_a_representing_room_when_below_min_share(ref):
    rooms = {"A": A, "B1": (6.5, 0.0, 12.5, 3.5), "B2": (6.5, 4.0, 12.5, 6.0)}  # B2 holds about 40 % of B
    run = run_with(REF_OBJECTS, rooms=rooms)  # the plant of B at (9,5) is attached to a place of B2
    assert modes(ref, run)[4] == Q.OK
    assert modes(ref, run, QueryConfig(min_ref_share=0.5))[4] == Q.WRONG_ROOM  # B2 no longer represents B
    assert modes(ref, run, QueryConfig(min_ref_share=0.5))[1] == Q.OK  # the chair under B1 is unaffected


def test_footprint_overlap_matrix_identity_and_merge(ref):
    same = Q.match_rooms(ref, run_with(REF_OBJECTS))
    assert np.allclose(same.oc, np.eye(2)) and np.allclose(same.coverage, 1) and same.structures() == ["one_to_one"] * 2
    merged = Q.match_rooms(ref, run_with(REF_OBJECTS, rooms={"M": (0.0, 0.0, 12.5, 6.0)}))
    assert merged.rep.shape == (2, 1) and merged.rep.all() and merged.structures() == ["merged", "merged"]
    assert merged.oc[:, 0] == pytest.approx([1.0, 1.0])


def test_footprint_matching_is_invariant_to_a_small_shift_of_the_run_places(ref):
    shifted = run_with(REF_OBJECTS)
    shifted.place_pos = shifted.place_pos + np.array([0.4, 0.3, 0.0])
    m = Q.match_rooms(ref, shifted)
    assert m.rep.tolist() == [[True, False], [False, True]]


# ---------------------------------------------------------------- ambiguity, duplicates, rivals
def test_duplicate_nodes_near_the_target_stay_ok_but_fail_the_strict_criterion(ref):
    run = run_with(replace(0, [obj(CHAIR, 2, 2), obj(CHAIR, 2.3, 2)]))
    ev = Q.evaluate(ref, run)
    assert ev.results[0].ok and ev.results[0].n_near == 2 and ev.results[0].mode_strict == Q.DUPLICATE
    assert ev.results[0].hit_id == run.obj_id[0]  # nearest of the two


def test_a_run_node_cannot_answer_for_two_targets_when_a_rival_is_closer():
    ref = make_view("ref", {"A": A}, [obj(CHAIR, 2, 2), obj(CHAIR, 2.8, 2)])
    run = make_view("run", {"A": A}, [obj(CHAIR, 2.5, 2)])  # one merged node, 0.5 from the first and 0.3 from the second
    got = Q.evaluate(ref, run)
    assert [r.mode for r in got.results] == [Q.WRONG_POSITION, Q.OK]
    loose = Q.evaluate(ref, run, QueryConfig(exclusive=False))
    assert [r.mode for r in loose.results] == [Q.OK, Q.OK]


def test_exact_tie_between_rivals_does_not_claim():
    ref = make_view("ref", {"A": A}, [obj(CHAIR, 2, 2), obj(CHAIR, 3, 2)])
    run = make_view("run", {"A": A}, [obj(CHAIR, 2.5, 2)])
    assert [r.mode for r in Q.evaluate(ref, run).results] == [Q.OK, Q.OK]


def test_resolve_rejects_queries_built_with_a_smaller_tolerance(ref):
    qs = Q.build_queries(ref, QueryConfig(tol_m=0.5))
    with pytest.raises(ValueError):
        Q.resolve(qs, run_with(REF_OBJECTS), QueryConfig(tol_m=1.0), ref=ref)


def test_tolerance_boundary(ref):
    inside = Q.evaluate(ref, run_with(replace(0, obj(CHAIR, 3.0, 2.0)))).results[0]  # exactly 1.0 m
    outside = Q.evaluate(ref, run_with(replace(0, obj(CHAIR, 3.01, 2.0)))).results[0]
    assert inside.ok and outside.mode == Q.WRONG_POSITION
    assert Q.evaluate(ref, run_with(replace(0, obj(CHAIR, 3.01, 2.0))), QueryConfig(tol_m=1.5)).results[0].ok


def test_z_matters_for_the_distance(ref):
    assert Q.evaluate(ref, run_with(replace(0, obj(CHAIR, 2, 2, 2.2)))).results[0].mode == Q.WRONG_POSITION


# ---------------------------------------------------------------- determinism, rows, io
def test_evaluation_is_deterministic_and_rows_are_flat(ref):
    run = run_with(replace(0, obj(CHAIR, 4.0, 2.0)))
    a, b = Q.evaluate(ref, run).rows(), Q.evaluate(ref, run).rows()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True) and len(a) == 5  # nan-safe comparison
    row = a[0]
    assert row["ref_node"] == "O(0)" and row["ref_room"] == "R(0)" and row["mode"] == Q.WRONG_POSITION
    assert all(not isinstance(v, (tuple, list, np.ndarray)) for v in row.values())


def test_scene_view_roundtrip(tmp_path, ref):
    ref.obj_bbox_dim = np.ones((ref.n_obj, 3), dtype=np.float32)
    ref.save(tmp_path / "v.npz")
    back = SceneView.load(tmp_path / "v.npz")
    for k in SceneView.ARRAYS:
        assert np.array_equal(getattr(back, k), getattr(ref, k)), k
    assert back.name == ref.name and back.label_names == ref.label_names and np.array_equal(back.obj_bbox_dim, ref.obj_bbox_dim)
    ref.obj_bbox_dim = None


class _Table:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def fields(self):
        return sorted(self.__dict__)


def test_snapshot_adapter_matches_the_view_for_single_parent_objects(ref):
    parent = np.array([int(ref.parents(i)[0]) if len(ref.parents(i)) else -1 for i in range(ref.n_obj)], dtype=np.int32)

    class Snap:
        meta = {"run": "ref"}
        label_names = dict(NAMES)
        objects = _Table(id=ref.obj_id, pos=ref.obj_pos, label=ref.obj_label, parent_place=parent)
        places = _Table(id=ref.place_id, pos=ref.place_pos, dist=ref.place_radius, room=ref.place_room)
        rooms = _Table(id=ref.room_id, pos=ref.room_pos)

    v = view_from_snapshot(Snap())
    for k in SceneView.ARRAYS:
        assert np.array_equal(getattr(v, k), getattr(ref, k)), k
    assert Q.evaluate(ref, v).summary()["n_fail"] == 0


# ---------------------------------------------------------------- benign band
def test_benign_band_and_masked_rule():
    band = Q.benign_band([0.0, 0.028, 0.009, 0.055, 0.138])
    assert band.n_runs == 5 and band.hi_rate == 0.138 and band.rates == (0.0, 0.009, 0.028, 0.055, 0.138)
    assert Q.is_masked(0.138, band) and Q.is_masked(0.02, band) and not Q.is_masked(0.139, band)
    with pytest.raises(ValueError):
        Q.benign_band([])


def test_summarize_counts_modes():
    ev_modes = [Q.OK, Q.WRONG_ROOM, Q.WRONG_ROOM, Q.TARGET_MISSING]
    base = dict(room_structure="one_to_one", rep_room_rows=(), rep_room_ids=(), hit_row=-1, hit_id=-1, hit_err_m=0.0,
                hit_place_ids=(), hit_room_ids=(), room_coverage=1.0, room_best_oc=1.0, n_near=0, n_same_class_in_rooms=0,
                err_in_rooms_m=0.0, err_any_m=0.0, nearest_any_id=-1, elsewhere_row=-1, elsewhere_id=-1, elsewhere_room_ids=(),
                other_class_row=-1, other_class_id=-1, other_class_label=-1)
    fake = [Q.QueryResult(qid=i, ok=m == Q.OK, mode=m, mode_strict=m, **base) for i, m in enumerate(ev_modes)]
    s = Q.summarize(fake)
    assert s["n_queries"] == 4 and s["n_fail"] == 3 and s["failure_rate"] == 0.75
    assert s["modes"][Q.WRONG_ROOM] == 2 and s["modes"][Q.TARGET_MISSING] == 1 and s["modes"][Q.WRONG_LABEL] == 0


def test_support_fields_of_a_failed_query(ref):
    run = run_with(replace(0, obj(CHAIR, 4.0, 2.0)))  # displaced by 2 m
    r = Q.evaluate(ref, run).results[0]
    assert r.nearest_any_id == int(run.obj_id[0]) and r.err_any_m == pytest.approx(2.0) and r.err_in_rooms_m == pytest.approx(2.0)
    assert r.room_coverage == pytest.approx(1.0) and r.room_best_oc == pytest.approx(1.0)
    lost = Q.evaluate(ref, make_view("noobj", {"A": A, "B": B}, [])).results[0]
    assert lost.nearest_any_id == -1 and lost.err_any_m == np.inf
