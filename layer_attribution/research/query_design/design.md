# Query-level failure: design (DRAFT 1, measured numbers are added in findings.md)

Status: draft written before the measurement pass. Module: `src/layer_attribution/queries.py`
(tests: `tests/queries/test_queries.py`, 33 tests, Windows python).

## 1. What the project needs from a query

A robot asks the scene graph for "the chair in room B". A failed query is the observable symptom that attribution starts
from, and the benign false-failure rate (runs with a negligible fault) decides which injected faults are visible at all.
The definition therefore has to be (a) a statement about what the graph returns, (b) stable under tiny perturbations, and
(c) rich enough to carry a support for query-local layer evidence.

## 2. Critique of the ad-hoc version (`D:/hydra_work/tools/analyze_runs.py`, block "query")

Re-implemented on the new views it reproduces the recorded numbers exactly (105 queries; success 0.571/0.571/0.629/0.419/0.352/0.752
for label_confusion/label_dropout/depth_noise/depth_holes/pose_drift/frame_drop at severity 0.5), so the critique below is about the
definition, not the data.

1. 105 queries instead of 109: it walks `has_parent()/get_parent()`, which exist only for nodes with exactly one parent, so the 4 objects
   with two parent places (O12, O77, O129, O161) are silently dropped.
2. Rooms are matched 1-1 by centroid Hungarian with a 3 m gate. A merge of two reference rooms into one run room leaves one of them
   unmatched, which the old code reports as `room_missing` although every object of that room is present under the merged room.
   Benign runs (severity 0.03) already merge rooms (R2+R3 in 7 of 12), so the old criterion has a benign failure rate up to 0.60.
3. `wrong_object` conflates "target absent, other chairs exist" with "target displaced"; no `wrong_label`, no `wrong_room`.
4. A run node can answer for several reference objects (one merged node between two chairs 0.8 m apart satisfies both).
5. Nothing is kept about which nodes were involved, so no query-local evidence can be computed.
6. The 1 m tolerance and the 3 m gate are unexplained.

## 3. Definition

### 3.1 Query generation (reference graph only)

One query per reference object that has an ancestor room (object -> parent place -> room). Rationale:

* 97 of the 109 reference objects share their (class, room) pair with at least one other object (only 12 pairs are unique), so
  "the chair in room B" is ill-posed for a pair-level query. The instance query carries the reference position as the
  disambiguating hint, the pair-level view is derived (`pair_groups`) and every query records `n_same_class_in_room`, the size of the
  reference answer set (1 = the definite description is unique).
* One chain per query (object -> its parent places -> room) keeps the support single-valued; a pair-level query with k objects would
  mix k supports and hide k-1 failures behind an any-of rule.
* Two parent places: all parents are kept in the support; the room of the parent place nearest to the object defines the query room
  (all rooms are kept in `ref_room_rows`; in the reference both parents of all 4 such objects lie in the same room).
* No ancestor room: not queried (`objects_without_room`); none in the reference.

A query stores: object id/class/target position, parent places, query room, answer-set size, and the other reference objects within
2*tol of the target (class, position) for the rival and explained-label rules below.

### 3.2 Resolver on a run graph

1. Room correspondence. Both graphs are rasterised on a common 0.5 m floor-plan grid; a cell belongs to the room of its nearest place
   if that place is within 1.0 m (so a room footprint is the union of its child places' neighbourhoods). The overlap coefficient
   `oc(i, j) = shared cells / min(|footprint_i|, |footprint_j|)` between reference room i and run room j is high when one room lies
   inside the other, which covers merges (reference room inside a run room) and splits (run fragment inside the reference room).
   Run room j REPRESENTS reference room i when `oc >= 0.3` and the shared area is at least 5 % of the reference room (no slivers).
   Many-to-one and one-to-many are allowed. Structure of a reference room: `missing` (no representing room), `one_to_one`, `merged`
   (a representing run room also represents another reference room), `split` (two or more representing rooms), `merged_split`.
2. Candidates: same-class run objects whose parent places belong to a representing room.
3. Success: a candidate within `tol` (3D, 1.0 m) of the target that is not claimed by a same-class rival (a reference object of the same
   class strictly closer to that node than the target is). Claiming stops one node from answering for two targets.
4. Failure modes, first match wins:
   * `room_missing`: no run room represents the reference room.
   * `wrong_room`: a same-class, unclaimed run node lies within `tol` of the target but under a room that does not represent the query room
     (including no room at all).
   * `wrong_label`: a run node of another class lies within `tol` of the target and no reference object of that class is as close to it as
     the target is (so a legitimately neighbouring computer does not count as a mislabelled chair).
   * `wrong_position`: same-class candidates exist under the representing rooms but none within `tol`.
   * `target_missing`: no same-class run node under the representing rooms.
5. Strict criterion (sensitivity analysis, not the headline): additionally `room_split`, `room_merged` (the query's room is split /
   merged in the run) and `duplicate` (two or more unclaimed candidates within `tol`) count as failures.

Why merges/splits/duplicates are flags and not failures in the headline criterion: the target is still retrievable (the hinted
instance is found), and the benign band of the strict criterion is unusable (see findings.md, section "strict").

### 3.3 Support kept per query

Reference side (`Query`): object id, class, target position, parent place rows+ids, query room row+id, all rooms, answer-set size,
neighbouring reference objects. Run side (`QueryResult`): representing run rooms (rows+ids), room structure, answering node (row, id,
error, parent places, rooms), number of near/same-class candidates, nearest same-class error under the representing rooms and anywhere,
the wrong_room witness (node, rooms) and the wrong_label witness (node, class).

## 4. Pending (filled by the measurement pass)

* tolerance justification from benign position errors; room matching comparison; benign band; masked faults; informativeness.
