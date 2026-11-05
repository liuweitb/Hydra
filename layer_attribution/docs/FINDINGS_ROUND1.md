# Round-1 findings (development data only; written by the coordinator from what the sub-agents left on disk)

The sub-agents of the two foundation workflows were killed by the usage limit before they returned. Their files survived;
this note records what is established and where, and what is not. Sources are under `research/`.

## Established

1. **DAG** (`dag_knobs/hydra_dependency_dag.md`, code-cited). Mesh and GVD places are siblings under the TSDF, not mesh -> places.
   Surface places ("MESH_PLACES") depend on the mesh and are a leaf. Labels are a channel of the mesh (semantic voxels), objects are
   clustered from labeled mesh vertices. Objects reach rooms only through the association edge object -> parent place -> room.
   Rooms are a pure function of the GVD places and the `room_finder` config. No loop closure happens in our runs, so pose drift is
   never corrected. Depth, pose and frame-drop faults enter mesh and places in parallel: accept set {mesh, places}.
   Label faults: objects (strict) or {objects, mesh} (lenient). `inject.py` metadata for pose_drift / frame_drop is out of date; use `dag.py`.
2. **Determinism** (`dsg_fields/out/determinism_*.txt`). Two clean runs differ only in pointer-valued `voxblox_mesh_connections`
   fields; every node, edge, attribute, mesh array and the backend status sequence are identical.
3. **Query definition** (`query_design/design.md`, `src/layer_attribution/queries.py`, 33 tests). The ad-hoc query test of the first
   sweep is flawed: its 1-1 room matching reports `room_missing` when Hydra merely merges two rooms, which gives a benign failure rate
   up to 0.60. The new criterion has a benign band of 0.14. Query failure rates of the dev runs (headline criterion):
   label faults 0.08 (s20) / 0.43 (s50); depth_noise 0.06 / 0.11; depth_holes 0.08 / 0.29 / 0.43 (s80); pose_drift 0.11 / 0.48;
   frame_drop 0.02 / 0.04 / 0.06. Faults inside the benign band are MASKED: most faults at severity 0.2, depth_noise at 0.5,
   frame_drop at every severity. **The "frame_drop fails 25 % of queries" statement made earlier came from the flawed test.**
4. **Benign runs are not inert** (`signal_probe/run_meta.csv`). Severity 0.03 is enough to change the rooms: depth_noise gives
   7 and 5 rooms, pose_drift 7 and 9, frame_drop 7 and 8 (reference 8); label_confusion adds 17-25 object nodes; depth_holes adds
   +26 % mesh vertices (a step response: 1.27 at s03, 1.45 at s20, 1.29 at s50, so its severity is not a graded dial).
   Hydra's room layer is fragile, so room features need wide thresholds and room-level faults are hard to see.
5. **Thresholds from benign runs make the rules much weaker** than a heuristic threshold suggests (`signal_probe/rule_eval.csv`,
   in-sample on development data): top-1 accuracy with all 12 benign runs ranges 0.38-0.69; with the mesh-neutral benign subset
   (selected by a property of the benign run itself, so partly circular) 0.62-1.0. depth_noise and frame_drop are mostly "none".
6. **Config faults** (`config_faults/`, `src/layer_attribution/pipeline/config_faults.py`, 11 faults tabulated, 4 recommended and
   validated on truncated runs): objects_min_size, objects_tol_merge (layer objects), rooms_min_size, rooms_max_dilation (layer rooms).
   Rooms are reproduced exactly by an offline emulator (screening only). Full-length runs: `configs/sweep_phase2.json`.

## Not established / open

- No agent produced a final findings.md for signal_probe, dsg_fields, config_faults or query_design; dose-response and rule
  results above are read from their tables, not from a written conclusion.
- Three snapshot formats exist (dsg_fields `Snapshot` schema 2 = canonical; signal_probe `Snap` npz; queries `SceneView`).
  They must be unified before the final evaluation (adapters from the canonical Snapshot).
- The signal_probe feature choices and rule results are in-sample on development data (the agent says so itself).
- No result on seeds 1/2 or on full-length config runs has been looked at.
