# Project context (read this first)

ROB 535 course project, group 28 (Wei Liu, Charles Huang, Zhaochen Yang): **layer-wise failure attribution for Hydra
3D scene graphs**. A scene-graph query such as "the chair in room B" can fail. We inject ONE known fault at a time into
the inputs or parameters of one Hydra layer, then try to attribute the failed query to the responsible (most upstream)
layer, and score the attribution against the injected fault. Layers: mesh, objects, places, rooms (+ buildings). They
form a dependency DAG, not a chain (rooms come from places, objects from the labeled mesh). Proposed diagnosis methods:
(1) reference-based = compare each layer with a scene graph built from clean inputs, blame the most upstream layer whose
discrepancy exceeds a threshold calibrated on benign runs; (2) reference-free = consistency checks inside and across
layers; baselines = "blame the layer with the largest discrepancy" and "blame the layer where the failure is visible".

## Rules for every agent working here

- Work only inside `D:\Hydra\layer_attribution\` (code, docs, tests) and the result dirs named below. Do NOT edit any
  other part of `D:\Hydra` (Hydra C++/python sources and `config/` are read-only for us).
- Branch `zhaochen` is checked out. Do NOT run git commit / push / checkout / branch / stash / reset. Leave changes
  uncommitted. Never touch `main`.
- Code comments: terse fragments, the WHY only, no process narrative ("fixed after review", dates), no emojis.
- A sweep of ~60 Hydra runs (6 parallel, ~16 min each) is using most CPU and ~10 GB RAM in WSL. Keep your own jobs
  small: at most 2 CPU cores and 4 GB RAM, no extra full Hydra runs. If your task needs Hydra, use `--max-steps N`
  (truncated run) and an output name starting with `tmp_`, and delete `tmp_*` run dirs when done.
- Dev/test split (protects the final result): **seed 0 + all benign runs = development data**; seeds 1 and 2 are the
  held-out test set. Do not tune thresholds, features or rules on seeds 1/2, and do not look at per-run results of seeds 1/2
  while developing. Only the final evaluation script reads them.

## Environment

- Host: Windows 11, shell tools = PowerShell / Git Bash. WSL2 distro `Ubuntu-24.04` (user `hydra`, 12 cores, 24 GB).
- WSL python env for anything that touches Hydra / `spark_dsg`: `source ~/hydra_env.sh` (ROS 2 Jazzy + workspace +
  venv `~/venvs/hydra`, has numpy, scipy, matplotlib, spark_dsg, hydra_python). **Always call WSL through a script
  file**: `wsl -d Ubuntu-24.04 -- bash /mnt/d/Hydra/layer_attribution/<path>.sh` (inline quoting breaks; from Git Bash the
  `/mnt/...` argument gets rewritten, so call `wsl` from PowerShell). `D:` is `/mnt/d` inside WSL.
- Windows python with numpy/pillow/tifffile/scipy/pyyaml/pytest/pandas/imageio: `D:\hydra_work\venvs\c\Scripts\python.exe`.
  Set `PYTHONPATH=D:\Hydra\layer_attribution\src` to import `layer_attribution`; tests: `python -m pytest` from
  `D:\Hydra\layer_attribution`.
- Hydra is built in `~/hydra_ws` (WSL clone of the same repo, on main); do not rebuild it. Run one Hydra job with
  `scripts/run_job.sh NAME DATASET_DIR [CONFIG_PATH|-] [MAX_STEPS|-]` (output `~/runs/uh2/NAME`).
  `hydra_python.commands.run` is used through `src/layer_attribution/pipeline/run_hydra.py` (the `hydra` CLI itself is broken
  by an upstream import mismatch).

## Data layout

- Source scene: uHumans2 office (TESSE sim), converted to Hydra's MP3D-style folder: `D:\hydra_work\data\uhumans2_office_mp3d`
  (2077 frames, 720x480, depth float32 m, labels in the `uhumans2_office` label space = 21 labels; object labels
  5 chair, 6 bench, 7 couch, 10 computer, 12 painting, 13 plant, 18 trashcan). Hydra config: `config/datasets/uhumans2.yaml`,
  label space `config/label_spaces/uhumans2_office_label_space.yaml`.
- Injected datasets (hard-linked copies): `D:\hydra_work\data\faults\<run_name>\` with `injection_manifest.json`
  (ground truth: fault, severity, seed, params, expected_first_layer, per-frame stats). Injector:
  `src/layer_attribution/injection/inject.py` (docs: `docs/fault_injection.md`). Faults: label_confusion, label_dropout (objects);
  depth_noise, depth_holes, pose_drift, frame_drop (geometry: mesh + places).
- Hydra outputs live on the WSL disk: `~/runs/uh2/<run_name>/` (= `\\wsl.localhost\Ubuntu-24.04\home\hydra\runs\uh2\<run_name>`):
  `backend/dsg_with_mesh.json` (scene graph + mesh, the main artefact, ~75 MB), `backend/dsg.json`, `backend/mesh.ply`,
  `backend/trajectory.csv`, `backend/layer_*_statistics.csv`, `timing_stats.csv`, `hydra_config.yaml`, and a copy of
  `injection_manifest.json`. Load with `spark_dsg.DynamicSceneGraph.load(...)`.
- Run names: reference `uh2_clean` (clean inputs) and `uh2_clean_b` (repeat of the clean run). First sweep (seed 0,
  severity 0.5): `uh2_<fault>_s50` (legacy names). Sweep v2 names: `uh2_<fault>_s<NN>_r<seed>`, NN = severity x 100
  (s03 = benign calibration runs, s20/s50/s80 = main grid, r0/r1/r2 = seeds). The legacy `uh2_<fault>_s50` IS seed 0 /
  severity 0.5. Always take fault/severity/seed from the manifest copy inside the run dir, not from the name.
  The sweep status log is `D:\hydra_work\sweep_logs\status_sweep_main.jsonl` (one JSON line per finished job); a run is
  usable only if `backend/dsg_with_mesh.json` exists.
- Results go to `D:\hydra_work\results\<topic>\` (figures, csv, json). Large artefacts stay out of the repo.

## What we already know (single scene, seed 0, severity 0.5, descriptive)

Clean baseline: 448k mesh vertices, 109 objects, 1068 places, 8 rooms, 1 building. Hydra is deterministic: two clean runs
give a byte-identical mesh and identical node positions in all layers. Per-layer comparison against the clean reference
(`D:\hydra_work\tools\analyze_runs.py`, to be ported) plus the query test "object of class c in room R" (105 queries):

| fault | layers that changed | query success |
|---|---|---|
| label_confusion / label_dropout | objects only (mesh, places, rooms positions identical) | 0.571 / 0.571 |
| depth_noise | all layers (places covered 0.48, rooms 8 -> 7) | 0.629 |
| depth_holes | all layers (mesh vertices x1.29) | 0.419 |
| pose_drift | all layers (objects 109 -> 135, places covered 0.29) | 0.352 |
| frame_drop (25 % of frames) | places and rooms | 0.752 |

Caveats: label faults are drawn independently per frame (flicker, not instance-consistent); the fraction of object pixels
actually changed overshoots the target (about 70 % at severity 0.5) because whole blobs are selected.

## Facts verified by the first-round studies (2026-11-01, files under research/)

- Saved scene graph layout (`uh2_clean`): layer key `2` OBJECTS 109 nodes, `2[97]` AGENTS 628, `3` PLACES (GVD free-space
  places) 1068, `3[1]` MESH_PLACES (2D surface places, `Place2dNodeAttributes`, no parents) 2774, `4` ROOMS 8, `5` BUILDINGS 1.
  Parent/child edges: place -> object 113 (4 objects have two parents), place -> agent 627, room -> place 1068; no
  room -> building edge. So an object's room is the room of its parent place, and rooms are built from places only.
  (`research/dag_knobs/inspect_clean_backend.txt`)
- Determinism: `uh2_clean` vs `uh2_clean_b` JSON hashes differ ONLY in `voxblox_mesh_connections[*].block/.vertex` of place nodes
  (raw memory addresses); node ids, edges, all attribute fields, mesh vertices/faces/labels/timestamps and the backend status
  sequence are identical. Safe claim: "same input and config give the same scene graph up to pointer-valued bookkeeping
  fields". (`research/dsg_fields/out/determinism_uh2_clean_vs_uh2_clean_b.txt`)
- Rooms are a pure function of the final GVD places layer plus `room_finder` config; `research/config_faults/emulate_rooms.py`
  re-implements the room finder offline (used only to screen parameters, final config-fault runs must be real Hydra runs).
- Effective Hydra config of a run is printed in `<run>/logs/hydra_python.*.INFO.*` with `(default)` markers; a wrong factory
  type name is a hard error, a wrong value type only a warning. Always verify an override landed there.
- Benign runs (`*_s03_r0`, `*_s03_r1`, 12 runs) have finished and are valid development data for threshold calibration.

## Status update (2026-11-01 evening)

- Sweep v2 is COMPLETE: 66/66 jobs (60 new runs + the 6 legacy seed-0 severity-0.5 runs), no failures. Test runs (seeds 1 and 2)
  exist in `~/runs/uh2` but are off limits until the final evaluation.
- Phase 2 is running (`configs/sweep_phase2.json`, status `D:\hydra_work\sweep_logs\status_sweep_phase2.jsonl`): 12 full-length
  config-fault runs `uh2_cfg_<fault>_sNN` (deterministic, no seed; manifest = the config dir's `config_fault.json` copied into the run as
  `injection_manifest.json`, `corrupted_input == "config"`) and 15 extra benign runs `*_s03_r3/r4/r5`. Split rule: `layer_attribution.runs`.
  Hydra uses 7 cores while it runs, so keep your own jobs small.
- Modules that exist: `runs.py` (run index + split guard), `queries.py` + `query_views.py` (query definition, SceneView),
  `pipeline/config_faults.py`, `sweep/orchestrate.py`, `evaluation/metrics.py`, `injection/inject.py`. Findings so far: `docs/FINDINGS_ROUND1.md`.
  Design and evaluation protocol: `docs/DESIGN.md`. DAG with code evidence: `research/dag_knobs/hydra_dependency_dag.md`.
- Evidence groups (feature layer tag -> layer): mesh_geom -> mesh, mesh_sem -> objects (labels are the object layer's input),
  objects -> objects, places -> places, rooms -> rooms. Data-flow DAG over groups: mesh -> objects, places -> rooms; mesh and places are
  siblings (shared TSDF parent). Accept sets: label faults {objects}; depth/pose/frame_drop {mesh, places}; objects_* config {objects};
  rooms_* config {rooms}; places_* config {places}.

## Conventions for the new code

Package `src/layer_attribution/` with sub-packages `injection/`, `pipeline/`, `sweep/`, `evaluation/` and modules `runs.py`,
`queries.py`, `dag.py`; to be added: `snapshot.py`, `features.py`, `features_extract.py`, `calibration.py`, `diagnosis.py`.
Tests under `tests/`. Prefer plain numpy/scipy; type hints where they help; no heavy new dependencies.
Your token budget is limited (a parallel fan-out already hit the 5-hour usage limit twice): read each large file once, do not print
big files into your context, avoid re-running expensive batteries (cache), and write every deliverable file as soon as a first
working version exists, then refine it.
