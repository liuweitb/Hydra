# Layer-wise failure attribution for Hydra 3D scene graphs

ROB 535 course project (group 28). A scene-graph query such as "the chair in room B" can fail. We inject ONE known fault at a
time into the inputs or parameters of one Hydra layer (mesh, objects, places, rooms), then try to name the layer that caused
the failure, and score the answer against the injected fault. Hydra itself is not modified; everything here sits next to it.

Read `docs/PROJECT_CONTEXT.md` (environment, data layout, rules) and `docs/DESIGN.md` (methods and evaluation protocol) first.
`docs/FINDINGS_ROUND1.md` records what the first studies established; the code-cited layer dependency analysis is
`research/dag_knobs/hydra_dependency_dag.md`.

## What is in here

| path | what |
|---|---|
| `src/layer_attribution/injection/` | seeded fault injector (label confusion / dropout, depth noise / holes, pose drift, frame drop) with a ground-truth manifest per run |
| `src/layer_attribution/pipeline/` | `run_hydra.py` (one Hydra run on a dataset folder), `config_faults.py` (parameter faults on clean data), `bag_to_mp3d.py` (uHumans2 bag -> Hydra folder) |
| `src/layer_attribution/sweep/orchestrate.py` | inject (Windows) -> Hydra (WSL) job queue, resume-safe, specs in `configs/` |
| `src/layer_attribution/runs.py` | run index and the development / test split (the held-out split needs `final=True`) |
| `src/layer_attribution/snapshot.py`, `features.py`, `features_extract.py` | numpy view of a saved scene graph and 99 per-layer features (reference-based and reference-free) |
| `src/layer_attribution/queries.py`, `query_views.py`, `query_table.py` | the query definition, failure modes and per-run query outcomes |
| `src/layer_attribution/dag.py`, `calibration.py`, `diagnosis.py` | layer dependency graph and accept sets, benign-run thresholds, the diagnosis rules and baselines |
| `src/layer_attribution/evaluation/` | `evaluate.py` (select a config on development runs, report), `metrics.py`, `figures.py` |
| `research/` | the exploratory studies and their raw outputs (not imported by `src/`) |
| `tests/` | about 300 tests; WSL ones need `scripts/wsl_pytest.sh` |

## Pipeline

Windows runs the injector and the job queue; WSL (Ubuntu-24.04) runs Hydra and everything that needs `spark_dsg`.

```
python -m layer_attribution.sweep.orchestrate --spec configs/sweep_main.json --workers 6     # sensor-fault grid
python -m layer_attribution.sweep.orchestrate --spec configs/sweep_phase2.json --workers 7   # config faults + extra benign runs
wsl bash scripts/wsl_tables.sh dev reference,repeat,benign,dev                                # feature + query tables
python -m layer_attribution.evaluation.evaluate select --features F --meta M --queries Q --out DIR
python -m layer_attribution.evaluation.evaluate report --features F --meta M --queries Q --out DIR [--splits test --final]
```

## Protocol in one paragraph

Hydra is deterministic, so there is no natural variance to calibrate on. Benign runs (a fault at severity 0.03) stand in for it:
a feature's threshold is 1.5 x its largest deviation on benign runs (floor: the smallest change worth calling a move). A layer's
score is the k-th largest strength among its features; a layer deviates at score >= 1. The DAG rule blames the most upstream
deviating layers, the baselines blame the largest, the deepest, or the layer where the queries fail. Faults whose query failure
rate stays inside the benign band are MASKED (no failure to explain) and are reported separately. Seed-0 runs and benign runs are
development data (they pick k and the feature filter); seeds 1 and 2 and the other config-fault severities are held out and are read
once, by the final report.

## Known limits

One scene and one trajectory (uHumans2 office). The reference-based method needs a clean run and is an upper bound. Thresholds
come from perturbation runs of the same scene. Label faults are drawn independently per frame (flicker, not instance-consistent)
and overshoot the target pixel fraction. Queries are generated from the clean reference graph, not from simulator ground truth.
Benign runs are not inert in Hydra's room layer (severity 0.03 changes the room count). The depth_holes severity is not a graded dial.
