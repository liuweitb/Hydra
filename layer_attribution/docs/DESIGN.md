# Design: layer-wise failure attribution (read after PROJECT_CONTEXT.md)

The code is the truth; this note states what the code does and why. Findings behind the choices: `FINDINGS_ROUND1.md`.

## 1. Claim to test

Given a scene graph on which a query fails, the layer that caused the failure can be named by walking the layer dependency
DAG to the most upstream layer whose evidence deviates, and this beats rules that ignore the dependencies. Two evidence sources:
reference-based (compare with a scene graph built from clean inputs; an upper bound, no robot has it) and reference-free
(features of one scene graph only). Ground truth is the injected fault (one fault per run, manifest in the run dir).
Honest framing: with single faults the answer is often easy; the benchmark, the propagation / masking analysis and the limits
of reference-free diagnosis are the contribution, not a headline accuracy.

## 2. Layers and the DAG (`dag.py`, evidence: `research/dag_knobs/hydra_dependency_dag.md`)

Evidence layers: mesh (geometry only), objects (object nodes plus the mesh per-vertex label channel, which is the object layer's
input), places (GVD free-space places), rooms. Data-flow DAG over them: mesh -> objects, places -> rooms; mesh and places are
siblings under the TSDF (the chain mesh -> places the proposal assumed is wrong and is kept only as a sensitivity variant).
Objects reach rooms only through the association edge object -> parent place -> room. Accept sets for top-1 scoring: label faults
{objects}; depth_noise, depth_holes, pose_drift, frame_drop {mesh, places} (they enter through the shared TSDF); config objects_* {objects};
config rooms_* {rooms}; config places_* {places}. Strict scoring requires the blame SET to equal the accept set.

## 3. Data protocol (`runs.py`, `evaluation/evaluate.py`)

- Reference `uh2_clean`, repeat `uh2_clean_b`. Hydra is deterministic, so there is no natural variance; perturbation runs stand in.
- Run splits: benign (severity <= 0.05, seeds 0, 1, 3, 4, 5), dev (seed 0 sensor faults at 0.2 / 0.5 / 0.8; one severity per config fault),
  test (seeds 1, 2; the other config severities). `select` and every `final=False` path never read test; the CLIs need `--final`.
- Benign seeds are split: CALIBRATION_SEEDS (3, 4, 5) set the thresholds and are disjoint from the dev seed (0) and the test seeds (1, 2),
  because the injector draws with `rng([seed, frame])` independent of severity, so a benign run of seed s is a scaled or nested copy of the
  fault runs of seed s. VALIDATION_SEEDS (0, 1) are never used for thresholds: they give the held-out false-alarm rate.
- BENIGN_EXCLUDE = depth_holes: its severity is a step response (+27 % mesh vertices already at 0.03), so its benign runs do not calibrate.
- Config faults are deterministic on clean data: one run per (fault, severity), no seed.
- One scene; thresholds come from perturbation runs of the same scene, so absolute accuracy is optimistic.

## 4. Queries (`queries.py`, design in `research/query_design/design.md`)

One query per reference object that has a room (object -> parent place -> room), resolved on the run graph with footprint-overlap
room matching and a 1 m tolerance; failure modes room_missing, wrong_room, wrong_label, wrong_position, target_missing. A fault run is
MASKED when its failure rate does not exceed the largest failure rate of the same fault family's benign runs (pooled over all benign
runs for config faults, which have no negligible-dose runs). Masked runs have no failure to explain: reported, not scored. The pooled
band is reported as a sensitivity.

## 5. Evidence (`features.py`, `calibration.py`)

99 features per run, reference-based (value against the reference) or reference-free (the run alone; verified by static and
empirical checks that they never read the reference). Deviation d = |x - center|; center = ideal for reference-based features, the
median over reference and calibration runs for reference-free ones. Threshold = max(margin x largest calibration deviation, floor);
strength = d / threshold; a feature moves at strength >= 1. Features without an evidence layer (query-derived, trajectory, process
counters) can never enter a layer score. Layer score = k-th largest strength among the layer's usable features; a usable feature that is
undefined on a run counts as strength 1 (the structure it measures is gone); a layer deviates at score >= 1.

## 6. Methods and baselines (`diagnosis.py`)

dag_rb / dag_rf: blame the most upstream deviating layers (roots of the deviating set), strongest first; naive_*: largest score, ignoring
dependencies; downstream_*: deepest deviating layer (where the failure is visible); symptom: the layer holding most failed-query symptoms
(features unused); constants const_<layer> and prior_dev (best constant on dev); exact uniform random; label-permutation null.
All abstain (empty blame) when no layer deviates, which counts as a miss on a scored run.
Layer scores are not comparable across layers of different size (the naive rule favours big groups); this is stated, not corrected.

## 7. Selection and evaluation

`select` grid: mode (rb, rf) x k (1, 2, 3) x count-based features excluded x margin (1.5, 2, 3). Rule per mode: among configs whose held-out
false-alarm rate (validation seeds) is <= 0.05 take the best dev macro accuracy of the DAG rule; if none qualifies the least-alarming ones
stay in the tie-break (`cap_met = false` is reported); ties go to the larger margin, larger k, then count-based features excluded. The chosen
config is written to `chosen_config.json` and its hash goes into the provenance of every report. `report --splits test --final` is run once.
Sensor faults and config faults are separate blocks (separate macros and priors). Per block: lenient and strict accuracy per fault and
macro (interval only for n >= 5, runs are not independent), all fault runs including masked ones, the pooled-band sensitivity, constants,
exact random, the label-permutation null, confusion, which accept-set member is named, the masked table, the propagation table, false
alarms (held-out benign runs, leave-one-out by run and by family). Anything done after seeing test results is labelled post hoc.

## 8. Known limits

One scene and trajectory; reference-based is an upper bound; thresholds are scene-specific; label faults are per-frame independent
(not instance-consistent) and overshoot the target pixel fraction; queries come from the clean reference graph, not simulator ground truth;
benign runs are not inert in the room layer; calibration is per fault family (a family absent from calibration is detected at 3 % dose), so
held-out accuracy measures same-family generalisation over seeds; the macro is dominated by few runs per fault; objects with two parent
places count as unattached in a few features.
