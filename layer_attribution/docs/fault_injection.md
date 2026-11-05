# Fault injection for Hydra MP3D-format datasets

`inject.py` copies a dataset folder (`camera_info.yaml`, `color/`, `depth/`, `labels/`, `poses.csv`) and applies
exactly ONE seeded fault of severity in [0, 1]. Untouched files are hard-linked (`--copy` to copy). Every run writes
`injection_manifest.json`: the ground-truth fault record (fault, resolved params, seed, `expected_first_layer`, and
per-frame statistics re-measured from the files actually written).

```
venv: D:\hydra_work\venvs\c\Scripts\python.exe        (numpy pillow tifffile pyyaml scipy; pytest imageio pandas for tests)
python inject.py SRC DST --fault label_confusion --severity 0.4 --seed 0
python inject.py SRC DST --fault label_dropout --severity 0.3 --param granularity=pixel
python inject.py SRC DST --fault depth_noise --severity 0.5 --param rel_sigma_max=0.08
python inject.py SRC DST --fault depth_holes --severity 0.6
python inject.py SRC DST --fault pose_drift --severity 0.5 --param mode=bias
python inject.py SRC DST --fault frame_drop --severity 0.4 --param chunk_len=15
python inject.py SRC DST --fault label_dropout --severity 0.3 --label-space D:\Hydra\config\label_spaces\ade20k_mp3d_label_space.yaml
python synthetic.py D:\hydra_work\tools\fault_injection\_syn --frames 30      # tiny test dataset
python -m pytest                                                              # 42 tests
```
DST must not exist or be empty and must not overlap SRC; keep it on the same volume for hard links. Linked files share
bytes with SRC, so never edit them in place.

| fault | expected_first_layer | severity 1 means |
|---|---|---|
| `label_confusion` | objects | every object blob relabelled via `confusion_map` (default: look-alike pairs, ade20k_mp3d ids) |
| `label_dropout` | objects | every object blob set to `void_label` (0) |
| `depth_noise` | mesh (also places) | gaussian, sigma = `rel_sigma_max` (0.05) x depth; invalid pixels untouched |
| `depth_holes` | mesh (also places) | holes cover `max_hole_area_frac` (0.5) of each frame; 20% of frames lose all depth |
| `pose_drift` | mesh | per-frame random walk: 0.02 m xyz, 0.3 deg yaw (`bias` mode: constant 0.005 m, 0.1 deg per frame) |
| `frame_drop` | all | `max_drop_frac` (0.5) of frames removed in chunks of <= `chunk_len` (10) |

Label faults select whole 4-connected blobs of one class until `severity x eligible pixels` is reached
(`granularity=pixel` picks pixels independently); the overshoot is at most one blob and is reported per frame.
Pose drift is odometry-style: the accumulated yaw error rotates every later increment, frame 0 is never changed.
Frame drop renumbers files 0..M-1 and keeps original timestamps (loader indexes files by `poses.csv` row).

Manifest keys: `fault, severity, seed, params, expected_first_layer, also_hits_directly, corrupted_input, layer_order,
n_frames_in/out, n_units_modified, summary{key:{mean,max,sum}}, extra, frame_stats[]`. Per frame: label faults give
`n_changed, frac_changed, frac_object_changed`; depth faults `n_changed` / `n_newly_invalid`, `mean_abs_err_m`,
`frame_dropped`; pose drift `trans_error_m, dx/dy/dz_m, yaw_error_deg`; frame drop `src_frame, dropped_before` plus
`extra.dropped_chunks_src, kept_src_indices`. All frames carry `frame` (output index) and `src_frame`.

Format facts (from spark_dataset_interfaces `FileDataLoader` and Hydra `python/bindings`): depth is single-channel float32
metres (NaN invalid) or uint16 millimetres (0 invalid); labels single-channel uint8/uint16 PNG; pose is world_T_camera in the
optical frame (z forward, y down, identity extrinsics) with quaternion read by column name.

Config-level faults (out of scope) would use `D:\Hydra\config\datasets\habitat.yaml`: `frontend.objects.min_cluster_size` (40),
`frontend.objects.cluster_tolerance` (0.25), `backend.update_functors.rooms.room_finder.{min_dilation_m 0.5, max_dilation_m 1.2,
min_component_size, min_room_size, dilation_threshold_mode, plateau_ratio, min_lifetime_length_m, dilation_diff_threshold_m}`;
for places `frontend.freespace_places.gvd.{min_distance_m, max_distance_m}` and `...graph.{min_node_distance_m, compression_distance_m}`.

Limits: uint8/uint16 labels only (PNG); label faults are per-frame independent, so confusion flickers across frames
instead of being a consistent per-instance error.
