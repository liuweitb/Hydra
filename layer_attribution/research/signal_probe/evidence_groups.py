"""Evidence-group assignment of every feature in features_proto.REGISTRY (research code).

Groups (DAG layers): MESH = geometry only; OBJECTS = object-node features plus the mesh per-vertex-label features
(tag mesh_sem: semantic labels are the object layer's input, so a label fault must not look like a mesh fault);
PLACES = GVD place and surface-place features; ROOMS = room features.
Not layers, never used as per-layer evidence in the DAG rule: INPUT_POSE (trajectory / frame timing),
CROSS (links between two layers), QUERY (outcome of the test queries).
tag: mesh_sem | process | surface_place | cross | outcome | "" ; reads: other layers whose content the feature reads
(a feature can move because that other layer changed, not because its own group did).
"""

from __future__ import annotations

GROUPS = ("MESH", "OBJECTS", "PLACES", "ROOMS")
DAG_ORDER = ("MESH", "OBJECTS", "PLACES", "ROOMS")  # MESH -> {OBJECTS, PLACES} -> ROOMS; OBJECTS and PLACES are siblings
UPSTREAM = {"MESH": (), "OBJECTS": ("MESH",), "PLACES": ("MESH",), "ROOMS": ("PLACES", "MESH")}

# name -> (group, tag, reads)
GROUP: dict[str, tuple[str, str, str]] = {}


def _put(group, names, tag="", reads=""):
    for n in names.split():
        GROUP[n] = (group, tag, reads)


# ---- MESH (geometry only)
_put("MESH", "mesh_vertex_count_ratio mesh_area_ratio mesh_extra_mean mesh_missing_mean mesh_extra_frac_gt10cm "
             "mesh_missing_frac_gt10cm mesh_centroid_shift mesh_roughness_median mesh_roughness_p90 wall_sheet_sd_median "
             "mesh_open_edge_frac")
_put("MESH", "floor_z_robust_sd floor_z_time_range floor_time_structure", reads="mesh_sem")  # floor level picked by the floor label
_put("MESH", "mesh_revisit_offset_median mesh_revisit_offset_median_k96")

# ---- OBJECTS: mesh per-vertex labels
_put("OBJECTS", "mesh_label_hist_js mesh_objclass_vertex_ratio mesh_label_edge_disagree_frac mesh_objclass_knn_purity "
                "mesh_void_frac", tag="mesh_sem")
_put("OBJECTS", "mesh_label_disagree_matched", tag="mesh_sem", reads="mesh_geom")  # vertex matching uses positions

# ---- OBJECTS: object nodes
_put("OBJECTS", "obj_count_ratio obj_recall_pos obj_precision_pos obj_label_agree_matched obj_recall_same_label "
                "obj_pos_err_median obj_bbox_iou_median obj_bbox_size_err_median obj_extra_near_ref_frac obj_class_hist_js "
                "obj_dup_same_class_frac_050 obj_dup_same_class_frac_100 obj_overlap_diffclass_frac "
                "obj_overlap_sameclass_frac obj_size_prior_violation_frac obj_tiny_frac obj_logvol_mad_within_class "
                "obj_support_median")
_put("OBJECTS", "obj_mesh_label_purity_median obj_mesh_label_purity_low_frac", reads="mesh_sem")
_put("OBJECTS", "obj_nearest_place_dist_p90 obj_orphan_frac obj_place_intrusion_frac", reads="places")
_put("OBJECTS", "obj_floor_gap_p90", reads="mesh_geom")
_put("OBJECTS", "obj_removed_per_final obj_merged_per_final", tag="process")

# ---- PLACES: GVD places and surface places
_put("PLACES", "place_count_ratio place_coverage_050 place_spurious_050 place_chamfer_mean place_freespace_iou "
               "place_radius_err_median place_radius_mean_ratio place_edge_count_ratio place_edge_gap_frac place_components "
               "place_freevol_ratio place_freevol_m3 "
               "place_isolated_frac place_degree_mean place_radius_mean place_small_radius_frac")
_put("PLACES", "meshplace_count_ratio", tag="surface_place")
_put("PLACES", "place_mesh_intrusion_frac place_radius_mesh_ratio_median place_edge_wall_cross_frac", reads="mesh_geom")
_put("PLACES", "place_orphan_frac", reads="rooms")
_put("PLACES", "place_removed_per_final place_merged_per_final", tag="process")

# ---- ROOMS
_put("ROOMS", "room_count_ratio room_match_frac room_centroid_err_median room_empty_count room_min_share room_footprint_min "
              "room_footprint_max room_components_mean room_boundary_mixing room_degree_mean room_elongation_max",
     reads="places")
_put("ROOMS", "room_assignment_agreement room_partition_ari", reads="places")
_put("ROOMS", "room_without_objects_frac", reads="places;objects")

# ---- not layers
_put("CROSS", "edge_obj_room_agreement", tag="cross", reads="objects;places;rooms")
_put("CROSS", "query_ambiguity_frac", tag="cross", reads="objects;places;rooms")
_put("CROSS", "obj_parent_dist_median", tag="cross", reads="objects;places")
_put("QUERY", "query_success_rate query_fail_room_missing query_fail_target_missing query_fail_wrong_object",
     tag="outcome", reads="objects;places;rooms")
_put("INPUT_POSE", "traj_ate_rmse traj_keyframe_ratio traj_dt_p95_over_median traj_step_max_over_median traj_accel_rms "
                   "traj_yaw_rate_p95")
_put("INPUT_POSE", "traj_mesh_collision_frac", reads="mesh_geom")


def check(registry) -> None:
    missing = [n for n in registry if n not in GROUP]
    extra = [n for n in GROUP if n not in registry]
    assert not missing and not extra, f"unmapped: {missing}; unknown: {extra}"
