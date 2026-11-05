"""Mesh <-> layer links of a reference run: which mesh vertices objects / 2D places / GVD places reference.

Reads only uh2_clean (reference, not test data). Output: json to argv[2].
"""
import collections
import json
import sys

import numpy as np
import spark_dsg as dsg


def label_hist(labels):
    c = collections.Counter(int(x) for x in labels)
    return dict(sorted(c.items()))


def main(root, out_path):
    res = {}
    gb = dsg.DynamicSceneGraph.load(f"{root}/backend/dsg_with_mesh.json")
    mesh = gb.mesh
    V = np.asarray(mesh.get_vertices())
    L = np.asarray(mesh.get_labels(), dtype=np.int64)
    res["backend_mesh"] = {
        "num_vertices": int(mesh.num_vertices()),
        "num_faces": int(mesh.num_faces()),
        "vertices_array_shape": list(V.shape),
        "label_histogram": label_hist(L),
    }
    nv = int(mesh.num_vertices())

    # objects -> mesh vertices
    tot = match = 0
    out_of_range = 0
    per_obj_frac = []
    for n in gb.get_layer("OBJECTS").nodes:
        a = n.attributes
        idx = np.asarray(list(a.mesh_connections), dtype=np.int64)
        if idx.size == 0:
            continue
        ok = idx < nv
        out_of_range += int((~ok).sum())
        idx = idx[ok]
        lab = L[idx]
        m = int((lab == int(a.semantic_label)).sum())
        tot += idx.size
        match += m
        per_obj_frac.append(m / idx.size)
    res["objects_to_mesh"] = {
        "num_connections": tot,
        "frac_vertex_label_equals_object_label": match / tot if tot else None,
        "min_per_object_frac": float(min(per_obj_frac)) if per_obj_frac else None,
        "out_of_range_indices": out_of_range,
    }

    # 2D places -> mesh vertices
    tot = 0
    lab_counter = collections.Counter()
    for n in gb.get_layer("MESH_PLACES").nodes:
        a = n.attributes
        idx = np.asarray(list(a.pcl_mesh_connections), dtype=np.int64)
        idx = idx[idx < nv]
        tot += idx.size
        lab_counter.update(int(x) for x in L[idx])
    res["mesh_places_to_mesh"] = {"num_connections": tot, "label_histogram_of_referenced_vertices": dict(sorted(lab_counter.items()))}

    # GVD places -> mesh vertices
    n_pcl = n_vox = n_def = 0
    nplaces = 0
    for n in gb.get_layer("PLACES").nodes:
        a = n.attributes
        nplaces += 1
        n_pcl += len(list(a.pcl_mesh_connections))
        n_vox += len(list(a.voxblox_mesh_connections))
        n_def += len(list(a.deformation_connections))
    res["places_to_mesh"] = {
        "num_places": nplaces,
        "total_pcl_mesh_connections": n_pcl,
        "total_voxblox_mesh_connections": n_vox,
        "total_deformation_connections": n_def,
    }

    # frontend mesh vs backend mesh
    gf = dsg.DynamicSceneGraph.load(f"{root}/frontend/dsg_with_mesh.json")
    Vf = np.asarray(gf.mesh.get_vertices())
    Lf = np.asarray(gf.mesh.get_labels(), dtype=np.int64)
    same_shape = Vf.shape == V.shape
    res["frontend_vs_backend_mesh"] = {
        "same_shape": bool(same_shape),
        "max_abs_vertex_diff": float(np.abs(Vf - V).max()) if same_shape else None,
        "labels_equal": bool(same_shape and np.array_equal(Lf, L)),
        "frontend_num_vertices": int(gf.mesh.num_vertices()),
    }
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print(json.dumps(res, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
