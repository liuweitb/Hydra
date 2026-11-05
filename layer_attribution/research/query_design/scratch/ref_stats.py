import sys, collections, itertools
import numpy as np
sys.path.insert(0, "D:/Hydra/layer_attribution/src")
from layer_attribution.queries import SceneView, node_name
V = "D:/hydra_work/results/query_design/views/"
ref = SceneView.load(V + "uh2_clean.npz")
names = ref.label_names
print("objects", ref.n_obj, "places", ref.n_place, "rooms", ref.n_room)
rooms_of = [ref.rooms_of_object(i) for i in range(ref.n_obj)]
print("objects with >=1 room:", sum(len(r) >= 1 for r in rooms_of), " with >1 distinct rooms:", sum(len(r) > 1 for r in rooms_of))
print("n parents hist", collections.Counter(len(ref.parents(i)) for i in range(ref.n_obj)))
# (class, room) multiplicity
pairs = collections.defaultdict(list)
for i in range(ref.n_obj):
    for r in rooms_of[i]:
        pairs[(int(ref.obj_label[i]), int(r))].append(i)
print("(class,room) pairs:", len(pairs))
mult = collections.Counter(len(v) for v in pairs.values())
print("pair multiplicity hist {k objects: n pairs}", dict(sorted(mult.items())))
print("objects whose (class,room) pair has exactly 1 object:", sum(1 for v in pairs.values() if len(v) == 1))
print("objects in ambiguous pairs:", sum(len(v) for v in pairs.values() if len(v) > 1))
# per class
for c in sorted(set(ref.obj_label)):
    n = int((ref.obj_label == c).sum())
    prs = [(k, v) for k, v in pairs.items() if k[0] == c]
    print(f"class {c} {names.get(int(c))}: {n} objects, {len(prs)} rooms, unique pairs {sum(len(v)==1 for k,v in prs)}")
# min distance between same-class objects (3D and 2D), within the same room and overall
d3, d2, same_room = [], [], []
for i, j in itertools.combinations(range(ref.n_obj), 2):
    if ref.obj_label[i] != ref.obj_label[j]:
        continue
    dd = ref.obj_pos[i] - ref.obj_pos[j]
    d3.append(np.linalg.norm(dd)); d2.append(np.linalg.norm(dd[:2]))
    same_room.append(bool(set(rooms_of[i]) & set(rooms_of[j])))
d3, d2, same_room = map(np.array, (d3, d2, same_room))
print("same-class pairs total", len(d3), "same room", int(same_room.sum()))
for thr in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
    print(f"  same-class same-room pairs closer than {thr} m: 3D {(d3[same_room] < thr).sum()}  2D {(d2[same_room] < thr).sum()}   (any room: 3D {(d3<thr).sum()})")
# nearest same-class neighbour of each object
nn = []
for i in range(ref.n_obj):
    m = (ref.obj_label == ref.obj_label[i]); m[i] = False
    nn.append(np.linalg.norm(ref.obj_pos[m] - ref.obj_pos[i], axis=1).min() if m.any() else np.inf)
nn = np.array(nn)
print("nearest same-class neighbour distance (3D) quantiles", np.round(np.quantile(nn, [0, .05, .1, .25, .5, .75, 1]), 2))
# ref room sizes
for r in range(ref.n_room):
    ch = np.where(ref.place_room == r)[0]
    print(node_name(ref.room_id[r]), "pos", np.round(ref.room_pos[r], 1), "places", len(ch), "objects", sum(r in rooms_of[i] for i in range(ref.n_obj)))
