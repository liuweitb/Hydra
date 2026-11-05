#!/usr/bin/env python3
"""Offline re-implementation of Hydra's RoomFinder (PLATEAU mode, NEIGHBORS clustering) on a saved places graph.

Rooms are a pure function of the final places layer and the room_finder config, so one saved graph answers
"what would rooms look like under parameter X" without a Hydra run. Validated against real Hydra output in findings.md.
Edge ties in the NEIGHBORS flood fill break differently from C++ (priority_queue), so sizes can differ at room borders.

usage: emulate_rooms.py RUN_DIR [--grid]       RUN_DIR = Hydra run dir (reads backend/dsg_with_mesh.json)
"""

import heapq
import json
import pathlib
import sys

DEFAULT = dict(min_dilation_m=0.5, max_dilation_m=1.2, min_component_size=10, min_room_size=10, plateau_ratio=0.15)


def load_places(run):
    """(node id -> distance, [(u, v, weight)]) of the PLACES layer; needs spark_dsg (WSL venv)."""
    import spark_dsg as dsg

    g = dsg.DynamicSceneGraph.load(str(pathlib.Path(run) / "backend" / "dsg_with_mesh.json"))
    layer = g.get_layer(dsg.DsgLayers.PLACES)
    dist = {n.id.value: float(n.attributes.distance) for n in layer.nodes}
    edges = [(e.source, e.target, float(e.info.weight)) for e in layer.edges if e.source in dist and e.target in dist]
    return dist, edges


def save_places(dist, edges, path):
    """Cache of load_places() so the emulator runs without spark_dsg."""
    pathlib.Path(path).write_text(json.dumps({"dist": {str(k): v for k, v in dist.items()}, "edges": edges}))


def load_cached_places(path):
    d = json.loads(pathlib.Path(path).read_text())
    return {int(k): v for k, v in d["dist"].items()}, [tuple(e) for e in d["edges"]]


class UF:
    def __init__(self, ids):
        self.p = {i: i for i in ids}
        self.size = {i: 1 for i in ids}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return None
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.p[b] = a
        self.size[a] += self.size[b]
        return a, b


def filtration(dist, edges, min_component_size):
    """[(distance, n_components>=min size)] ascending distance, one entry per edge (diff threshold -1 keeps all)."""
    uf = UF(dist)
    count = len(dist) if min_component_size <= 1 else 0
    out = []
    for u, v, w in sorted(edges, key=lambda e: -e[2]):
        sa, sb = uf.size[uf.find(u)], uf.size[uf.find(v)]
        if uf.union(u, v) is not None:
            m = min_component_size
            count += (sa + sb >= m) - (sa >= m) - (sb >= m)
        out.append((w, count))
    return out[::-1]


def trimmed(fil, lo, hi):
    start, end = 0, len(fil)
    for i, (d, _) in enumerate(fil):
        if d < lo:
            start = i + 1
        if d < hi:
            end = i
    return start, end


def best_plateau(fil, ratio, start, end):
    if start >= len(fil):
        return None
    end = min(end, len(fil))
    if start == end:
        return None
    seqs = [[start, start]]
    vals = [fil[start][1]]
    for i in range(start + 1, end):
        if fil[i][1] != vals[-1]:
            vals.append(fil[i][1])
            seqs.append([i, i])
        else:
            seqs[-1][1] = i
    life = [fil[e][0] - fil[s][0] for s, e in seqs]
    thr = ratio * max(life)
    best, best_c = 0, 0
    for i, lt in enumerate(life):
        if lt < thr:
            continue
        c = fil[seqs[i][0]][1]
        if c > best_c:
            best, best_c = i, c
    return fil[seqs[best][0]]


def components(dist, edges, thr, min_size):
    ok = {n for n, d in dist.items() if d > thr}
    uf = UF(ok)
    for u, v, w in edges:
        if w > thr and u in ok and v in ok:
            uf.union(u, v)
    groups = {}
    for n in ok:
        groups.setdefault(uf.find(n), []).append(n)
    return [g for g in groups.values() if len(g) >= min_size]


def neighbor_clusters(dist, edges, comps):
    adj = {n: [] for n in dist}
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    label = {n: i for i, c in enumerate(comps) for n in c}
    heap, k = [], 0
    for n in dist:
        if n in label:
            continue
        for s, w in adj[n]:
            if s in label:
                heapq.heappush(heap, (-w, k, n, label[s]))
                k += 1
    while heap:
        _, _, n, lab = heapq.heappop(heap)
        if n in label:
            continue
        label[n] = lab
        for s, w in adj[n]:
            if s not in label:
                heapq.heappush(heap, (-w, k, s, lab))
                k += 1
    sizes = {}
    for lab in label.values():
        sizes[lab] = sizes.get(lab, 0) + 1
    return sizes, label


def find_rooms(dist, edges, return_labels=False, **over):
    """Rooms for the given room_finder overrides. return_labels adds {place id: cluster index} of places in kept rooms."""
    p = {**DEFAULT, **over}
    fil = filtration(dist, edges, p["min_component_size"])
    start, end = trimmed(fil, p["min_dilation_m"], p["max_dilation_m"])
    if start >= len(fil):
        return dict(n_rooms=0, sizes=[], threshold=None, components=None)
    cand = best_plateau(fil, p["plateau_ratio"], start, end)
    if cand is None:
        return dict(n_rooms=0, sizes=[], threshold=None, components=None)
    comps = components(dist, edges, cand[0], p["min_component_size"])
    if not comps:
        return dict(n_rooms=0, sizes=[], threshold=round(cand[0], 6), components=cand[1])
    cluster_sizes, label = neighbor_clusters(dist, edges, comps)
    kept = {c for c, s in cluster_sizes.items() if s >= p["min_room_size"]}
    sizes = sorted((cluster_sizes[c] for c in kept), reverse=True)
    out = dict(n_rooms=len(sizes), sizes=sizes, threshold=round(cand[0], 6), components=cand[1])
    if return_labels:
        out["labels"] = {n: c for n, c in label.items() if c in kept}
    return out


def main():
    run = sys.argv[1]
    dist, edges = load_places(run)
    print("places", len(dist), "edges", len(edges))
    print("default:", json.dumps(find_rooms(dist, edges)))
    if "--grid" in sys.argv:
        from layer_attribution.pipeline import config_faults as cf

        grid = {}
        for name, f in cf.FAULTS.items():
            if f.layer != "rooms":
                continue
            grid[name] = {}
            for s in [i / 10 for i in range(11)]:
                over = {k.rsplit(".", 1)[1]: v for k, v in cf.values_at(name, s).items()}
                grid[name][s] = find_rooms(dist, edges, **over)
                print(name, s, over, json.dumps(grid[name][s]))
        out = pathlib.Path(run).name + "_room_grid.json"
        pathlib.Path(sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else out).write_text(json.dumps(grid, indent=1))


if __name__ == "__main__":
    main()
