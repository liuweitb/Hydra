#!/usr/bin/env python3
"""Outcome intervals of one room_finder parameter on a cached places graph: where does the room count change?

usage: scan_intervals.py CACHE_JSON KEY LO HI STEP [int]
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import emulate_rooms as er  # noqa: E402

dist, edges = er.load_cached_places(sys.argv[1])
key, lo, hi, step = sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
as_int = len(sys.argv) > 6 and sys.argv[6] == "int"
vals, v = [], lo
while v <= hi + 1e-9:
    vals.append(round(v, 6))
    v += step
prev, start = None, None
for v in vals + [None]:
    cur = None
    if v is not None:
        r = er.find_rooms(dist, edges, **{key: int(round(v)) if as_int else v})
        cur = (r["n_rooms"], tuple(r["sizes"]), r["threshold"])
    if cur != prev:
        if prev is not None:
            print(f"{key} in [{start}, {last}]: rooms={prev[0]} sizes={list(prev[1])} threshold={prev[2]}")
        start, prev = v, cur
    last = v
