#!/usr/bin/env python3
"""Cache the PLACES layer (distance per node, weighted edges) of Hydra runs for emulate_rooms.py.

usage: dump_places.py OUT_DIR RUN_DIR [RUN_DIR ...]      writes OUT_DIR/places_<run name>.json
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import emulate_rooms as er  # noqa: E402

out = pathlib.Path(sys.argv[1])
for run in sys.argv[2:]:
    dist, edges = er.load_places(run)
    er.save_places(dist, edges, out / f"places_{pathlib.Path(run).name}.json")
    print(run, len(dist), "places", len(edges), "edges", flush=True)
