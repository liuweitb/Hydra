#!/usr/bin/env python3
"""Summarise room detection per input step from a Hydra INFO log (no Hydra import needed).

usage: scan_log.py LOG [every_n_steps]
"""

import re
import sys

RE_INPUT = re.compile(r"^I\d{8} (\d+):(\d+):([\d.]+).*Got input @ (\d+) \[ns\]")
RE_NODES = re.compile(r"Detecting rooms for (\d+) nodes")
RE_BOUNDS = re.compile(r"Bounds: \[([\d.e+-]+), ([\d.e+-]+)\],\s+window: \[(\d+), (\d+)\]")
RE_BEST = re.compile(r"Best threshold: ([\d.e+-]+) \((\d+) components\)")
RE_NOROOM = re.compile(r"No rooms found")


def main():
    path = sys.argv[1]
    every = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    step = -1
    rec = {}
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            m = RE_INPUT.search(line)
            if m:
                if rec:
                    rows.append((step, rec))
                step += 1
                rec = {"t_s": round(int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)), 1)}
                continue
            m = RE_NODES.search(line)
            if m:
                rec["places"] = int(m.group(1))
                continue
            m = RE_BOUNDS.search(line)
            if m:
                rec["window"] = (int(m.group(3)), int(m.group(4)))
                continue
            m = RE_BEST.search(line)
            if m:
                rec["thr"] = float(m.group(1))
                rec["comp"] = int(m.group(2))
                continue
            if RE_NOROOM.search(line):
                rec["noroom"] = True
    if rec:
        rows.append((step, rec))
    print("steps", step + 1)
    for s, r in rows:
        if s % every == 0 or s == rows[-1][0]:
            print(s, r)


if __name__ == "__main__":
    main()
