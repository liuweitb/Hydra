#!/usr/bin/env bash
# which json fields differ between two runs with the same effective config (aggregated by field pattern)
A=${1:-$HOME/runs/uh2/tmp_dk_base/backend/dsg.json}
B=${2:-$HOME/runs/uh2/tmp_dk_mis/backend/dsg.json}
python3 - "$A" "$B" <<'PY'
import collections, json, re, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
cnt = collections.Counter()
def walk(x, y, path=""):
    if type(x) != type(y):
        cnt[re.sub(r"\[\d+\]", "[]", path) + " (type)"] += 1; return
    if isinstance(x, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x or k not in y:
                cnt[re.sub(r"\[\d+\]", "[]", path + "/" + k) + " (key)"] += 1; continue
            walk(x[k], y[k], path + "/" + k)
    elif isinstance(x, list):
        if len(x) != len(y):
            cnt[re.sub(r"\[\d+\]", "[]", path) + " (len)"] += 1; return
        for i, (p, q) in enumerate(zip(x, y)):
            walk(p, q, f"{path}[{i}]")
    elif x != y:
        cnt[re.sub(r"\[\d+\]", "[]", path)] += 1
walk(a, b)
for k, v in cnt.most_common():
    print(v, k)
print("distinct field patterns differing:", len(cnt))
PY
