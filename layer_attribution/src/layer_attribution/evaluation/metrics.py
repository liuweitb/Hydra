"""Scoring and aggregation of diagnosis records (numpy only, no scene graphs).

A record is one (run, method) decision, a plain dict:
  run, fault, severity, seed, split, kind ('sensor' | 'config'),
  accept   layers that count as correct (the injected fault's entry set),
  masked   True when the fault is present but the queries still succeed (not scored),
  failure_rate, method, blame (ordered layer list, blame[0] is the top-1), scores {layer: float}, deviating [layer]
"""

from collections import defaultdict

import numpy as np

LAYERS = ("mesh", "objects", "places", "rooms")


def is_correct(rec, strict=False):
    """Lenient: top-1 is in the accept set. Strict: the blame SET equals the accept set."""
    blame, accept = list(rec["blame"]), set(rec["accept"])
    if not blame:
        return False
    return set(blame) == accept if strict else blame[0] in accept


def scored(records, method=None, include_masked=False):
    """Records that are scored: unmasked ones by default (a failed query exists), all fault runs with include_masked."""
    return [r for r in records if (include_masked or not r["masked"]) and (method is None or r["method"] == method)]


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    """(mean, lo, hi), percentile bootstrap over the values; NaNs for an empty input."""
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def _group(records, by):
    key = by if callable(by) else (lambda r: r[by])
    groups = defaultdict(list)
    for r in records:
        groups[key(r)].append(r)
    return dict(sorted(groups.items(), key=lambda kv: str(kv[0])))


def accuracy(records, method, by="fault", strict=False, n_boot=2000, seed=0, include_masked=False):
    """{group: {acc, lo, hi, n}, 'macro': ...}. The macro entry averages the per-group accuracies, its interval
    resamples every group separately, so a group with many runs cannot dominate."""
    rows = scored(records, method, include_masked)
    out, per_group = {}, {}
    for g, recs in _group(rows, by).items():
        hits = [float(is_correct(r, strict)) for r in recs]
        acc, lo, hi = bootstrap_ci(hits, n_boot, seed=seed)
        out[g] = dict(acc=acc, lo=lo, hi=hi, n=len(hits))
        per_group[g] = np.asarray(hits)
    if per_group:
        rng = np.random.default_rng(seed)
        boots = np.stack([h[rng.integers(0, h.size, size=(n_boot, h.size))].mean(axis=1) for h in per_group.values()])
        macro = boots.mean(axis=0)
        out["macro"] = dict(acc=float(np.mean([h.mean() for h in per_group.values()])),
                            lo=float(np.quantile(macro, 0.025)), hi=float(np.quantile(macro, 0.975)),
                            n=int(sum(h.size for h in per_group.values())))
    return out


def confusion(records, method, layers=LAYERS):
    """rows: accept set as 'a+b'; columns: blamed top-1 layer or 'none'. Values are counts."""
    cols = list(layers) + ["none"]
    table = {}
    for r in scored(records, method):
        row = "+".join(sorted(r["accept"]))
        table.setdefault(row, dict.fromkeys(cols, 0))
        table[row][r["blame"][0] if r["blame"] else "none"] += 1
    return dict(sorted(table.items()))


def masked_table(records, method=None):
    """Per (fault, severity): runs, mean query failure rate, fraction masked. One row per run when method is None
    would repeat runs, so only the first method's records are used."""
    recs = records if method is None else [r for r in records if r["method"] == method]
    seen, uniq = set(), []
    for r in recs:
        if r["run"] not in seen:
            seen.add(r["run"])
            uniq.append(r)
    rows = {}
    for (fault, sev), grp in _group(uniq, lambda r: (r["fault"], r["severity"])).items():
        rows[(fault, sev)] = dict(n=len(grp), failure_rate=float(np.mean([r["failure_rate"] for r in grp])),
                                  masked=float(np.mean([r["masked"] for r in grp])))
    return rows


def propagation_table(records, method, layers=LAYERS):
    """Per fault: fraction of runs in which each layer deviates (all runs, masked ones included)."""
    out = {}
    for fault, grp in _group([r for r in records if r["method"] == method], "fault").items():
        out[fault] = {layer: float(np.mean([layer in r["deviating"] for r in grp])) for layer in layers}
    return out


def expected_random(records, layers=LAYERS):
    """Exact expected lenient accuracy of a uniform random top-1 over the layers."""
    rows = scored(records)
    seen, vals = set(), []
    for r in rows:
        if r["run"] not in seen:
            seen.add(r["run"])
            vals.append(len(set(r["accept"]) & set(layers)) / len(layers))
    return float(np.mean(vals)) if vals else float("nan")


def fit_prior(dev_records, layers=LAYERS):
    """The single layer with the highest macro (over fault types) lenient accuracy on dev records."""
    seen, rows = set(), []
    for r in scored(dev_records):
        if r["run"] not in seen:
            seen.add(r["run"])
            rows.append(r)
    best, best_acc = None, -1.0
    for layer in layers:
        per_fault = [np.mean([layer in r["accept"] for r in grp]) for grp in _group(rows, "fault").values()]
        acc = float(np.mean(per_fault)) if per_fault else -1.0
        if acc > best_acc:
            best, best_acc = layer, acc
    return best


def prior_records(records, layer, method="prior"):
    """Records of the constant 'always blame `layer`' baseline for the runs in `records`."""
    seen, out = set(), []
    for r in records:
        if r["run"] in seen:
            continue
        seen.add(r["run"])
        out.append(dict(r, method=method, blame=[layer], scores={layer: 1.0}, deviating=[layer]))
    return out


MIN_CI_N = 5  # below this a bootstrap interval says nothing; report the point estimate and n only


def permutation_null(records, method, n_perm=2000, seed=0, include_masked=False):
    """Macro accuracy of `method` against the distribution obtained when the fault labels (with their accept sets) are
    shuffled over the scored runs: the like-for-like chance level of a method that never blames some layers.
    Returns {real, null_mean, null_lo, null_hi, p_ge (one-sided, add-one), n}."""
    rows = scored(records, method, include_masked)
    n = len(rows)
    if n < 2:
        return dict(real=float("nan"), null_mean=float("nan"), null_lo=float("nan"), null_hi=float("nan"), p_ge=float("nan"), n=n)
    top = [r["blame"][0] if r["blame"] else None for r in rows]
    accepts = [set(r["accept"]) for r in rows]
    faults = sorted({r["fault"] for r in rows})
    fidx = np.array([faults.index(r["fault"]) for r in rows])
    hit = np.array([[b is not None and b in accepts[j] for j in range(n)] for b in top], dtype=float)  # hit[i, j]: run i against label j

    def macro(perm):
        h, lab = hit[np.arange(n), perm], fidx[perm]
        cnt = np.bincount(lab, minlength=len(faults))
        hits = np.bincount(lab, weights=h, minlength=len(faults))
        return float(np.mean(hits[cnt > 0] / cnt[cnt > 0]))

    real = macro(np.arange(n))
    rng = np.random.default_rng(seed)
    null = np.array([macro(rng.permutation(n)) for _ in range(n_perm)])
    return dict(real=real, null_mean=float(null.mean()), null_lo=float(np.quantile(null, 0.025)),
                null_hi=float(np.quantile(null, 0.975)), p_ge=float((1 + np.sum(null >= real - 1e-12)) / (1 + n_perm)), n=n)


def named_member(records, method, include_masked=False):
    """For faults with an accept set of several layers: which layer the top-1 names. {fault: {layer or 'none': count}}."""
    out = {}
    for r in scored(records, method, include_masked):
        if len(r["accept"]) < 2:
            continue
        key = r["blame"][0] if r["blame"] else "none"
        out.setdefault(r["fault"], {}).setdefault(key, 0)
        out[r["fault"]][key] += 1
    return dict(sorted(out.items()))
