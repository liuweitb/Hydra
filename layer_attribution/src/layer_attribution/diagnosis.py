"""Layer scores from feature strengths, and the diagnosis rules (DAG-aware method and baselines).

A layer score is the k-th largest strength among the layer's usable features (k = 1 is the most sensitive, k > 1 asks for
several features to agree). A layer DEVIATES when its score is >= 1. A usable feature (finite on the benign runs) that is
undefined on a run counts as strength NAN_STRENGTH: the structure it measures is gone, which is itself a deviation.
Every rule returns a Diagnosis; an empty blame list means "no fault found" and counts as a miss when a query failed.
"""

from dataclasses import dataclass, field

import numpy as np

from layer_attribution import calibration as C
from layer_attribution import dag

LAYERS = dag.LAYERS
MODES = ("rb", "rf", "both")
NAN_STRENGTH = 1.0


@dataclass(frozen=True)
class ScoreConfig:
    mode: str = "rb"  # rb reference-based features, rf reference-free, both
    k: int = 2
    exclude_count_based: bool = False
    margin: float = C.MARGIN  # threshold = margin x largest benign deviation

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if self.margin <= 0:
            raise ValueError("margin must be positive")

    def label(self) -> str:
        return f"{self.mode}-k{self.k}{'-nocount' if self.exclude_count_based else ''}-m{self.margin:g}"


@dataclass
class Diagnosis:
    method: str
    blame: list  # ordered, blame[0] is the top-1; empty = no fault found
    scores: dict  # layer -> score
    deviating: list  # layers with score >= 1
    nan_counts: dict = field(default_factory=dict)  # layer -> usable features undefined on this run


def _strengths(strength_row, names) -> np.ndarray:
    vals = np.asarray([strength_row[n] for n in names if n in strength_row.index], dtype=float)
    return np.where(np.isfinite(vals), vals, NAN_STRENGTH), int((~np.isfinite(vals)).sum())


def layer_scores(strength_row, features_by_layer: dict, k: int = 2) -> dict:
    """{layer: k-th largest strength}. Undefined strengths count as NAN_STRENGTH; a layer with fewer than k usable
    features uses its smallest; 0.0 when the layer has no usable feature."""
    out = {}
    for layer in LAYERS:
        vals, _ = _strengths(strength_row, features_by_layer.get(layer, []))
        vals = np.sort(vals)[::-1]
        out[layer] = float(vals[min(k, len(vals)) - 1]) if len(vals) else 0.0
    return out


def nan_counts(strength_row, features_by_layer: dict) -> dict:
    return {layer: _strengths(strength_row, features_by_layer.get(layer, []))[1] for layer in LAYERS}


def _deviating(scores: dict) -> list:
    return [l for l in LAYERS if scores[l] >= 1.0]


def dag_rule(scores: dict, dag_name: str = "dataflow") -> Diagnosis:
    """Most upstream deviating layers (roots of the deviating set), strongest first."""
    dev = _deviating(scores)
    blame = sorted(dag.roots(dev, dag_name), key=lambda l: -scores[l])
    return Diagnosis(f"dag[{dag_name}]", blame, dict(scores), dev)


def naive_rule(scores: dict) -> Diagnosis:
    """Layer with the largest score, ignoring dependencies (abstains when nothing deviates)."""
    dev = _deviating(scores)
    blame = [max(LAYERS, key=lambda l: (scores[l], -LAYERS.index(l)))] if dev else []
    return Diagnosis("naive", blame, dict(scores), dev)


def downstream_rule(scores: dict, dag_name: str = "dataflow") -> Diagnosis:
    """Deepest deviating layer (where the failure is visible), ties by score."""
    dev = _deviating(scores)
    depth = {l: len(dag.ancestors(l, dag_name)) for l in LAYERS}
    blame = sorted(dev, key=lambda l: (-depth[l], -scores[l]))[:1]
    return Diagnosis("downstream", blame, dict(scores), dev)


# failure mode of a query -> layer where the failure is visible
SYMPTOM_LAYER = {"room_missing": "rooms", "wrong_room": "rooms", "wrong_label": "objects", "target_missing": "objects",
                 "wrong_position": "objects"}


def symptom_rule(mode_counts: dict) -> Diagnosis:
    """Query-only baseline: the layer holding most of the failed queries' symptoms (no features at all)."""
    tot = {l: 0.0 for l in LAYERS}
    for mode, n in mode_counts.items():
        if mode in SYMPTOM_LAYER:
            tot[SYMPTOM_LAYER[mode]] += float(n)
    n_all = sum(tot.values())
    scores = {l: (tot[l] / n_all if n_all else 0.0) for l in LAYERS}
    blame = [max(("objects", "rooms"), key=lambda l: (tot[l], l == "objects"))] if n_all else []
    return Diagnosis("symptom", blame, scores, [l for l in LAYERS if tot[l] > 0])


def diagnose_run(strength_row, cal: C.Calibration, cfg: ScoreConfig, dag_name: str = "dataflow") -> dict:
    """All feature-based rules for one run: {method name: Diagnosis}. `cal` must be fitted with cfg.margin."""
    feats = C.usable_features(cal, cfg.mode, cfg.exclude_count_based)
    scores = layer_scores(strength_row, feats, cfg.k)
    nans = nan_counts(strength_row, feats)
    out = {f"dag_{cfg.mode}": dag_rule(scores, dag_name), f"naive_{cfg.mode}": naive_rule(scores),
           f"downstream_{cfg.mode}": downstream_rule(scores, dag_name)}
    for d in out.values():
        d.nan_counts = nans
    return out
