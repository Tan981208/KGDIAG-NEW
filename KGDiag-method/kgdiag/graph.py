"""Build directed KG tensors from documented edges and normal-data Pearson links.

Matrix rows are receivers (dst); columns are message sources (src). Graph
construction accepts normal Fit and normal Validation only. It never reads
test observations or labels. Relation labels are kept distinct from signed
statistical associations and from nonnegative reference weights.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np


ENGINEERING_TYPES = (
    "physical_flow", "control_logic", "feedback_signal", "causal_lag"
)
SELF = "self"
AUXILIARY = "implicit_learned"


def _nodes(schema: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> list[dict]:
    raw = schema["nodes"] if isinstance(schema, Mapping) else schema
    nodes = [dict(n) for n in raw]
    if not nodes:
        raise ValueError("The node registry must not be empty.")
    names = [n.get("name") for n in nodes]
    if any(not isinstance(n, str) or not n.strip() for n in names):
        raise ValueError("Each node requires a nonempty string name.")
    if len(set(names)) != len(names):
        raise ValueError("Node names must be unique.")
    for n in nodes:
        if n.get("kind") not in {"continuous", "categorical"}:
            raise ValueError(f"Declare continuous/categorical kind for {n['name']}.")
    return nodes


def _matrix(value: Any, n: int, name: str) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != n or a.shape[0] == 0:
        raise ValueError(f"{name} must have nonempty shape [time, {n}].")
    return a


def _integer(config: Mapping, name: str, minimum: int) -> int:
    v = config[name]
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(v)


def _threshold(config: Mapping, name: str, allow_zero: bool) -> float:
    v = float(config[name])
    if not math.isfinite(v) or not (0 <= v <= 1) or (not allow_zero and v == 0):
        raise ValueError(f"{name} must be {'in [0, 1]' if allow_zero else 'in (0, 1]' }.")
    return v


def _segments(length: int, count: int, supplied: Sequence[slice] | None) -> list[slice]:
    if supplied is None:
        if count > length:
            raise ValueError("fit_segment_count exceeds the number of Fit rows.")
        boundaries = np.linspace(0, length, count + 1, dtype=int)
        return [slice(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:])]
    if not supplied:
        raise ValueError("fit_segments must contain at least one nonempty slice.")
    out = []
    previous_stop = 0
    for s in supplied:
        if not isinstance(s, slice) or s.step not in (None, 1):
            raise ValueError("Fit segments must be chronological unit-step slices.")
        start, stop = s.start, s.stop
        if start is None or stop is None or not isinstance(start, (int, np.integer)) or not isinstance(stop, (int, np.integer)):
            raise ValueError("Each Fit segment needs explicit integer start and stop.")
        if not (0 <= start < stop <= length) or start < previous_stop:
            raise ValueError("Fit segments must be ordered, disjoint, nonempty, and within Fit.")
        out.append(slice(int(start), int(stop)))
        previous_stop = stop
    return out


def _pearson(x: np.ndarray, y: np.ndarray, min_observations: int) -> float | None:
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < min_observations:
        return None
    x, y = x[valid], y[valid]
    # Rescaling before centering avoids overflow without changing Pearson rho.
    sx, sy = float(np.max(np.abs(x))), float(np.max(np.abs(y)))
    if sx == 0 or sy == 0:
        return None
    x, y = x / sx, y / sy
    x, y = x - x.mean(), y - y.mean()
    nx, ny = float(np.linalg.norm(x)), float(np.linalg.norm(y))
    if nx == 0 or ny == 0 or not math.isfinite(nx + ny):
        return None
    r = float(np.dot(x / nx, y / ny))
    return float(np.clip(r, -1.0, 1.0)) if math.isfinite(r) else None


def _provenance(edge: Mapping) -> dict:
    fields = ("source_document", "source_location", "evidence_status")
    for name in fields:
        if not isinstance(edge.get(name), str) or not edge[name].strip():
            raise ValueError(f"Engineering edges require a nonempty {name}.")
    return {name: edge[name].strip() for name in fields}


def build_graph(
    schema: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    fit_values: np.ndarray,
    validation_values: np.ndarray,
    engineering_edges: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    fit_segments: Sequence[slice] | None = None,
) -> dict[str, Any]:
    """Create graph tensors in registry order, with no test-data dependency.

    Required config fields are ``top_k``, ``min_pair_observations``,
    ``min_abs_correlation``, ``min_segment_abs_correlation``,
    ``fit_segment_count``, ``engineering_weight`` and ``self_weight``.
    Both fixed weights must equal one. Optional ``additional_relation_types``
    declares supported engineering labels beyond the four canonical types.

    Pearson candidates require finite paired samples, nonconstant values,
    sufficient absolute correlation in full Fit, and matching signs plus
    minimum magnitude in every Fit segment and in normal Validation.
    The caller supplies explicit Fit segments when recording boundaries must
    be respected; otherwise the Fit array is split into chronological blocks.
    Every supplied array must contain only the declared normal partition.

    Duplicate engineering edges merge provenance. Conflicting relation labels
    or attributes on the same ordered pair raise an error. An auxiliary edge
    never overwrites an engineering edge, but a statistical reverse-direction
    edge is allowed and remains ``implicit_learned``. Lag attributes are
    retained as metadata; they do not cause delayed reads in this model.

    Returns NumPy arrays ``support`` [N,N] bool, ``weights`` [N,N] float64,
    ``prior`` [N,N] float64, ``relation_ids`` [N,N] int64 (-1 off support),
    plus ``relation_names``, ``node_names`` and JSON-compatible edge metadata.
    """
    nodes = _nodes(schema)
    n = len(nodes)
    names = [v["name"] for v in nodes]
    index = {name: i for i, name in enumerate(names)}
    fit = _matrix(fit_values, n, "fit_values")
    validation = _matrix(validation_values, n, "validation_values")
    k = _integer(config, "top_k", 0)
    min_obs = _integer(config, "min_pair_observations", 2)
    full_min = _threshold(config, "min_abs_correlation", allow_zero=False)
    part_min = _threshold(config, "min_segment_abs_correlation", allow_zero=True)
    count = _integer(config, "fit_segment_count", 1)
    segments = _segments(len(fit), count, fit_segments)
    for key in ("engineering_weight", "self_weight"):
        if float(config[key]) != 1.0:
            raise ValueError(f"{key} must equal 1 for this method definition.")
    extra = config.get("additional_relation_types", [])
    if not isinstance(extra, (list, tuple)) or any(not isinstance(r, str) or not r.strip() for r in extra):
        raise ValueError("additional_relation_types must be a list of nonempty names.")
    if any(r in {SELF, AUXILIARY} for r in extra):
        raise ValueError("self and implicit_learned are reserved computational relation types.")
    engineering_types = list(dict.fromkeys([*ENGINEERING_TYPES, *extra]))
    relation_names = [SELF, *engineering_types, AUXILIARY]
    rel_index = {r: i for i, r in enumerate(relation_names)}
    edges: dict[tuple[int, int], dict] = {}

    for raw in engineering_edges:
        e = dict(raw)
        src, dst, relation = e.get("src"), e.get("dst"), e.get("relation_type")
        if src not in index or dst not in index:
            raise ValueError(f"Unknown engineering endpoint: {src!r} -> {dst!r}.")
        if src == dst:
            raise ValueError("Engineering self edges are not accepted; computational self edges are added once.")
        if relation not in engineering_types:
            raise ValueError(f"Undeclared engineering relation {relation!r}.")
        if "prior_strength" in e and float(e["prior_strength"]) != 1.0:
            raise ValueError("Engineering prior_strength must be 1; confidence is not a message weight.")
        source = _provenance(e)
        attributes = {key: e[key] for key in ("lag_seconds", "applicable_condition", "association_sign", "dataset_version") if key in e}
        if "lag_seconds" in attributes:
            lag = float(attributes["lag_seconds"])
            if not math.isfinite(lag) or lag < 0:
                raise ValueError("lag_seconds must be finite and nonnegative.")
        pair = (index[dst], index[src])
        if pair in edges:
            previous = edges[pair]
            if previous["relation_type"] != relation or previous["attributes"] != attributes:
                raise ValueError(f"Conflicting engineering labels/attributes for {src} -> {dst}.")
            if source not in previous["provenance"]:
                previous["provenance"].append(source)
            continue
        edges[pair] = dict(src=src, dst=dst, relation_type=relation,
                           prior_strength=1.0, origin="engineering",
                           provenance=[source], attributes=attributes)

    continuous = [i for i, v in enumerate(nodes) if v["kind"] == "continuous"]
    pair_statistics: dict[tuple[int, int], dict | None] = {}
    if k:
        for offset, i in enumerate(continuous):
            for j in continuous[offset + 1:]:
                r = _pearson(fit[:, i], fit[:, j], min_obs)
                stats = None
                if r is not None and abs(r) >= full_min:
                    rs = [_pearson(fit[s, i], fit[s, j], min_obs) for s in segments]
                    rv = _pearson(validation[:, i], validation[:, j], min_obs)
                    if all(q is not None and abs(q) >= part_min and q * r > 0 for q in [*rs, rv]):
                        stats = {"fit_correlation": r, "fit_segment_correlations": rs,
                                 "validation_correlation": rv}
                pair_statistics[(i, j)] = stats

        for dst in continuous:
            candidates = []
            for src in continuous:
                if src == dst or (dst, src) in edges:
                    continue
                stats = pair_statistics.get(tuple(sorted((src, dst))))
                if stats is not None:
                    candidates.append((src, stats))
            candidates.sort(key=lambda item: (-abs(item[1]["fit_correlation"]), item[0]))
            for src, stats in candidates[:k]:
                r = stats["fit_correlation"]
                edges[(dst, src)] = dict(
                    src=names[src], dst=names[dst], relation_type=AUXILIARY,
                    prior_strength=abs(r), association_sign=1 if r > 0 else -1,
                    origin="normal_data_pearson", provenance=[], attributes={}, **stats)

    for i in range(n):
        edges[(i, i)] = dict(src=names[i], dst=names[i], relation_type=SELF,
                             prior_strength=1.0, origin="computational_self",
                             provenance=[], attributes={})
    support = np.zeros((n, n), dtype=bool)
    weights = np.zeros((n, n), dtype=np.float64)
    relation_ids = np.full((n, n), -1, dtype=np.int64)
    ordered_edges = []
    for (dst, src), edge in sorted(edges.items()):
        support[dst, src] = True
        weights[dst, src] = edge["prior_strength"]
        relation_ids[dst, src] = rel_index[edge["relation_type"]]
        ordered_edges.append(dict(edge, dst_index=dst, src_index=src))
    prior = weights / weights.sum(axis=1, keepdims=True)
    if not np.isfinite(prior).all() or (prior[support] <= 0).any():
        raise ValueError("Graph reference contains nonfinite or nonpositive supported mass.")
    return dict(node_names=names, relation_names=relation_names,
                support=support, weights=weights, prior=prior,
                relation_ids=relation_ids, edges=ordered_edges,
                construction={"config": dict(config),
                              "fit_segments": [[s.start, s.stop] for s in segments],
                              "fit_rows": len(fit), "validation_rows": len(validation),
                              "matrix_convention": "row=dst, column=src"})
