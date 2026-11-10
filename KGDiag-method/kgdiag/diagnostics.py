"""Event-average inspection rankings, with explicit candidate coverage."""
import numpy as np
from scipy.stats import rankdata


def rank_candidates(values, names, available=None):
    values = np.asarray(values, dtype=float)
    if values.shape != (len(names),) or len(set(names)) != len(names):
        raise ValueError("One score per uniquely named candidate is required")
    available = np.ones(len(names), bool) if available is None else np.asarray(available, bool)
    if available.shape != values.shape:
        raise ValueError("Candidate availability does not match names")
    valid = available & np.isfinite(values)
    ranks = np.full(len(names), np.nan)
    ranks[valid] = rankdata(-values[valid], method="average")
    order = sorted(np.flatnonzero(valid), key=lambda i: (-values[i], i))
    return dict(ranking=[dict(name=names[i], score=float(values[i]), rank=float(ranks[i])) for i in order],
                unavailable=[names[i] for i in np.flatnonzero(~valid)])


def event_evidence(endpoints, mixed_node, relation, start, stop, target_names, relation_names,
                   relation_available, available_until=None):
    """Average scores before ranking over [start,stop) raw row indices.

    A complete interval gives retrospective evidence. available_until can cap
    the query at an actual output endpoint; no later observations enter its mean.
    """
    end = np.asarray(endpoints)
    node, rel = np.asarray(mixed_node, float), np.asarray(relation, float)
    if end.ndim != 1 or np.any(np.diff(end) <= 0) or start >= stop:
        raise ValueError("Endpoints must increase and event interval must be nonempty")
    if node.shape != (len(end),len(target_names)) or rel.shape != (len(end),len(relation_names)):
        raise ValueError("Evidence shapes do not match candidate names")
    selected = (end >= start) & (end < stop)
    if available_until is not None:
        selected &= end <= available_until
    if not selected.any():
        return dict(scored_endpoints=0, variables=None, relations=None)
    if not np.isfinite(node[selected]).all() or not np.isfinite(rel[selected]).all():
        raise ValueError("Nonfinite event evidence")
    return dict(scored_endpoints=int(selected.sum()),
                first_endpoint=int(end[selected][0]), last_endpoint=int(end[selected][-1]),
                variables=rank_candidates(node[selected].mean(0),target_names),
                relations=rank_candidates(rel[selected].mean(0),relation_names,relation_available))
