"""Full-support association losses, endpoint scores, and additive graph evidence.

The probability and Jeffreys operations are extracted from KGDiag's audited
weak-association implementation. The reductions follow the current manuscript:
training averages layers; detection sums layers and uses a fixed factor of 10.
No relation-specific renormalization, checkpoint, dataset, or plotting code is
used here.

Model output contract::

    reconstruction:  Tensor[B, L, Nrec]
    temporal_series, temporal_prior: lists of Tensor[B, H, L, L]
    graph_series, graph_prior:       lists of Tensor[B, L, H, N, N]
    support:                         BoolTensor[N, N]

Graph row i is a receiver, and column j is a message source. Nrec is allowed to
differ from N because categorical context nodes need not be reconstructed.
"""

import math
from typing import Dict, Mapping, Optional

import torch


def normalize_supported(values: torch.Tensor, support: Optional[torch.Tensor] = None,
                        eps: float = 1e-8) -> torch.Tensor:
    """Row-normalize and smooth only the permitted entries.

    For a support of size m, the result is (x/sum(x) + eps)/(1 + eps*m)
    on the support, and exactly zero elsewhere. Float16/bfloat16 inputs use
    float32 for this calculation; float32 and float64 are preserved. Invalid or
    empty supported rows are rejected instead of being silently repaired.
    """
    if not isinstance(values, torch.Tensor) or not values.is_floating_point():
        raise TypeError("values must be a floating-point tensor")
    if values.ndim < 1 or values.shape[-1] == 0:
        raise ValueError("values must have a nonempty distribution axis")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    x = values.float() if values.dtype in (torch.float16, torch.bfloat16) else values
    mask = torch.ones_like(x, dtype=torch.bool) if support is None else torch.as_tensor(
        support, device=x.device, dtype=torch.bool)
    try:
        mask = torch.broadcast_to(mask, x.shape)
    except RuntimeError as error:
        raise ValueError("support is not broadcastable to the probability tensor") from error
    degree = mask.sum(dim=-1, keepdim=True)
    if bool((degree == 0).any()):
        raise ValueError("every probability row requires at least one supported entry")
    supported = x.masked_select(mask)
    if not bool(torch.isfinite(supported).all()) or bool((supported < 0).any()):
        raise ValueError("supported probabilities must be finite and nonnegative")
    masked = torch.where(mask, x, torch.zeros_like(x))
    mass = masked.sum(dim=-1, keepdim=True)
    if not bool(torch.isfinite(mass).all()) or bool((mass <= 0).any()):
        raise ValueError("supported probability rows must have positive finite mass")
    normalized = masked / mass
    smoothed = torch.where(mask, normalized + eps, torch.zeros_like(normalized))
    return smoothed / (1.0 + eps * degree.to(x.dtype))


def edge_jeffreys(series: torch.Tensor, prior: torch.Tensor,
                  support: Optional[torch.Tensor] = None, eps: float = 1e-8) -> torch.Tensor:
    """Nonnegative per-entry contributions to symmetric KL on full support.

    Unsupported logarithms are never evaluated. Do not mask to an individual
    relation type before calling this function; group the returned entries
    afterward to conserve node and relation-type evidence.
    """
    if series.shape != prior.shape:
        raise ValueError("series and prior associations must have the same shape")
    if series.device != prior.device:
        raise ValueError("series and prior must be on the same device")
    s = normalize_supported(series, support, eps)
    p = normalize_supported(prior, support, eps)
    mask = torch.ones_like(s, dtype=torch.bool) if support is None else torch.as_tensor(
        support, dtype=torch.bool, device=s.device)
    log_s = torch.where(mask, s, torch.ones_like(s)).log()
    log_p = torch.where(mask, p, torch.ones_like(p)).log()
    return (s - p) * (log_s - log_p)


def _validate_output(output: Mapping, target: torch.Tensor, support: torch.Tensor):
    prediction = output["reconstruction"]
    if prediction.ndim != 3 or prediction.shape != target.shape:
        raise ValueError("reconstruction and target must have matching [B,L,Nrec] shapes")
    if any(size == 0 for size in prediction.shape):
        raise ValueError("empty reconstruction dimensions are not supported")
    if prediction.device != target.device:
        raise ValueError("target and reconstruction must be on the same device")
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(target).all()):
        raise ValueError("reconstruction and target must be finite")
    support = torch.as_tensor(support, dtype=torch.bool, device=prediction.device)
    if support.ndim != 2 or support.shape[0] != support.shape[1] or not support.shape[0]:
        raise ValueError("graph support must be a nonempty square [N,N] matrix")
    b, length, _ = prediction.shape
    n = support.shape[0]
    groups = []
    for branch in ("temporal", "graph"):
        series = output[branch + "_series"]
        prior = output[branch + "_prior"]
        if not isinstance(series, (list, tuple)) or not isinstance(prior, (list, tuple)):
            raise TypeError("association layers must be lists or tuples")
        if not series or len(series) != len(prior):
            raise ValueError("each branch requires matched nonempty association-layer lists")
        for s, p in zip(series, prior):
            if s.shape != p.shape or s.device != prediction.device or p.device != prediction.device:
                raise ValueError("association pairs must match shape and model device")
            valid_shape = (s.ndim == 4 and s.shape[0] == b and s.shape[1] > 0
                           and s.shape[-2:] == (length, length)) if branch == "temporal" else (
                               s.ndim == 5 and s.shape[:2] == (b, length)
                               and s.shape[2] > 0 and s.shape[-2:] == (n, n))
            if not valid_shape:
                raise ValueError("invalid " + branch + " association dimensions")
        groups.append(list(zip(series, prior)))
    return prediction, support, groups[0], groups[1]


def loss_terms(output: Mapping, target: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Whole-window reconstruction and four distinct stop-gradient loss roots."""
    prediction, support, temporal, graph = _validate_output(output, target, output["support"])
    terms = {"reconstruction": (prediction - target).square().mean()}
    for branch, pairs, mask in (("temporal", temporal, None), ("graph", graph, support)):
        series_terms = [edge_jeffreys(s, p.detach(), mask).sum(-1).mean() for s, p in pairs]
        prior_terms = [edge_jeffreys(s.detach(), p, mask).sum(-1).mean() for s, p in pairs]
        terms[branch + "_series"] = torch.stack(series_terms).mean()
        terms[branch + "_prior"] = torch.stack(prior_terms).mean()
    return terms


@torch.no_grad()
def components(output: Mapping, target: torch.Tensor,
               support: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Compute raw endpoint scores and graph evidence without calibration.

    DT averages heads and all time-query rows; DG averages heads and nodes at
    the endpoint. Both sum layers and multiply by 10. ``edge`` averages layers
    and heads, so ``edge.sum((-2,-1)) == N*DG/(10*K_G)``. ``residual`` is the
    raw per-target squared error, not divided by Nrec.
    """
    prediction, support, temporal, graph = _validate_output(output, target, support)
    residual = (prediction[:, -1] - target[:, -1]).square()
    temporal_rows = [edge_jeffreys(s, p).sum(-1).mean(dim=(1, 2)) for s, p in temporal]
    graph_edges = [edge_jeffreys(s[:, -1], p[:, -1], support) for s, p in graph]
    dg = 10.0 * torch.stack([edge.sum(-1).mean(dim=(1, 2)) for edge in graph_edges]).sum(0)
    edge = torch.stack([entry.mean(dim=1) for entry in graph_edges]).mean(0)
    return {
        "R": residual.mean(-1),
        "DT": 10.0 * torch.stack(temporal_rows).sum(0),
        "DG": dg,
        "residual": residual,
        "edge": edge,
        "node_graph": edge.sum(-1),
    }
