"""One-step auxiliary-gradient assembly for weak-association KGDiag training.

Adapted from KGDiag's audited shared-gradient-cap implementation. The bound is
on gradients supplied to the optimizer, not on Adam's parameter-update norms.
This module neither steps the optimizer nor loads data or checkpoints.
"""

import math
from typing import Dict, Mapping

import torch


def _norm(values, device):
    items = [x.detach().double().square().sum() for x in values if x is not None]
    return torch.stack(items).sum().sqrt() if items else torch.zeros((), device=device, dtype=torch.float64)


def capped_backward(model: torch.nn.Module, parts: Mapping[str, torch.Tensor],
                    lambda_t: float = 0.001, lambda_g: float = 0.001,
                    cap: float = 0.1) -> Dict[str, float]:
    """Assign grads for 2R plus weak association updates, then return diagnostics.

    Shared parameters are identified by autograd connectivity (a gradient is
    present for both roots), not by parameter names or by nonzero magnitude.
    Their aggregate auxiliary gradient is bounded by ``cap * ||g_R||``.
    Parameters connected to only one root receive that root's nominal gradient.
    All existing parameter gradients are overwritten, including unused ones.
    Call optimizer.step() exactly once after this function.
    """
    if not all(math.isfinite(v) and v >= 0 for v in (lambda_t, lambda_g, cap)):
        raise ValueError("association coefficients and gradient cap must be finite and nonnegative")
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        raise ValueError("model has no trainable parameters")
    device = params[0].device
    if any(parameter.device != device for parameter in params):
        raise ValueError("this gradient assembly requires model parameters on one device")
    rec = 2.0 * parts["reconstruction"]
    assoc = lambda_t * (-parts["temporal_series"] + parts["temporal_prior"])
    assoc = assoc + lambda_g * (-parts["graph_series"] + parts["graph_prior"])
    if rec.numel() != 1 or assoc.numel() != 1:
        raise ValueError("loss terms must reduce to scalars")
    if not bool(torch.isfinite(rec).all()) or not bool(torch.isfinite(assoc).all()):
        raise FloatingPointError("nonfinite loss")
    gr = torch.autograd.grad(rec, params, retain_graph=True, allow_unused=True) if rec.requires_grad else (None,) * len(params)
    ga = torch.autograd.grad(assoc, params, allow_unused=True) if assoc.requires_grad else (None,) * len(params)
    if any(g is not None and not bool(torch.isfinite(g).all()) for g in (*gr, *ga)):
        raise FloatingPointError("nonfinite objective gradient")
    shared = [i for i, (r, a) in enumerate(zip(gr, ga)) if r is not None and a is not None]
    shared_set = set(shared)
    nr = _norm([gr[i] for i in shared], device)
    na = _norm([ga[i] for i in shared], device)
    # Leave a small margin for the parameter dtype's final addition rounding.
    machine_eps = max((torch.finfo(params[i].dtype).eps for i in shared), default=torch.finfo(torch.float32).eps)
    reserve = max(0.0, 1.0 - 8.0 * machine_eps)
    coefficient = min(1.0, float(reserve * cap * nr / (na + 1e-12))) if shared else 1.0
    if float(nr) == 0 and shared:
        coefficient = 0.0

    def assembled(scale):
        values, scaled, effective = [], [], []
        for i, (r, a) in enumerate(zip(gr, ga)):
            if i in shared_set:
                auxiliary = a.detach() * scale
                value = r.detach() + auxiliary
                scaled.append(auxiliary)
                effective.append(value - r.detach())
            elif r is not None:
                value = r.detach()
            elif a is not None:
                value = a.detach()
            else:
                value = None
            values.append(value)
        return values, _norm(scaled, device), _norm(effective, device)

    final, after, effective = assembled(coefficient)
    bound = cap * nr
    # Usually the reserve is sufficient. If final dtype rounding breaks the
    # declared bound, reduce only the auxiliary multiplier until it is valid.
    attempts = 0
    while bool((after > bound) | (effective > bound)):
        coefficient *= 0.5
        final, after, effective = assembled(coefficient)
        attempts += 1
        if attempts > 80:
            coefficient = 0.0
            final, after, effective = assembled(coefficient)
            break
    if not all(value is None or bool(torch.isfinite(value).all()) for value in final):
        raise FloatingPointError("nonfinite assembled gradient")
    if bool((after > bound) | (effective > bound)):
        raise FloatingPointError("shared auxiliary gradient bound failed")
    # Assign only after all checks succeed; stale frozen-parameter gradients are
    # cleared as well, so this function never accidentally accumulates updates.
    for parameter in model.parameters():
        parameter.grad = None
    for parameter, value in zip(params, final):
        parameter.grad = value
    exclusive = _norm([a for r, a in zip(gr, ga) if r is None], device)
    dot = sum(float((gr[i].detach().double() * ga[i].detach().double()).sum()) for i in shared)
    nr_float, na_float = float(nr), float(na)
    return {
        "reconstruction_loss": float(parts["reconstruction"].detach()),
        "association_loss": float(assoc.detach()),
        "nominal_loss": float((rec + assoc).detach()),
        "norm_R_shared": nr_float,
        "norm_A_before": na_float,
        "norm_A_after": float(after),
        "norm_effective_A_after": float(effective),
        "coefficient": float(coefficient),
        "rho_before": na_float / nr_float if nr_float else 0.0,
        "rho_after": float(after / nr) if nr_float else 0.0,
        "effective_rho_after": float(effective / nr) if nr_float else 0.0,
        "cosine_R_A": dot / (nr_float * na_float) if nr_float and na_float else 0.0,
        "association_only_norm": float(exclusive),
        "shared_parameter_count": float(len(shared)),
        "clipped": float(coefficient < 1.0),
        "zero_R_shared": float(nr_float == 0.0),
        "bound_passed": 1.0,
    }
