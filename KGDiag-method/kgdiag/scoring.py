"""Normal-data score calibration; test labels are not accepted by this module."""
from __future__ import annotations

import warnings
import numpy as np
from scipy.optimize import fmin
from scipy.stats import genpareto


def _vector(values, name):
    """Validate scalar scores before any operation can accidentally broadcast."""
    try:
        x = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric scores") from error
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array")
    return x


def _scalar(value, name):
    try:
        x = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if x.ndim != 0 or not np.isfinite(x):
        raise ValueError(f"{name} must be a finite scalar")
    return float(x)


def _component_vectors(components):
    vectors = {name: _vector(components[name], name) for name in ("R", "DT", "DG")}
    if len({len(x) for x in vectors.values()}) != 1:
        raise ValueError("R, DT, and DG must have equal point counts")
    return vectors


def fit_standardization(components):
    """Population statistics from aligned normal Fit components only.

    A constant component is recorded as unusable for the combined score. Numeric
    overflow is rejected, rather than saving an infinite mean or denominator.
    """
    result = {}
    for name, x in _component_vectors(components).items():
        with np.errstate(over="ignore", invalid="ignore"):
            mean, std = float(x.mean()), float(x.std(ddof=0))
        if not np.isfinite([mean, std]).all():
            raise ValueError(f"Normal Fit statistics overflowed for {name}")
        result[name] = dict(mean=mean, std=std, usable=std > 1e-12 * max(1., abs(mean)))
    return result


def combine(components, statistics, readout="combined"):
    if readout == "R":
        return _vector(components["R"], "R").copy()
    if readout != "combined":
        raise ValueError("readout must be combined or R")
    vectors = _component_vectors(components)
    z = {}
    for name, values in vectors.items():
        stat = statistics[name]
        mean = _scalar(stat["mean"], name + " mean")
        std = _scalar(stat["std"], name + " std")
        usable = stat["usable"]
        if not isinstance(usable, (bool, np.bool_)):
            raise ValueError("Component usability must be a boolean")
        if not usable or std <= 1e-12 * max(1., abs(mean)):
            raise ValueError("Combined score unavailable: degenerate Fit component scale; choose the declared R readout for self-loop controls")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            z[name] = (values - mean) / std
        if not np.isfinite(z[name]).all():
            raise ValueError(f"Nonfinite standardized component {name}")
    with np.errstate(over="ignore", invalid="ignore"):
        score = z["R"] - z["DT"] + z["DG"]
    if not np.isfinite(score).all():
        raise ValueError("Nonfinite combined score")
    return score


def fit_pot(normal_scores, initial_quantile=0.95, tail_probability=0.01, min_exceedances=50):
    """Static POT: strict exceedances, GPD MLE with zero location, no fallback.

    Threshold = u + beta/xi * ((N_u/(n*q))**xi - 1), with its xi=0 limit.
    Overlapping windows do not imply independent exceedances or coverage guarantees.
    """
    s = _vector(normal_scores, "POT normal scores")
    initial_quantile = _scalar(initial_quantile, "POT initial quantile")
    tail_probability = _scalar(tail_probability, "POT tail probability")
    integer_count = isinstance(min_exceedances, (int, np.integer)) and not isinstance(min_exceedances, (bool, np.bool_))
    if not 0 < initial_quantile < 1 or not 0 < tail_probability < 1 or not integer_count or min_exceedances < 1:
        raise ValueError("Invalid POT configuration")
    min_exceedances = int(min_exceedances)
    u = float(np.quantile(s, initial_quantile, method="higher"))
    with np.errstate(over="ignore", invalid="ignore"):
        excess = s[s > u] - u
    if not np.isfinite(excess).all():
        raise ValueError("Nonfinite POT excesses")
    if len(excess) < min_exceedances or not tail_probability < len(excess) / len(s):
        raise ValueError("Insufficient strict POT exceedances or invalid target tail rate")
    with np.errstate(over="ignore", invalid="ignore"):
        condition = float(excess.mean())
    if not np.isfinite(condition) or not condition > 0 or np.ptp(excess) == 0:
        raise ValueError("Degenerate POT excess distribution")
    record = {}

    def optimizer(function, x0, args=(), disp=0):
        x, objective, iterations, calls, flag = fmin(function, x0, args=args, disp=False,
            full_output=True, maxiter=10000, maxfun=20000, xtol=1e-10, ftol=1e-10)
        record.update(converged=flag == 0, iterations=int(iterations), calls=int(calls), objective=float(objective))
        return x

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shape, location, scaled_scale = genpareto.fit(excess / condition, floc=0, optimizer=optimizer)
    scale = float(scaled_scale * condition)
    shape = float(shape)
    if not record.get("converged") or not np.isfinite(record["objective"]):
        raise ValueError("POT maximum-likelihood optimization did not converge")
    if location != 0 or not np.isfinite([shape, scale]).all() or scale <= 0:
        raise ValueError("Invalid POT parameters")
    if not np.all(1 + shape * excess / scale > 0):
        raise ValueError("Observed POT excess outside fitted support")
    ratio = float(np.log(len(excess)) - np.log(len(s)) - np.log(tail_probability))
    with np.errstate(over="ignore", invalid="ignore"):
        increment = scale * ratio if abs(shape) < 1e-6 else scale * np.expm1(shape * ratio) / shape
        threshold = float(u + increment)
    if not np.isfinite(threshold) or threshold < u or 1 + shape * increment / scale <= 0:
        raise ValueError("POT threshold outside fitted support")
    survival = genpareto.sf(increment, shape, loc=0, scale=scale)
    if not np.isclose(survival, len(s) * tail_probability / len(excess), rtol=1e-5, atol=1e-10):
        raise ValueError("POT tail probability check failed")
    return dict(threshold=threshold, initial_threshold=u, shape=shape, scale=scale,
                normal_count=len(s), exceedances=len(excess), initial_quantile=initial_quantile,
                tail_probability=tail_probability, min_exceedances=min_exceedances,
                optimizer=record, warnings=[str(w.message) for w in caught])


def residual_rarity(calibration, residuals):
    cal, values = np.asarray(calibration, float), np.asarray(residuals, float)
    if cal.ndim != 2 or values.ndim != 2 or cal.shape[1] != values.shape[1] or not len(cal) or not cal.shape[1]:
        raise ValueError("Residual reference and queries must have matching target columns")
    if not np.isfinite(cal).all() or not np.isfinite(values).all():
        raise ValueError("Residual rarity requires finite values")
    out = np.empty_like(values)
    for i in range(values.shape[1]):
        ordered = np.sort(cal[:, i])
        # side=left includes all ties in the >= count.
        at_least = len(ordered) - np.searchsorted(ordered, values[:, i], side="left")
        out[:, i] = 1 - (1 + at_least) / (len(ordered) + 1)
    return out
