"""Optional label-based evaluation, isolated from fitting and calibration."""
import numpy as np
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score


def _binary(y, pred):
    tp, fp = int(((y == 1) & pred).sum()), int(((y == 0) & pred).sum())
    fn, tn = int(((y == 1) & ~pred).sum()), int(((y == 0) & ~pred).sum())
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0., F1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def evaluate(labels, scores, threshold, adjacent=None):
    """Evaluate aligned scores without changing their values or threshold.

    ``adjacent[i]`` means that point i immediately follows point i-1 in the
    original time series. Its first value must be False. Unknown labels (-1)
    are excluded from metrics and break positive intervals for point adjustment.
    """
    y, s = np.asarray(labels), np.asarray(scores, float)
    if y.shape != s.shape or y.ndim != 1 or not len(y) or not np.isfinite(s).all():
        raise ValueError("Labels and scores must align point by point")
    if not np.isin(y, [-1, 0, 1]).all():
        raise ValueError("Use labels 0/1 and -1 for unknown; no implicit label conversion")
    threshold_value = np.asarray(threshold, dtype=float)
    if threshold_value.ndim != 0 or not np.isfinite(threshold_value):
        raise ValueError("Threshold must be a finite scalar")
    threshold = float(threshold_value)
    if adjacent is None:
        adjacent = np.r_[False, np.ones(len(y)-1, bool)]
    else:
        adjacent = np.asarray(adjacent)
        if adjacent.shape != y.shape or not np.isin(adjacent, [False, True]).all():
            raise ValueError("Adjacency must be an aligned boolean or 0/1 vector")
        adjacent = adjacent.astype(bool)
        if adjacent[0]:
            raise ValueError("The first scored point has no preceding point; adjacent[0] must be False")
    raw = s > threshold
    adjusted = raw.copy()
    i = 0
    while i < len(y):
        if y[i] != 1:
            i += 1
            continue
        end = i + 1
        while end < len(y) and y[end] == 1 and adjacent[end]:
            end += 1
        if raw[i:end].any():
            adjusted[i:end] = True
        i = end
    known = y >= 0
    metrics = dict(raw=_binary(y[known], raw[known]), point_adjusted=_binary(y[known], adjusted[known]),
                   known_points=int(known.sum()), unknown_points=int((~known).sum()), threshold=float(threshold))
    if len(np.unique(y[known])) == 2:
        precision, recall, _ = precision_recall_curve(y[known], s[known])
        metrics.update(AP=float(average_precision_score(y[known], s[known])),
                       ROC_AUC=float(roc_auc_score(y[known], s[known])),
                       PR_AUC_trapezoidal=float(auc(recall, precision)))
    else:
        metrics.update(AP=None, ROC_AUC=None, PR_AUC_trapezoidal=None)
    return metrics
