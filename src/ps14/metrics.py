"""Evaluation metrics: regression + event (threshold) + probabilistic (R3 §8).

All regression metrics operate in **log10 space**. Event metrics compare a binary
exceedance prediction / probability against the truth at the 1000 pfu threshold.
Probabilistic metrics score quantile/interval forecasts. These are FULLY IMPLEMENTED and
unit-tested (see tests/test_metrics.py); ``evaluate.py`` calls them per named horizon.

Conventions
-----------
* ``y_true``/``y_pred`` are equal-shaped arrays (1-D ``[N]`` or 2-D ``[N, n_h]``).
* Regression helpers ignore NaN pairwise.
* The "uncertainty factor" ``10**RMSE`` is the intuitive "off by xN" (R1 §5 note).
"""

from __future__ import annotations

import numpy as np

# ======================================================================================
# Helpers
# ======================================================================================


def _finite_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten + drop pairs where either value is non-finite."""
    a = np.asarray(y_true, dtype="float64").ravel()
    b = np.asarray(y_pred, dtype="float64").ravel()
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


# ======================================================================================
# Regression (log10 space)
# ======================================================================================


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    a, b = _finite_pair(y_true, y_pred)
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    a, b = _finite_pair(y_true, y_pred)
    if a.size == 0:
        return float("nan")
    return float(np.mean(np.abs(a - b)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean error (pred - true); positive = over-prediction."""
    a, b = _finite_pair(y_true, y_pred)
    if a.size == 0:
        return float("nan")
    return float(np.mean(b - a))


def prediction_efficiency(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Prediction Efficiency ``PE = 1 - MSE/Var(obs)`` (R3 §8; the field skill score).

    Equivalent to R^2 against the mean: 1 = perfect, 0 = climatology, < 0 = worse.
    """
    a, b = _finite_pair(y_true, y_pred)
    if a.size == 0:
        return float("nan")
    var = np.var(a)
    if var == 0:
        return float("nan")
    return float(1.0 - np.mean((a - b) ** 2) / var)


# R^2 against the mean is identical to PE; expose under the common name too.
r2 = prediction_efficiency


def skill_score(y_true: np.ndarray, y_pred: np.ndarray, y_ref: np.ndarray) -> float:
    """Skill score vs a reference forecast ``1 - MSE_model/MSE_ref`` (R3 §8).

    Reference is persistence (short horizons) or climatology/REFM (long horizons). Must be
    > 0 for the model to be useful.
    """
    a = np.asarray(y_true, dtype="float64").ravel()
    p = np.asarray(y_pred, dtype="float64").ravel()
    r = np.asarray(y_ref, dtype="float64").ravel()
    mask = np.isfinite(a) & np.isfinite(p) & np.isfinite(r)
    a, p, r = a[mask], p[mask], r[mask]
    if a.size == 0:
        return float("nan")
    mse_ref = np.mean((a - r) ** 2)
    if mse_ref == 0:
        return float("nan")
    return float(1.0 - np.mean((a - p) ** 2) / mse_ref)


def linear_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson linear correlation coefficient."""
    a, b = _finite_pair(y_true, y_pred)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def uncertainty_factor(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Intuitive ``10**RMSE(log10 flux)`` — the multiplicative error factor (R1 §5)."""
    r = rmse(y_true, y_pred)
    return float(10.0**r) if r == r else float("nan")


def regression_report(
    y_true: np.ndarray, y_pred: np.ndarray, y_ref: np.ndarray | None = None
) -> dict[str, float]:
    """Bundle the standard per-horizon regression metrics into a dict."""
    out = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "pe": prediction_efficiency(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "lc": linear_correlation(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "uncertainty_factor": uncertainty_factor(y_true, y_pred),
    }
    if y_ref is not None:
        out["skill_vs_ref"] = skill_score(y_true, y_pred, y_ref)
    return out


# ======================================================================================
# Event / threshold (>= 1000 pfu) — from the 2x2 contingency table (R3 §8b)
# ======================================================================================


def contingency_table(y_true_event: np.ndarray, y_pred_event: np.ndarray) -> dict[str, int]:
    """2x2 contingency counts for binary event forecasts.

    Returns ``{"hits": TP, "misses": FN, "false_alarms": FP, "correct_neg": TN}``.
    """
    a = np.asarray(y_true_event).astype(bool).ravel()
    b = np.asarray(y_pred_event).astype(bool).ravel()
    hits = int(np.sum(a & b))
    misses = int(np.sum(a & ~b))
    false_alarms = int(np.sum(~a & b))
    correct_neg = int(np.sum(~a & ~b))
    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_neg": correct_neg,
    }


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else float("nan")


def pod(table: dict[str, int]) -> float:
    """Probability of Detection (hit rate / recall) = H/(H+M)."""
    return _safe_div(table["hits"], table["hits"] + table["misses"])


def far(table: dict[str, int]) -> float:
    """False Alarm Ratio = F/(H+F)."""
    return _safe_div(table["false_alarms"], table["hits"] + table["false_alarms"])


def pofd(table: dict[str, int]) -> float:
    """Probability of False Detection = F/(F+C)."""
    return _safe_div(table["false_alarms"], table["false_alarms"] + table["correct_neg"])


def csi(table: dict[str, int]) -> float:
    """Critical Success Index (threat score) = H/(H+M+F)."""
    return _safe_div(table["hits"], table["hits"] + table["misses"] + table["false_alarms"])


def f1(table: dict[str, int]) -> float:
    """F1 score = 2H/(2H+F+M)."""
    return _safe_div(2 * table["hits"], 2 * table["hits"] + table["false_alarms"] + table["misses"])


def hss(table: dict[str, int]) -> float:
    """Heidke Skill Score (skill vs random chance)."""
    h, m, f, c = table["hits"], table["misses"], table["false_alarms"], table["correct_neg"]
    n = h + m + f + c
    if n == 0:
        return float("nan")
    expected = ((h + m) * (h + f) + (c + m) * (c + f)) / n
    denom = n - expected
    return _safe_div((h + c) - expected, denom)


def tss(table: dict[str, int]) -> float:
    """True Skill Statistic (Peirce) = POD - POFD."""
    p, q = pod(table), pofd(table)
    if p != p or q != q:
        return float("nan")
    return float(p - q)


def roc_auc(y_true_event: np.ndarray, y_score: np.ndarray) -> float:
    """ROC area under curve via the rank (Mann-Whitney U) statistic.

    Parameters
    ----------
    y_true_event:
        Binary labels.
    y_score:
        Predicted probabilities / scores (higher = more likely event).

    Returns
    -------
    float
        AUC in [0, 1]; NaN if only one class is present.
    """
    y = np.asarray(y_true_event).astype(bool).ravel()
    s = np.asarray(y_score, dtype="float64").ravel()
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    n_pos = int(np.sum(y))
    n_neg = int(np.sum(~y))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype="float64")
    ranks[order] = np.arange(1, len(s) + 1)
    # Average ranks for ties.
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sum_ranks = np.zeros(len(counts))
    np.add.at(sum_ranks, inv, ranks)
    avg_ranks = sum_ranks / counts
    ranks = avg_ranks[inv]
    sum_pos = np.sum(ranks[y])
    auc = (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def brier_score(y_true_event: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score = mean((p - o)^2) for probabilistic event forecasts (R3 §8b)."""
    o = np.asarray(y_true_event, dtype="float64").ravel()
    p = np.asarray(y_prob, dtype="float64").ravel()
    mask = np.isfinite(o) & np.isfinite(p)
    o, p = o[mask], p[mask]
    if o.size == 0:
        return float("nan")
    return float(np.mean((p - o) ** 2))


def brier_skill_score(y_true_event: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier Skill Score vs climatology BS_ref (base-rate forecast); > 0 = skillful."""
    o = np.asarray(y_true_event, dtype="float64").ravel()
    p = np.asarray(y_prob, dtype="float64").ravel()
    mask = np.isfinite(o) & np.isfinite(p)
    o, p = o[mask], p[mask]
    if o.size == 0:
        return float("nan")
    base_rate = np.mean(o)
    bs = np.mean((p - o) ** 2)
    bs_ref = np.mean((base_rate - o) ** 2)
    if bs_ref == 0:
        return float("nan")
    return float(1.0 - bs / bs_ref)


def reliability_curve(
    y_true_event: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reliability (calibration) curve: predicted-prob bins vs observed frequency.

    Returns ``(bin_centers, observed_freq, bin_counts)`` for a reliability diagram.
    """
    o = np.asarray(y_true_event, dtype="float64").ravel()
    p = np.asarray(y_prob, dtype="float64").ravel()
    mask = np.isfinite(o) & np.isfinite(p)
    o, p = o[mask], p[mask]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    obs = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype="int64")
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        sel = idx == b
        counts[b] = int(np.sum(sel))
        if counts[b] > 0:
            obs[b] = float(np.mean(o[sel]))
    return centers, obs, counts


def event_report(
    y_true_event: np.ndarray, y_pred_event: np.ndarray, y_prob: np.ndarray | None = None
) -> dict[str, float]:
    """Bundle the standard per-horizon event metrics into a dict."""
    table = contingency_table(y_true_event, y_pred_event)
    out = {
        "pod": pod(table),
        "far": far(table),
        "pofd": pofd(table),
        "csi": csi(table),
        "f1": f1(table),
        "hss": hss(table),
        "tss": tss(table),
        **{f"n_{k}": float(v) for k, v in table.items()},
    }
    if y_prob is not None:
        out["roc_auc"] = roc_auc(y_true_event, y_prob)
        out["brier"] = brier_score(y_true_event, y_prob)
        out["bss"] = brier_skill_score(y_true_event, y_prob)
    return out


# ======================================================================================
# Probabilistic continuous (R3 §8c)
# ======================================================================================


def pinball_loss(y_true: np.ndarray, q_pred: np.ndarray, tau: float) -> float:
    """Pinball (quantile) loss for quantile level ``tau`` in (0, 1)."""
    a, b = _finite_pair(y_true, q_pred)
    if a.size == 0:
        return float("nan")
    diff = a - b
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def crps_from_quantiles(y_true: np.ndarray, quantile_preds: dict[float, np.ndarray]) -> float:
    """Approximate CRPS as the mean pinball loss across provided quantile levels (R3 §8c).

    For a set of quantile forecasts this average-pinball approximation is a standard,
    cheap CRPS estimator.
    """
    if not quantile_preds:
        return float("nan")
    losses = [pinball_loss(y_true, q, tau) for tau, q in quantile_preds.items()]
    losses = [v for v in losses if v == v]
    if not losses:
        return float("nan")
    return float(2.0 * np.mean(losses))


def picp(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction Interval Coverage Probability: fraction of truths within [lower, upper]."""
    a = np.asarray(y_true, dtype="float64").ravel()
    lo = np.asarray(lower, dtype="float64").ravel()
    hi = np.asarray(upper, dtype="float64").ravel()
    mask = np.isfinite(a) & np.isfinite(lo) & np.isfinite(hi)
    a, lo, hi = a[mask], lo[mask], hi[mask]
    if a.size == 0:
        return float("nan")
    return float(np.mean((a >= lo) & (a <= hi)))


__all__ = [
    # regression
    "rmse",
    "mae",
    "bias",
    "prediction_efficiency",
    "r2",
    "skill_score",
    "linear_correlation",
    "uncertainty_factor",
    "regression_report",
    # event
    "contingency_table",
    "pod",
    "far",
    "pofd",
    "csi",
    "f1",
    "hss",
    "tss",
    "roc_auc",
    "brier_score",
    "brier_skill_score",
    "reliability_curve",
    "event_report",
    # probabilistic
    "pinball_loss",
    "crps_from_quantiles",
    "picp",
]
