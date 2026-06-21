"""Real tests for the evaluation metrics (ps14.metrics).

Validates regression, event/contingency, and probabilistic metrics against hand-computed
or scikit-learn reference values (R3 §8).
"""

from __future__ import annotations

import numpy as np
import pytest

from ps14 import metrics

# --------------------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------------------


def test_rmse_mae_bias_exact():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.0, 2.0, 4.0, 6.0])  # errors 0,0,+1,+2
    assert metrics.rmse(y, p) == pytest.approx(np.sqrt((0 + 0 + 1 + 4) / 4))
    assert metrics.mae(y, p) == pytest.approx((0 + 0 + 1 + 2) / 4)
    assert metrics.bias(y, p) == pytest.approx((0 + 0 + 1 + 2) / 4)


def test_perfect_prediction_efficiency():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert metrics.prediction_efficiency(y, y) == pytest.approx(1.0)
    assert metrics.r2(y, y) == pytest.approx(1.0)


def test_mean_forecast_has_zero_pe():
    rng = np.random.default_rng(0)
    y = rng.normal(size=500)
    p = np.full_like(y, y.mean())
    assert metrics.prediction_efficiency(y, p) == pytest.approx(0.0, abs=1e-12)


def test_skill_score_zero_when_equal_to_reference():
    rng = np.random.default_rng(1)
    y = rng.normal(size=200)
    ref = y + rng.normal(scale=0.5, size=200)
    assert metrics.skill_score(y, ref, ref) == pytest.approx(0.0)
    # A better-than-reference model has positive skill.
    better = y + rng.normal(scale=0.1, size=200)
    assert metrics.skill_score(y, better, ref) > 0


def test_uncertainty_factor():
    y = np.array([0.0, 0.0, 0.0])
    p = np.array([1.0, 1.0, 1.0])  # RMSE = 1 in log10 -> factor 10
    assert metrics.uncertainty_factor(y, p) == pytest.approx(10.0)


def test_linear_correlation_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert metrics.linear_correlation(y, 2 * y + 1) == pytest.approx(1.0)


def test_nan_pairs_ignored():
    y = np.array([1.0, np.nan, 3.0])
    p = np.array([1.0, 2.0, 3.0])
    assert metrics.rmse(y, p) == pytest.approx(0.0)


# --------------------------------------------------------------------------------------
# Event / contingency
# --------------------------------------------------------------------------------------


def test_contingency_counts():
    truth = np.array([1, 1, 0, 0, 1])
    pred = np.array([1, 0, 0, 1, 1])
    t = metrics.contingency_table(truth, pred)
    assert t == {"hits": 2, "misses": 1, "false_alarms": 1, "correct_neg": 1}


def test_perfect_event_scores():
    truth = np.array([1, 0, 1, 0, 1, 0])
    t = metrics.contingency_table(truth, truth)
    assert metrics.pod(t) == pytest.approx(1.0)
    assert metrics.far(t) == pytest.approx(0.0)
    assert metrics.csi(t) == pytest.approx(1.0)
    assert metrics.f1(t) == pytest.approx(1.0)
    assert metrics.hss(t) == pytest.approx(1.0)
    assert metrics.tss(t) == pytest.approx(1.0)


def test_tss_equals_pod_minus_pofd():
    truth = np.array([1, 1, 0, 0, 0, 1, 0, 1])
    pred = np.array([1, 0, 1, 0, 0, 1, 1, 1])
    t = metrics.contingency_table(truth, pred)
    assert metrics.tss(t) == pytest.approx(metrics.pod(t) - metrics.pofd(t))


def test_roc_auc_perfect_and_random():
    truth = np.array([0, 0, 1, 1])
    perfect = np.array([0.1, 0.2, 0.8, 0.9])
    assert metrics.roc_auc(truth, perfect) == pytest.approx(1.0)
    inverted = np.array([0.9, 0.8, 0.2, 0.1])
    assert metrics.roc_auc(truth, inverted) == pytest.approx(0.0)


def test_roc_auc_single_class_is_nan():
    assert np.isnan(metrics.roc_auc(np.array([1, 1, 1]), np.array([0.2, 0.3, 0.9])))


def test_brier_and_skill():
    truth = np.array([1.0, 0.0, 1.0, 0.0])
    perfect = np.array([1.0, 0.0, 1.0, 0.0])
    assert metrics.brier_score(truth, perfect) == pytest.approx(0.0)
    assert metrics.brier_skill_score(truth, perfect) == pytest.approx(1.0)


def test_reliability_curve_shapes():
    rng = np.random.default_rng(2)
    p = rng.uniform(size=1000)
    o = (rng.uniform(size=1000) < p).astype(float)
    centers, obs, counts = metrics.reliability_curve(o, p, n_bins=10)
    assert centers.shape == (10,)
    assert obs.shape == (10,)
    assert counts.sum() == 1000


# --------------------------------------------------------------------------------------
# Probabilistic continuous
# --------------------------------------------------------------------------------------


def test_pinball_loss_median_equals_half_mae():
    y = np.array([1.0, 2.0, 3.0, 10.0])
    q = np.array([1.0, 2.0, 3.0, 4.0])
    # For tau=0.5 the pinball loss is 0.5 * MAE.
    assert metrics.pinball_loss(y, q, 0.5) == pytest.approx(0.5 * metrics.mae(y, q))


def test_pinball_asymmetry():
    y = np.array([10.0])
    q = np.array([0.0])  # under-prediction
    # High tau penalizes under-prediction more.
    assert metrics.pinball_loss(y, q, 0.9) > metrics.pinball_loss(y, q, 0.1)


def test_picp_coverage():
    y = np.array([0.5, 1.5, 2.5, 3.5])
    lo = np.array([0.0, 1.0, 2.0, 3.0])
    hi = np.array([1.0, 2.0, 4.0, 3.2])  # samples 0-2 inside, sample 3 (3.5>3.2) outside
    assert metrics.picp(y, lo, hi) == pytest.approx(0.75)


def test_crps_from_quantiles_runs():
    rng = np.random.default_rng(3)
    y = rng.normal(size=100)
    preds = {0.1: y - 1.0, 0.5: y, 0.9: y + 1.0}
    val = metrics.crps_from_quantiles(y, preds)
    assert val >= 0
