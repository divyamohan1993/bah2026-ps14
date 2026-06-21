"""Tests for the baseline tier (ps14.models.baselines).

Persistence is FULLY IMPLEMENTED and tested end-to-end against the windowing contract.
The other baselines are contract stubs: we assert they instantiate and expose the
Forecaster interface (and raise NotImplementedError until implemented).
"""

from __future__ import annotations

import numpy as np
import pytest

from ps14.constants import HORIZON_NAMES, LOG_HARSH
from ps14.datasets import schema
from ps14.models import Forecaster
from ps14.models.baselines import (
    Climatology,
    LightGBMForecaster,
    Persistence,
    RefmLinearFilter,
)


def _make_windows(n: int = 16, lookback: int = 10):
    f = len(schema.FEATURE_COLUMNS)
    f_kf = len(schema.KNOWN_FUTURE_COLUMNS)
    n_h = len(HORIZON_NAMES)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, lookback, f)).astype("float32")
    X_future = rng.normal(size=(n, 144, f_kf)).astype("float32")
    y = rng.normal(size=(n, n_h)).astype("float32")
    y_exceed = (y >= LOG_HARSH).astype("float32")
    return X, X_future, y, y_exceed


def test_persistence_is_a_forecaster():
    assert issubclass(Persistence, Forecaster)
    assert Persistence().horizon_names == HORIZON_NAMES


def test_persistence_repeats_last_observed_log_flux():
    X, X_future, y, y_exceed = _make_windows()
    model = Persistence().fit(X, X_future, y, y_exceed)
    pred = model.predict(X, X_future)
    # Shape is [N, n_h].
    assert pred.shape == (X.shape[0], len(HORIZON_NAMES))
    # Every horizon equals the last encoder step of the target channel.
    target_channel = schema.FEATURE_COLUMNS.index(schema.TARGET)
    expected_last = X[:, -1, target_channel]
    for h in range(pred.shape[1]):
        np.testing.assert_allclose(pred[:, h], expected_last, rtol=1e-6)


def test_persistence_quantiles_degenerate_and_proba_step():
    X, X_future, y, y_exceed = _make_windows()
    model = Persistence()
    q = model.predict_quantiles(X, X_future)
    assert set(q.keys()) == {0.1, 0.5, 0.9}
    np.testing.assert_array_equal(q[0.1], q[0.9])  # degenerate
    proba = model.predict_proba_exceed(X, X_future)
    assert proba.shape == (X.shape[0], len(HORIZON_NAMES))
    assert set(np.unique(proba)).issubset({0.0, 1.0})


def test_persistence_save_load_roundtrip(tmp_path):
    X, X_future, y, y_exceed = _make_windows()
    model = Persistence()
    path = tmp_path / "persistence.npz"
    model.save(path)
    loaded = Persistence.load(path)
    np.testing.assert_array_equal(model.predict(X, X_future), loaded.predict(X, X_future))


@pytest.mark.parametrize("cls", [Climatology, LightGBMForecaster, RefmLinearFilter])
def test_other_baselines_are_forecasters_and_stubbed(cls):
    assert issubclass(cls, Forecaster)
    model = cls()
    X, X_future, y, y_exceed = _make_windows()
    with pytest.raises(NotImplementedError):
        model.fit(X, X_future, y, y_exceed)
