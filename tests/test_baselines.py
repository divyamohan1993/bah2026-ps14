"""Tests for the baseline tier + model contract surface (ps14.models).

Constructs SMALL synthetic window tensors inline (matching the CONTRACTS.md §4 shapes) so
the tests do not depend on other builders' data modules. Asserts that Persistence /
Climatology / REFM fit + predict produce the right shapes, that ``predict_quantiles``
returns ordered P10 <= P50 <= P90, that ``predict_proba_exceed`` is in ``[0, 1]``, and that
``save``/``load`` round-trip. LightGBM is guarded by ``importorskip``; the deep-learning
models are asserted importable with an informative error when torch is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from ps14.constants import HORIZON_NAMES, HORIZON_STEPS, LOG_HARSH
from ps14.datasets import schema
from ps14.models import Forecaster
from ps14.models.baselines import (
    Climatology,
    LightGBMForecaster,
    Persistence,
    REFMForecaster,
    RefmLinearFilter,
)

_DECODER = HORIZON_STEPS["12h"]


def _make_windows(n: int = 64, lookback: int = 12, seed: int = 0):
    """Build small synthetic window tensors with the contract shapes."""
    f = len(schema.FEATURE_COLUMNS)
    f_kf = len(schema.KNOWN_FUTURE_COLUMNS)
    n_h = len(HORIZON_NAMES)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, lookback, f)).astype("float32")
    # Make the known-future cyclic covariates look like real sin/cos pairs over the decoder.
    X_future = rng.uniform(-1.0, 1.0, size=(n, _DECODER, f_kf)).astype("float32")
    # Spread the target across the harsh threshold so both event classes appear.
    y = rng.normal(loc=LOG_HARSH, scale=1.0, size=(n, n_h)).astype("float32")
    y_exceed = (y >= LOG_HARSH).astype("float32")
    return X, X_future, y, y_exceed


def _assert_quantiles_ordered(q: dict[float, np.ndarray], n: int) -> None:
    assert set(q.keys()) == {0.1, 0.5, 0.9}
    for tau, arr in q.items():
        assert arr.shape == (n, len(HORIZON_NAMES)), f"quantile {tau} wrong shape"
    assert np.all(q[0.1] <= q[0.5] + 1e-5)
    assert np.all(q[0.5] <= q[0.9] + 1e-5)


def _assert_proba_unit(p: np.ndarray, n: int) -> None:
    assert p.shape == (n, len(HORIZON_NAMES))
    assert np.all(p >= 0.0) and np.all(p <= 1.0)


# ======================================================================================
# Contract surface
# ======================================================================================
@pytest.mark.parametrize(
    "cls", [Persistence, Climatology, LightGBMForecaster, REFMForecaster]
)
def test_baselines_are_forecasters(cls):
    assert issubclass(cls, Forecaster)
    assert cls().horizon_names == HORIZON_NAMES


def test_refm_alias_is_refm_forecaster():
    assert RefmLinearFilter is REFMForecaster


# ======================================================================================
# Persistence
# ======================================================================================
def test_persistence_repeats_last_observed_log_flux():
    X, X_future, y, y_exceed = _make_windows()
    model = Persistence().fit(X, X_future, y, y_exceed)
    pred = model.predict(X, X_future)
    assert pred.shape == (X.shape[0], len(HORIZON_NAMES))
    target_channel = schema.FEATURE_COLUMNS.index(schema.TARGET)
    expected_last = X[:, -1, target_channel]
    for h in range(pred.shape[1]):
        np.testing.assert_allclose(pred[:, h], expected_last, rtol=1e-6)


def test_persistence_quantiles_degenerate_and_proba_step():
    X, X_future, _, _ = _make_windows()
    model = Persistence()
    q = model.predict_quantiles(X, X_future)
    assert set(q.keys()) == {0.1, 0.5, 0.9}
    np.testing.assert_array_equal(q[0.1], q[0.9])
    proba = model.predict_proba_exceed(X, X_future)
    _assert_proba_unit(proba, X.shape[0])
    assert set(np.unique(proba)).issubset({0.0, 1.0})


def test_persistence_save_load_roundtrip(tmp_path):
    X, X_future, y, y_exceed = _make_windows()
    model = Persistence()
    path = tmp_path / "persistence.npz"
    model.save(path)
    loaded = Persistence.load(path)
    np.testing.assert_array_equal(model.predict(X, X_future), loaded.predict(X, X_future))


# ======================================================================================
# Climatology
# ======================================================================================
def test_climatology_fit_predict_shapes_and_quantiles():
    X, X_future, y, y_exceed = _make_windows()
    model = Climatology(n_tod_bins=6, n_kp_bins=4, n_doy_bins=4).fit(X, X_future, y, y_exceed)
    pred = model.predict(X, X_future)
    assert pred.shape == (X.shape[0], len(HORIZON_NAMES))
    assert np.all(np.isfinite(pred))
    _assert_quantiles_ordered(model.predict_quantiles(X, X_future), X.shape[0])
    _assert_proba_unit(model.predict_proba_exceed(X, X_future), X.shape[0])


def test_climatology_predict_requires_fit():
    X, X_future, _, _ = _make_windows()
    with pytest.raises(RuntimeError):
        Climatology().predict(X, X_future)


def test_climatology_save_load_roundtrip(tmp_path):
    X, X_future, y, y_exceed = _make_windows()
    model = Climatology(n_tod_bins=6, n_kp_bins=4, n_doy_bins=4).fit(X, X_future, y, y_exceed)
    path = tmp_path / "climatology.json"
    model.save(path)
    loaded = Climatology.load(path)
    np.testing.assert_allclose(
        model.predict(X, X_future), loaded.predict(X, X_future), rtol=1e-5
    )
    q0, q1 = model.predict_quantiles(X, X_future), loaded.predict_quantiles(X, X_future)
    for tau in (0.1, 0.5, 0.9):
        np.testing.assert_allclose(q0[tau], q1[tau], rtol=1e-5)


# ======================================================================================
# REFM
# ======================================================================================
def test_refm_fit_predict_shapes_and_quantiles():
    X, X_future, y, y_exceed = _make_windows()
    model = REFMForecaster(recent_steps=8).fit(X, X_future, y, y_exceed)
    pred = model.predict(X, X_future)
    assert pred.shape == (X.shape[0], len(HORIZON_NAMES))
    assert np.all(np.isfinite(pred))
    _assert_quantiles_ordered(model.predict_quantiles(X, X_future), X.shape[0])
    _assert_proba_unit(model.predict_proba_exceed(X, X_future), X.shape[0])


def test_refm_predict_requires_fit():
    X, X_future, _, _ = _make_windows()
    with pytest.raises(RuntimeError):
        REFMForecaster().predict(X, X_future)


def test_refm_save_load_roundtrip(tmp_path):
    X, X_future, y, y_exceed = _make_windows()
    model = REFMForecaster(recent_steps=8).fit(X, X_future, y, y_exceed)
    path = tmp_path / "refm.joblib"
    model.save(path)
    loaded = REFMForecaster.load(path)
    np.testing.assert_allclose(
        model.predict(X, X_future), loaded.predict(X, X_future), rtol=1e-5
    )


# ======================================================================================
# LightGBM (optional dependency)
# ======================================================================================
def test_lightgbm_fit_predict_and_roundtrip(tmp_path):
    pytest.importorskip("lightgbm")
    X, X_future, y, y_exceed = _make_windows(n=128)
    model = LightGBMForecaster(params={"n_estimators": 20, "num_leaves": 7}).fit(
        X, X_future, y, y_exceed
    )
    pred = model.predict(X, X_future)
    assert pred.shape == (X.shape[0], len(HORIZON_NAMES))
    _assert_quantiles_ordered(model.predict_quantiles(X, X_future), X.shape[0])
    _assert_proba_unit(model.predict_proba_exceed(X, X_future), X.shape[0])
    importances = model.feature_importance()
    assert set(importances.keys()) == set(HORIZON_NAMES)

    path = tmp_path / "lightgbm.joblib"
    model.save(path)
    loaded = LightGBMForecaster.load(path)
    np.testing.assert_allclose(
        model.predict(X, X_future), loaded.predict(X, X_future), rtol=1e-5
    )


def test_lightgbm_missing_dep_raises_informative():
    """When lightgbm is absent, fit must raise a clear ImportError (not ModuleNotFound)."""
    import importlib.util

    if importlib.util.find_spec("lightgbm") is not None:
        pytest.skip("lightgbm is installed; missing-dep path not exercised")
    X, X_future, y, y_exceed = _make_windows()
    with pytest.raises(ImportError, match="lightgbm"):
        LightGBMForecaster().fit(X, X_future, y, y_exceed)


# ======================================================================================
# Deep-learning models: importable without torch; informative error on fit
# ======================================================================================
def test_dl_models_importable_and_guarded():
    import importlib.util

    from ps14.models.foundation import FoundationForecaster
    from ps14.models.nhits import NHiTSForecaster
    from ps14.models.tft import TFTForecaster

    # All must instantiate without any heavy dependency.
    assert issubclass(TFTForecaster, Forecaster)
    assert issubclass(NHiTSForecaster, Forecaster)
    assert issubclass(FoundationForecaster, Forecaster)
    TFTForecaster()
    NHiTSForecaster()
    FoundationForecaster()

    if importlib.util.find_spec("torch") is None:
        X, X_future, y, y_exceed = _make_windows(n=8)
        with pytest.raises(ImportError, match="\\[dl\\]|deep-learning"):
            TFTForecaster().fit(X, X_future, y, y_exceed)
        with pytest.raises(ImportError, match="\\[dl\\]|deep-learning"):
            NHiTSForecaster().fit(X, X_future, y, y_exceed)


def test_foundation_missing_dep_raises_informative():
    import importlib.util

    if importlib.util.find_spec("chronos") is not None:
        pytest.skip("chronos is installed; missing-dep path not exercised")
    from ps14.models.foundation import FoundationForecaster

    X, X_future, y, y_exceed = _make_windows(n=8)
    with pytest.raises(ImportError, match="chronos"):
        FoundationForecaster().fit(X, X_future, y, y_exceed)


# ======================================================================================
# Deep-learning smoke tests (only when the [dl] stack is installed) — tiny data, 1 epoch
# ======================================================================================
def _tiny_dl_params():
    return {
        "hidden_size": 8,
        "lstm_layers": 1,
        "attention_head_size": 1,
        "hidden_continuous_size": 4,
        "max_epochs": 1,
        "batch_size": 16,
    }


def test_tft_smoke_fit_predict_roundtrip(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pytorch_forecasting")
    from ps14.models.tft import TFTForecaster

    X, X_future, y, y_exceed = _make_windows(n=24, lookback=16)
    model = TFTForecaster(params=_tiny_dl_params(), lookback=16).fit(X, X_future, y, y_exceed)
    pred = model.predict(X, X_future)
    assert pred.shape == (X.shape[0], len(HORIZON_NAMES))
    assert np.all(np.isfinite(pred))
    _assert_quantiles_ordered(model.predict_quantiles(X, X_future), X.shape[0])
    _assert_proba_unit(model.predict_proba_exceed(X, X_future), X.shape[0])

    path = tmp_path / "tft.joblib"
    model.save(path)
    loaded = TFTForecaster.load(path)
    assert loaded.predict(X, X_future).shape == (X.shape[0], len(HORIZON_NAMES))


def test_nhits_smoke_fit_predict(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pytorch_forecasting")
    from ps14.models.nhits import NHiTSForecaster

    X, X_future, y, y_exceed = _make_windows(n=24, lookback=16)
    model = NHiTSForecaster(
        params={"hidden_size": 16, "max_epochs": 1, "batch_size": 16}, lookback=16
    ).fit(X, X_future, y, y_exceed)
    pred = model.predict(X, X_future)
    assert pred.shape == (X.shape[0], len(HORIZON_NAMES))
    assert np.all(np.isfinite(pred))
    _assert_quantiles_ordered(model.predict_quantiles(X, X_future), X.shape[0])
    _assert_proba_unit(model.predict_proba_exceed(X, X_future), X.shape[0])


# ======================================================================================
# train + evaluate orchestration (baseline path; no heavy deps required)
# ======================================================================================
def test_train_and_evaluate_baseline_end_to_end(tmp_path):
    from ps14 import evaluate as ev
    from ps14 import train as tr
    from ps14.config import load_config
    from ps14.datasets.windowing import WindowTensors

    n = 240
    X, X_future, y, y_exceed = _make_windows(n=n, lookback=16)
    t_index = np.arange(n).astype("datetime64[ns]")
    wt = WindowTensors(
        X, X_future, y, y_exceed, t_index,
        schema.FEATURE_COLUMNS, schema.KNOWN_FUTURE_COLUMNS, HORIZON_NAMES,
    )
    cfg = load_config()
    cfg.split.embargo_steps = 5
    cfg.paths.models = tmp_path

    assert set(tr.MODEL_REGISTRY) == {
        "persistence", "climatology", "lightgbm", "refm", "tft", "nhits", "foundation"
    }

    model = tr.train(cfg, model_name="climatology", windows=wt, save=True)
    assert (tmp_path / "climatology.json").exists()

    split_kwargs = dict(train=cfg.split.train, val=cfg.split.val, embargo_steps=5)
    results = ev.evaluate(model, wt, split="test", split_kwargs=split_kwargs)
    assert set(results) == set(HORIZON_NAMES)
    for horizon in HORIZON_NAMES:
        row = results[horizon]
        for key in ("rmse", "mae", "pe", "skill_vs_persistence", "pod", "roc_auc", "crps"):
            assert key in row
    # metrics table is a DataFrame indexed by horizon.
    table = ev.metrics_table(results)
    assert list(table.index) == HORIZON_NAMES


def test_evaluate_handles_empty_split():
    """With the contract default embargo, a small array yields an empty split (no crash)."""
    from ps14 import evaluate as ev
    from ps14.datasets.windowing import WindowTensors

    n = 120
    X, X_future, y, y_exceed = _make_windows(n=n, lookback=8)
    t_index = np.arange(n).astype("datetime64[ns]")
    wt = WindowTensors(
        X, X_future, y, y_exceed, t_index,
        schema.FEATURE_COLUMNS, schema.KNOWN_FUTURE_COLUMNS, HORIZON_NAMES,
    )
    model = Climatology(n_tod_bins=4, n_kp_bins=3, n_doy_bins=3).fit(X, X_future, y, y_exceed)
    results = ev.evaluate(model, wt, split="test")  # default embargo 1296 -> empty
    # Empty split: metrics are NaN but the call must not raise.
    assert set(results) == set(HORIZON_NAMES)
    assert np.isnan(results["nowcast"]["rmse"])
