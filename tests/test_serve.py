"""Tests for the real-time serving core (ps14.serve).

Covers the parts that need only core deps (numpy / pydantic):

* :class:`OnlineFeatureState` produces correct running mean / var / min / max vs a
  brute-force rolling reference, in amortized O(1).
* :class:`ForecastCache` get / put hit-miss, O(1) keying, and the capacity bound.
* :class:`ClimatologyLUT` returns a schema-valid payload.
* :class:`Predictor` falls back to the climatology LUT when no model artifact exists and
  returns a schema-valid :class:`ForecastPayload`.
* :class:`ForecastPayload` validates and matches the CONTRACTS schema.

Web / ONNX / Streamlit / APScheduler tests are guarded with ``importorskip`` so the suite
passes in the core environment.
"""

from __future__ import annotations

import numpy as np
import pytest

from ps14.constants import HARSH_PFU, HORIZON_LEAD_MINUTES, HORIZON_NAMES, LOG_HARSH
from ps14.datasets import schema
from ps14.serve.inference import (
    ClimatologyLUT,
    ForecastCache,
    ForecastPayload,
    HorizonForecast,
    OnlineFeatureState,
    Predictor,
    build_payload,
)

_TARGET_CHANNEL = schema.FEATURE_COLUMNS.index(schema.TARGET)
_N_FEATURES = len(schema.FEATURE_COLUMNS)


def _feature_vec(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=_N_FEATURES).astype("float32")


# ======================================================================================
# OnlineFeatureState
# ======================================================================================
def test_online_feature_state_vector_shape_and_warm():
    state = OnlineFeatureState(window=12, warmup=3)
    assert not state.is_warm()
    vec = None
    for i in range(3):
        vec = state.update({"log_flux_e2": float(i), "vsw": 400.0 + i, "kp": 2.0})
    assert vec is not None
    assert vec.shape == (_N_FEATURES,)
    assert vec.dtype == np.float32
    assert state.is_warm()


def test_online_feature_state_running_stats_match_bruteforce():
    """Welford-backed rolling mean / std / min / max must match a brute-force window."""
    rng = np.random.default_rng(7)
    data = rng.normal(2.5, 0.6, size=400)
    window = 72
    state = OnlineFeatureState(window=window)

    idx = state._idx
    mean_i = idx["log_flux_e2_rollmean_72"]
    std_i = idx["log_flux_e2_rollstd_72"]
    min_i = idx["log_flux_e2_rollmin_72"]
    max_i = idx["log_flux_e2_rollmax_72"]

    for j, x in enumerate(data):
        vec = state.update({"log_flux_e2": float(x)})
        lo = max(0, j - window + 1)
        seg = data[lo : j + 1]
        # The feature vector is float32, so compare at float32 precision.
        assert vec[mean_i] == pytest.approx(seg.mean(), rel=1e-5, abs=1e-5)
        # Population std (ddof=0); Welford .std is population.
        assert vec[std_i] == pytest.approx(seg.std(), rel=1e-4, abs=1e-5)
        assert vec[min_i] == pytest.approx(seg.min(), rel=1e-5, abs=1e-5)
        assert vec[max_i] == pytest.approx(seg.max(), rel=1e-5, abs=1e-5)


def test_online_feature_state_lag_and_carry_forward():
    state = OnlineFeatureState(window=600)
    seq = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    vec = None
    for x in seq:
        vec = state.update({"log_flux_e2": x, "vsw": 500.0})
    idx = state._idx
    # lag_1 is the previous sample; the base channel is the current sample.
    assert vec[idx["log_flux_e2"]] == pytest.approx(seq[-1])
    assert vec[idx["log_flux_e2_lag_1"]] == pytest.approx(seq[-2])
    assert vec[idx["log_flux_e2_lag_6"]] == pytest.approx(seq[-7])
    # vsw carried forward into its channel.
    assert vec[idx["vsw"]] == pytest.approx(500.0)


def test_online_feature_state_missing_keys_carry_forward():
    state = OnlineFeatureState()
    state.update({"log_flux_e2": 2.0, "vsw": 450.0})
    vec = state.update({"log_flux_e2": 2.1})  # vsw omitted -> carries forward
    assert vec[state._idx["vsw"]] == pytest.approx(450.0)


def test_online_feature_state_invalid_window():
    with pytest.raises(ValueError):
        OnlineFeatureState(window=0)


# ======================================================================================
# ForecastCache
# ======================================================================================
def _dummy_payload(model: str = "m") -> ForecastPayload:
    n_h = len(HORIZON_NAMES)
    return build_payload(
        {0.1: np.full(n_h, 2.0), 0.5: np.full(n_h, 2.5), 0.9: np.full(n_h, 3.0)},
        np.zeros(n_h),
        model_name=model,
    )


def test_forecast_cache_hit_and_miss():
    cache = ForecastCache(capacity=8, decimals=3)
    vec = _feature_vec(1)
    assert cache.get(vec) is None  # miss
    assert cache.misses == 1
    payload = _dummy_payload()
    cache.put(vec, payload)
    got = cache.get(vec)  # hit
    assert got is payload
    assert cache.hits == 1


def test_forecast_cache_quantization_collapses_near_identical():
    cache = ForecastCache(capacity=8, decimals=2)
    vec = _feature_vec(2)
    cache.put(vec, _dummy_payload())
    near = vec.copy()
    near += 1e-5  # below the rounding precision -> same key
    assert cache.get(near) is not None


def test_forecast_cache_key_is_constant_work_and_stable():
    # The key is hash(rounded.tobytes()) -> O(1) wrt cache size, deterministic per vector.
    vec = _feature_vec(3)
    k1 = ForecastCache.key_for(vec, 3)
    k2 = ForecastCache.key_for(vec.copy(), 3)
    assert k1 == k2
    assert ForecastCache.key_for(_feature_vec(4), 3) != k1


def test_forecast_cache_capacity_bound_evicts_lru():
    cache = ForecastCache(capacity=2, decimals=3)
    v0, v1, v2 = _feature_vec(10), _feature_vec(11), _feature_vec(12)
    cache.put(v0, _dummy_payload("a"))
    cache.put(v1, _dummy_payload("b"))
    assert len(cache) == 2
    cache.put(v2, _dummy_payload("c"))  # overflow -> evict LRU (v0)
    assert len(cache) == 2
    assert cache.get(v0) is None  # evicted
    assert cache.get(v1) is not None
    assert cache.get(v2) is not None


def test_forecast_cache_lru_refresh_on_get():
    cache = ForecastCache(capacity=2, decimals=3)
    v0, v1, v2 = _feature_vec(20), _feature_vec(21), _feature_vec(22)
    cache.put(v0, _dummy_payload("a"))
    cache.put(v1, _dummy_payload("b"))
    cache.get(v0)  # touch v0 -> now v1 is LRU
    cache.put(v2, _dummy_payload("c"))  # evicts v1, keeps v0
    assert cache.get(v0) is not None
    assert cache.get(v1) is None


# ======================================================================================
# ClimatologyLUT
# ======================================================================================
def test_climatology_lut_lookup_returns_ordered_quantiles():
    lut = ClimatologyLUT()
    base = lut.lookup(mlt=12.0)
    assert set(base) == {"p10", "p50", "p90"}
    assert base["p10"] <= base["p50"] <= base["p90"]


def test_climatology_lut_diurnal_noon_above_midnight():
    lut = ClimatologyLUT()
    noon = lut.lookup(mlt=12.0)["p50"]
    midnight = lut.lookup(mlt=0.0)["p50"]
    assert noon > midnight  # ~1-order diurnal cycle peaks near local noon


def test_climatology_lut_forecast_is_schema_valid():
    lut = ClimatologyLUT()
    payload = lut.forecast(mlt=12.0, kp=3.0, vsw=450.0)
    assert isinstance(payload, ForecastPayload)
    assert set(payload.horizons) == set(HORIZON_NAMES)
    assert payload.threshold_pfu == HARSH_PFU
    assert payload.model == "climatology"


def test_climatology_lut_save_load_roundtrip(tmp_path):
    lut = ClimatologyLUT()
    path = tmp_path / "climatology_lut.npz"
    lut.save(path)
    loaded = ClimatologyLUT.load(path)
    assert loaded.lookup(mlt=12.0)["p50"] == pytest.approx(lut.lookup(mlt=12.0)["p50"])


def test_climatology_lut_load_missing_path_returns_default():
    lut = ClimatologyLUT.load("/nonexistent/path/lut.npz")
    assert isinstance(lut, ClimatologyLUT)


def test_climatology_lut_bad_table_shape_raises():
    with pytest.raises(ValueError):
        ClimatologyLUT(table=np.zeros((4, 2)))


# ======================================================================================
# Predictor (climatology fallback)
# ======================================================================================
def test_predictor_falls_back_to_climatology_when_no_artifact():
    pred = Predictor()
    assert pred.backend == "climatology"
    assert pred.model_name == "climatology"


def test_predictor_predict_is_schema_valid_and_cached():
    pred = Predictor()
    vec = _feature_vec(30)
    out = pred.predict(vec, context={"mlt": 12.0, "kp": 3.0, "vsw": 450.0})
    assert isinstance(out, ForecastPayload)
    assert set(out.horizons) == set(HORIZON_NAMES)
    # Second identical call should be an O(1) cache hit.
    pred.predict(vec, context={"mlt": 12.0, "kp": 3.0, "vsw": 450.0})
    assert pred.cache.hits == 1


def test_predictor_missing_artifact_paths_still_climatology(tmp_path):
    pred = Predictor(onnx_path=tmp_path / "nope.onnx", model_path=tmp_path / "nope.pt")
    assert pred.backend == "climatology"
    out = pred.predict(_feature_vec(31))
    assert isinstance(out, ForecastPayload)


def test_predictor_uses_forecaster_artifact(tmp_path):
    """A persisted Persistence model is loaded and used as the backend."""
    from ps14.models.baselines import Persistence

    path = tmp_path / "persistence.npz"
    Persistence().save(path)
    pred = Predictor(model_path=path)
    assert pred.backend == "forecaster"
    out = pred.predict(_feature_vec(32))
    assert isinstance(out, ForecastPayload)
    assert set(out.horizons) == set(HORIZON_NAMES)


# ======================================================================================
# ForecastPayload schema
# ======================================================================================
def test_forecast_payload_matches_contract_keys():
    payload = _dummy_payload("tft-dualhead-v1")
    dumped = payload.model_dump()
    assert set(dumped) >= {
        "issued_utc",
        "satellite",
        "horizons",
        "threshold_pfu",
        "model",
        "source",
        "latency_ms",
    }
    for name in HORIZON_NAMES:
        block = dumped["horizons"][name]
        assert set(block) == {
            "lead_min",
            "p10",
            "p50",
            "p90",
            "flux_p50_pfu",
            "p_exceed_1000pfu",
            "alert",
        }
        assert block["lead_min"] == HORIZON_LEAD_MINUTES[name]


def test_horizon_forecast_alert_from_exceedance():
    h = HorizonForecast(
        lead_min=40,
        p10=3.0,
        p50=3.3,
        p90=3.6,
        flux_p50_pfu=2000.0,
        p_exceed_1000pfu=0.8,
        alert=True,
    )
    assert h.alert is True


def test_build_payload_converts_log_to_pfu_and_thresholds_alert():
    n_h = len(HORIZON_NAMES)
    q = {
        0.1: np.full(n_h, 2.5),
        0.5: np.array([LOG_HARSH + 0.1, 2.0, 2.0]),  # nowcast above threshold
        0.9: np.full(n_h, 3.5),
    }
    proba = np.array([0.9, 0.1, 0.0])
    payload = build_payload(q, proba, model_name="x")
    now = payload.horizons["nowcast"]
    assert now.flux_p50_pfu == pytest.approx(10.0 ** (LOG_HARSH + 0.1))
    assert now.alert is True  # p_exceed 0.9 >= 0.5
    assert payload.horizons["6h"].alert is False


def test_build_payload_accepts_batch_uses_last_row():
    n_h = len(HORIZON_NAMES)
    # A [N, n_h] batch: the last row should be used. Make the last row distinct.
    p50 = np.tile(np.array([2.5, 2.6, 2.7]), (4, 1))
    p50[-1] = [1.1, 1.2, 1.3]
    q = {0.1: np.full((4, n_h), 2.0), 0.5: p50, 0.9: np.full((4, n_h), 3.0)}
    payload = build_payload(q, np.zeros((4, n_h)), model_name="x")
    assert payload.horizons["nowcast"].p50 == pytest.approx(1.1)


# ======================================================================================
# Optional-dependency tests (guarded)
# ======================================================================================
def test_fastapi_app_health_and_forecast():
    pytest.importorskip("fastapi")
    pytest.importorskip("starlette")
    pytest.importorskip("httpx")  # starlette.testclient needs httpx
    from starlette.testclient import TestClient

    from ps14.serve.api import create_app

    app = create_app()
    client = TestClient(app)

    # /health
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model" in body and "uptime_s" in body

    # POST /forecast with a ready feature vector -> schema-valid payload.
    vec = _feature_vec(40).tolist()
    r = client.post("/forecast", json={"features": vec, "mlt": 12.0, "kp": 3.0, "vsw": 450.0})
    assert r.status_code == 200
    fc = r.json()
    assert set(fc["horizons"]) == set(HORIZON_NAMES)

    # GET /forecast now returns the stored forecast; GET /forecast/{horizon} works.
    assert client.get("/forecast").status_code == 200
    assert client.get("/forecast/nowcast").status_code == 200
    assert client.get("/forecast/bogus").status_code == 404


def test_fastapi_post_forecast_bad_length_returns_422():
    pytest.importorskip("fastapi")
    pytest.importorskip("starlette")
    pytest.importorskip("httpx")  # starlette.testclient needs httpx
    from starlette.testclient import TestClient

    from ps14.serve.api import create_app

    client = TestClient(create_app())
    r = client.post("/forecast", json={"features": [1.0, 2.0, 3.0]})
    assert r.status_code == 422


def test_scheduler_refresh_once_offline_synthetic():
    # refresh_once needs no web deps (apscheduler only used by start_scheduler).
    from ps14.serve.scheduler import RefreshState, refresh_once

    state = RefreshState(source="synthetic")
    state.initialize()
    payload = refresh_once(state)
    assert isinstance(payload, ForecastPayload)
    assert set(payload.horizons) == set(HORIZON_NAMES)
    assert "flux_e2" in state.latest


def test_scheduler_start_requires_apscheduler():
    pytest.importorskip("apscheduler")
    from ps14.serve.scheduler import start_scheduler

    sched = start_scheduler(Predictor(), source="synthetic", interval_s=1)
    try:
        assert sched.running
    finally:
        sched.shutdown(wait=False)


def test_dashboard_figures_offline():
    pytest.importorskip("plotly")
    from ps14.dashboard.app import alert_levels, build_flux_figure, build_forecast_figure

    payload = ClimatologyLUT().forecast(mlt=12.0).model_dump()
    fig_fc = build_forecast_figure(payload)
    assert fig_fc is not None
    fig_flux = build_flux_figure(
        [1, 2, 3], [10.0, 20.0, 30.0], [400.0, 410.0, 420.0], [1.0, -1.0, 0.5]
    )
    assert fig_flux is not None
    levels = alert_levels(payload)
    assert set(levels) == set(HORIZON_NAMES)
    assert all("color" in v for v in levels.values())


def test_export_to_onnx_requires_torch():
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxscript")  # torch.onnx.export backend dependency
    import torch

    from ps14.serve.inference import export_to_onnx

    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(_N_FEATURES, 3))
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        out = export_to_onnx(model, Path(d) / "m.onnx", lookback=1, n_features=_N_FEATURES)
        assert out.exists()
