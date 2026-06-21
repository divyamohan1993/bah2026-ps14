"""Tests for preprocessing (clean, resample, align, transform, features).

Covers the Hampel despike, gap detection/interpolation, log10 floor (DONE primitives)
and the implemented layer: uniform resampling, L1->GEO align/merge (passes
``validate_merged``), train-only scalers (no leakage), and ``build_feature_matrix``
(yields exactly ``FEATURE_COLUMNS`` and passes ``validate_features``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ps14.datasets import schema
from ps14.features import offline
from ps14.preprocess import align, clean, resample, transform
from ps14.utils import timeops


# --------------------------------------------------------------------------------------
# Shared synthetic merged frame (built inline, independent of Builder-1's generator).
# --------------------------------------------------------------------------------------
def _make_merged(n: int = 1500, *, seed: int = 0) -> pd.DataFrame:
    """A minimal canonical MERGED frame with every required schema column."""
    idx = pd.date_range("2020-01-01", periods=n, freq="5min", name="time")
    k = np.arange(n)
    df = pd.DataFrame(index=idx)
    df["flux_e2"] = 10.0 ** (2.0 + 0.6 * np.sin(k / 50.0))
    df["log_flux_e2"] = np.log10(df["flux_e2"])
    df["flux_seed"] = 10.0 ** (1.0 + 0.3 * np.cos(k / 40.0))
    df["log_flux_seed"] = np.log10(df["flux_seed"])
    df["vsw"] = 400.0 + 60.0 * np.sin(k / 100.0)
    df["density"] = 5.0 + np.abs(np.sin(k / 30.0))
    df["pdyn"] = 2.0 + np.abs(np.cos(k / 35.0))
    df["bz_gsm"] = 3.0 * np.sin(k / 20.0)
    df["bt"] = 5.0 + np.abs(np.cos(k / 25.0))
    df["ae"] = 100.0 + 60.0 * np.abs(np.sin(k / 15.0))
    df["al"] = -df["ae"]
    df["kp"] = np.clip(3.0 + np.sin(k / 200.0), 0.0, 9.0)
    df["sym_h"] = -12.0 * np.abs(np.sin(k / 120.0))
    df["f107"] = 120.0 + 10.0 * np.sin(k / 300.0)
    df["mlt"] = (k * 5.0 / 60.0) % 24.0
    df["longitude"] = -75.0
    df["sat_id"] = pd.Categorical(["GOES-16"] * n)
    df["flux_e2_imputed"] = np.int8(0)
    return df


# --------------------------------------------------------------------------------------
# Hampel despike
# --------------------------------------------------------------------------------------


def test_hampel_flags_and_removes_spike():
    idx = pd.date_range("2020-01-01", periods=21, freq="5min", name="time")
    base = pd.Series(np.ones(21), index=idx)
    base.iloc[10] = 100.0  # a clear spike
    filtered, mask = clean.hampel_filter(base, window=7, n_sigma=3.0, replace="nan")
    assert bool(mask.iloc[10])
    assert np.isnan(filtered.iloc[10])
    # Non-spike points are untouched.
    assert filtered.iloc[0] == 1.0


def test_hampel_replace_median():
    idx = pd.date_range("2020-01-01", periods=21, freq="5min", name="time")
    s = pd.Series(np.ones(21), index=idx)
    s.iloc[5] = 50.0
    filtered, mask = clean.hampel_filter(s, replace="median")
    assert filtered.iloc[5] == pytest.approx(1.0)
    assert bool(mask.iloc[5])


def test_hampel_does_not_flag_linear_trend():
    # A locally linear signal is the realistic "clean" case: the centered rolling median
    # tracks the trend exactly, so deviations are ~0 and nothing is flagged.
    idx = pd.date_range("2020-01-01", periods=200, freq="5min", name="time")
    s = pd.Series(np.linspace(0.0, 10.0, 200), index=idx)
    _filtered, mask = clean.hampel_filter(s, window=7, n_sigma=3.0)
    assert mask.sum() == 0


def test_hampel_isolates_single_spike_on_trend():
    idx = pd.date_range("2020-01-01", periods=100, freq="5min", name="time")
    s = pd.Series(np.linspace(0.0, 10.0, 100), index=idx)
    s.iloc[50] += 100.0
    _filtered, mask = clean.hampel_filter(s, window=7, n_sigma=3.0)
    assert mask.sum() == 1 and bool(mask.iloc[50])


def test_hampel_constant_signal_never_flagged():
    idx = pd.date_range("2020-01-01", periods=50, freq="5min", name="time")
    s = pd.Series(np.full(50, 7.0), index=idx)
    _filtered, mask = clean.hampel_filter(s, window=7, n_sigma=3.0)
    assert mask.sum() == 0


def test_hampel_invalid_replace():
    s = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        clean.hampel_filter(s, replace="bogus")


# --------------------------------------------------------------------------------------
# Gap detection / interpolation
# --------------------------------------------------------------------------------------


def test_nan_run_lengths():
    s = pd.Series([1.0, np.nan, np.nan, 4.0, np.nan, 6.0])
    runs = clean.nan_run_lengths(s.isna())
    assert list(runs) == [0, 2, 2, 0, 1, 0]


def test_detect_gaps_short_vs_long():
    idx = pd.date_range("2020-01-01", periods=12, freq="5min", name="time")
    vals = [1.0, np.nan, np.nan, 4.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 12.0]
    s = pd.Series(vals, index=idx)
    short, long = clean.detect_gaps(s, max_gap_steps=3)
    # Run of 2 is short; run of 7 is long.
    assert short.iloc[1] and short.iloc[2]
    assert long.iloc[4] and long.iloc[8]
    assert not short.iloc[4]


def test_interpolate_short_gaps_only():
    idx = pd.date_range("2020-01-01", periods=10, freq="5min", name="time")
    vals = [0.0, np.nan, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 9.0]
    s = pd.Series(vals, index=idx)
    filled, imputed = clean.interpolate_short_gaps(s, max_gap_steps=2)
    # The single-sample gap at index 1 is filled; the long run stays NaN.
    assert filled.iloc[1] == pytest.approx(1.0)
    assert imputed.iloc[1] == 1
    assert np.isnan(filled.iloc[5])
    assert imputed.iloc[5] == 0


# --------------------------------------------------------------------------------------
# log10 floor transform
# --------------------------------------------------------------------------------------


def test_log10_floor_basic_and_floor():
    x = np.array([1.0, 10.0, 100.0, 0.0, -5.0, np.nan])
    out = transform.log10_floor(x, floor=0.01)
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(1.0)
    assert out[2] == pytest.approx(2.0)
    # Non-positives floored to log10(0.01) = -2.
    assert out[3] == pytest.approx(-2.0)
    assert out[4] == pytest.approx(-2.0)
    assert np.isnan(out[5])


def test_log10_floor_inverse_roundtrip():
    x = np.array([0.5, 5.0, 500.0])
    out = transform.log10_floor(x)
    back = transform.inverse_log10(out)
    np.testing.assert_allclose(back, x, rtol=1e-12)


def test_log10_floor_preserves_series():
    s = pd.Series([1.0, 10.0], index=pd.date_range("2020", periods=2, freq="5min"), name="flux_e2")
    out = transform.log10_floor(s)
    assert isinstance(out, pd.Series)
    assert out.name == "flux_e2"


# --------------------------------------------------------------------------------------
# Resampling -> uniform grid
# --------------------------------------------------------------------------------------


def test_resample_uniform_grid_is_regular():
    # Irregular, higher-rate samples within a 1-hour span.
    rng = np.random.default_rng(1)
    secs = np.sort(rng.integers(0, 3600, 90))
    times = pd.to_datetime("2020-01-01") + pd.to_timedelta(secs, unit="s")
    idx = pd.DatetimeIndex(times, name="time")
    df = pd.DataFrame({"vsw": rng.normal(400.0, 10.0, 90)}, index=idx)
    out = resample.resample_uniform(df, cadence="5min")
    deltas = out.index.to_series().diff().dropna().unique()
    assert len(deltas) == 1 and deltas[0] == pd.Timedelta("5min")
    assert out.index.is_monotonic_increasing and out.index.is_unique
    assert out.index.name == "time"


def test_resample_uniform_marks_missing_rows():
    # Two clusters of samples with an empty 5-min bin in between.
    idx = pd.DatetimeIndex(
        ["2020-01-01 00:00:30", "2020-01-01 00:01:00", "2020-01-01 00:11:00"], name="time"
    )
    df = pd.DataFrame({"vsw": [400.0, 410.0, 420.0]}, index=idx)
    out = resample.resample_uniform(df, cadence="5min")
    assert "row_missing" in out.columns
    # The 00:05 bin has no underlying sample -> flagged 1; the 00:00 bin -> 0.
    assert out.loc["2020-01-01 00:05:00", "row_missing"] == 1
    assert out.loc["2020-01-01 00:00:00", "row_missing"] == 0


def test_resample_uniform_empty_bins_are_nan():
    idx = pd.DatetimeIndex(["2020-01-01 00:00:00", "2020-01-01 00:30:00"], name="time")
    df = pd.DataFrame({"vsw": [400.0, 500.0]}, index=idx)
    out = resample.resample_uniform(df, cadence="5min")
    assert np.isnan(out.loc["2020-01-01 00:05:00", "vsw"])


def test_harmonize_renames_and_casts():
    idx = pd.DatetimeIndex(["2020-01-01"], name="t")
    df = pd.DataFrame({"V": [400], "flag": [1]}, index=idx)
    out = resample.harmonize(df, rename={"V": "vsw"}, dtypes={"vsw": "float64"})
    assert "vsw" in out.columns and "V" not in out.columns
    assert out["vsw"].dtype == np.float64
    assert out.index.name == "time"


# --------------------------------------------------------------------------------------
# L1 -> GEO shift + align/merge
# --------------------------------------------------------------------------------------


def test_shift_l1_to_geo_ballistic_lag_in_band():
    idx = pd.date_range("2020-01-01", periods=10, freq="5min", name="time")
    df = pd.DataFrame({"vsw": np.full(10, 500.0)}, index=idx)
    lag = timeops.shift_l1_to_geo(df, method="ballistic", return_lag=True)
    # dx/Vsw = 1.5e6 km / 500 km/s / 60 = 50 min, inside the documented 20-90 min band.
    assert np.allclose(lag, 50.0)
    assert (lag >= 20.0).all() and (lag <= 90.0).all()


def test_shift_l1_to_geo_omni_is_noop():
    idx = pd.date_range("2020-01-01", periods=5, freq="5min", name="time")
    df = pd.DataFrame({"vsw": [400.0] * 5}, index=idx)
    out = timeops.shift_l1_to_geo(df, method="omni_preshifted")
    assert out.index.equals(df.index)


def test_shift_l1_to_geo_unknown_method_raises():
    df = pd.DataFrame({"vsw": [400.0]}, index=pd.DatetimeIndex(["2020-01-01"], name="time"))
    with pytest.raises(ValueError):
        timeops.shift_l1_to_geo(df, method="bogus")


def _split_geo_l1(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    geo_cols = [
        "flux_e2",
        "log_flux_e2",
        "flux_seed",
        "log_flux_seed",
        "mlt",
        "longitude",
        "sat_id",
        "flux_e2_imputed",
    ]
    l1_cols = ["vsw", "density", "pdyn", "bz_gsm", "bt", "ae", "al", "kp", "sym_h", "f107"]
    return df[geo_cols].copy(), df[l1_cols].copy()


def test_align_omni_passes_validate_merged():
    merged_src = _make_merged()
    geo, l1 = _split_geo_l1(merged_src)
    merged = align.align_l1_to_geo(geo, l1, method="omni", cadence="5min")
    # No exception -> valid canonical merged frame (CONTRACTS.md §2).
    assert schema.validate_merged(merged, raise_on_error=False) == []
    assert merged.index.name == "time"


def test_align_ballistic_passes_validate_merged():
    merged_src = _make_merged()
    geo, l1 = _split_geo_l1(merged_src)
    merged = align.align_l1_to_geo(geo, l1, method="ballistic", cadence="5min")
    assert schema.validate_merged(merged, raise_on_error=False) == []
    # Ballistic shift moves drivers forward in time -> a few head rows are not co-covered.
    assert len(merged) <= len(merged_src)


def test_merge_sources_unifies_grid():
    merged_src = _make_merged(n=200)
    geo, l1 = _split_geo_l1(merged_src)
    out = align.merge_sources({"geo": geo, "l1": l1}, how="inner")
    assert out.index.name == "time"
    for col in ("vsw", "log_flux_e2"):
        assert col in out.columns


# --------------------------------------------------------------------------------------
# Scalers — TRAIN-only fit (no leakage)
# --------------------------------------------------------------------------------------


def test_fit_scaler_uses_train_only_no_leakage():
    df = _make_merged()
    cols = ["vsw", "log_flux_e2", "ae"]
    half = len(df) // 2
    train = df.iloc[:half]
    state = transform.fit_scaler(train, cols, kind="standard")
    # The fitted center/scale must come ONLY from the train slice (leakage check).
    assert np.isclose(state["center"][0], train["vsw"].mean())
    assert np.isclose(state["scale"][0], train["vsw"].std(ddof=0))
    # Crucially, they must NOT equal the full-series statistics (would be leakage).
    assert not np.isclose(state["center"][0], df["vsw"].mean())


def test_apply_and_inverse_scaler_roundtrip():
    df = _make_merged()
    cols = ["vsw", "density", "bt"]
    state = transform.fit_scaler(df.iloc[: len(df) // 2], cols, kind="robust")
    scaled = transform.apply_scaler(df, state)
    restored = transform.inverse_scaler(scaled, state)
    for c in cols:
        np.testing.assert_allclose(restored[c].to_numpy(), df[c].to_numpy(), rtol=1e-9, atol=1e-9)


def test_apply_scaler_train_zscore_is_standardized():
    df = _make_merged()
    cols = ["vsw"]
    half = len(df) // 2
    state = transform.fit_scaler(df.iloc[:half], cols, kind="standard")
    scaled_train = transform.apply_scaler(df.iloc[:half], state)
    # On the TRAIN slice a standard scaler yields ~0 mean / ~1 std.
    assert abs(scaled_train["vsw"].mean()) < 1e-9
    assert abs(scaled_train["vsw"].std(ddof=0) - 1.0) < 1e-9


def test_scaler_persist_roundtrip(tmp_path):
    df = _make_merged()
    cols = ["vsw", "ae"]
    state = transform.fit_scaler(df, cols)
    path = tmp_path / "scaler_train.joblib"
    transform.save_scaler(state, path)
    loaded = transform.load_scaler(path)
    assert loaded["columns"] == state["columns"]
    assert np.allclose(loaded["center"], state["center"])
    assert np.allclose(loaded["scale"], state["scale"])


def test_fit_scaler_invalid_kind_raises():
    df = _make_merged(n=50)
    with pytest.raises(ValueError):
        transform.fit_scaler(df, ["vsw"], kind="bogus")


# --------------------------------------------------------------------------------------
# Feature matrix assembly
# --------------------------------------------------------------------------------------


def test_build_feature_matrix_exact_feature_columns():
    df = _make_merged()
    feat = offline.build_feature_matrix(df, None)
    # Every canonical feature + known-future column is present...
    for col in schema.FEATURE_COLUMNS + schema.KNOWN_FUTURE_COLUMNS:
        assert col in feat.columns, f"missing {col}"
    # ...and no stray rolling columns leaked beyond the schema set.
    roll_like = [c for c in feat.columns if "_roll" in c]
    assert set(roll_like) == set(schema.ROLLING_COLUMNS)


def test_build_feature_matrix_passes_validate_features():
    df = _make_merged()
    feat = offline.build_feature_matrix(df, None)
    assert schema.validate_features(feat, raise_on_error=False) == []


def test_build_feature_matrix_no_target_leakage_in_lags():
    df = _make_merged()
    feat = offline.build_feature_matrix(df, None)
    # log_flux_e2_lag_1[t] must equal log_flux_e2[t-1] (strictly past, no look-ahead).
    lag1 = feat["log_flux_e2_lag_1"].to_numpy()
    base = feat["log_flux_e2"].to_numpy()
    np.testing.assert_allclose(lag1[1:], base[:-1], rtol=1e-12)
    assert np.isnan(lag1[0])  # nothing before the first sample


def test_build_feature_matrix_rolling_is_trailing_only():
    df = _make_merged()
    feat = offline.build_feature_matrix(df, None)
    # The trailing rollmean at t equals the mean of the last 12 base values up to t.
    rm = feat["log_flux_e2_rollmean_12"].to_numpy()
    base = feat["log_flux_e2"].to_numpy()
    t = 500
    assert np.isclose(rm[t], base[t - 11 : t + 1].mean())


def test_add_coupling_functions_derives_pdyn_when_missing():
    df = _make_merged(n=100)
    df["pdyn"] = np.nan
    out = offline.add_coupling_functions(df)
    assert out["pdyn"].notna().any()
    for col in schema.COUPLING_COLUMNS:
        assert col in out.columns
