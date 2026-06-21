"""Tests for preprocessing.

The Hampel despike, gap detection/interpolation, and log10 floor are FULLY IMPLEMENTED
and tested here; the resample/align/scaler functions are stubs (asserted to raise until
implemented).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ps14.preprocess import clean, transform

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
# Stubs (contract markers)
# --------------------------------------------------------------------------------------


def test_scaler_stub_raises():
    df = pd.DataFrame({"vsw": [1.0, 2.0, 3.0]})
    with pytest.raises(NotImplementedError):
        transform.fit_scaler(df, ["vsw"])
