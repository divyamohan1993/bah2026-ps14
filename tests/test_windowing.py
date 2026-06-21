"""Tests for supervised windowing + chronological split (ps14.datasets.windowing).

Covers ``chronological_split`` (leakage-free ordering + embargo) and ``make_supervised``:
tensor shapes, no future leakage (X ends at t, y at t+horizon), binary ``y_exceed`` that
matches ``flux >= 1000``, long-gap NaN dropping, and a save/load NPZ round-trip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ps14.constants import HARSH_PFU, HORIZON_STEPS, LOG_HARSH
from ps14.datasets import schema, windowing


def _make_feature_df(n: int = 1600, *, ramp_target: bool = False) -> pd.DataFrame:
    """A feature-matrix-shaped frame (FEATURE + KNOWN_FUTURE cols) for windowing tests.

    With ``ramp_target`` the target ramps across ``log10(1000)=3.0`` so exceedance labels
    take both values; otherwise the target is a distinct per-row value so the alignment
    between ``X[-1]`` / ``y[h]`` and absolute time can be verified exactly.
    """
    idx = pd.date_range("2020-01-01", periods=n, freq="5min", name="time")
    df = pd.DataFrame(index=idx)
    if ramp_target:
        df["log_flux_e2"] = np.linspace(2.0, 4.0, n)
    else:
        df["log_flux_e2"] = np.arange(n, dtype="float64") * 1e-3
    for c in schema.FEATURE_COLUMNS:
        if c not in df.columns:
            df[c] = np.arange(n, dtype="float64") * 1e-4 + (hash(c) % 5)
    for c in schema.KNOWN_FUTURE_COLUMNS:
        df[c] = np.sin(np.arange(n) + (hash(c) % 7))
    return df


def test_chronological_split_order_and_embargo():
    n = 1000
    t_index = pd.date_range("2020-01-01", periods=n, freq="5min")
    embargo = 50
    train_idx, val_idx, test_idx = windowing.chronological_split(
        t_index, train=0.7, val=0.15, embargo_steps=embargo
    )
    # Strictly increasing, non-overlapping segments.
    assert train_idx.max() < val_idx.min()
    assert val_idx.max() < test_idx.min()
    # Embargo gap of at least `embargo` between segments.
    assert val_idx.min() - train_idx.max() >= embargo
    assert test_idx.min() - val_idx.max() >= embargo
    # Test segment reaches the end.
    assert test_idx.max() == n - 1


def test_chronological_split_no_leakage_indices_disjoint():
    n = 500
    t_index = pd.date_range("2020-01-01", periods=n, freq="5min")
    train_idx, val_idx, test_idx = windowing.chronological_split(t_index, embargo_steps=20)
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    # No index appears twice across splits.
    assert len(all_idx) == len(set(all_idx.tolist()))


def test_chronological_split_empty():
    train_idx, val_idx, test_idx = windowing.chronological_split(np.array([]), embargo_steps=10)
    assert len(train_idx) == len(val_idx) == len(test_idx) == 0


def test_chronological_split_large_embargo_yields_empty_val():
    n = 100
    t_index = pd.date_range("2020-01-01", periods=n, freq="5min")
    # Embargo larger than the val band collapses it but must not error or overlap.
    train_idx, val_idx, test_idx = windowing.chronological_split(
        t_index, train=0.7, val=0.15, embargo_steps=80
    )
    if len(val_idx) and len(train_idx):
        assert val_idx.min() > train_idx.max()


def test_window_tensors_dataclass_fields():
    # The container exposes the contract fields (CONTRACTS.md §4).
    wt = windowing.WindowTensors(
        X=np.zeros((2, 3, 4), dtype="float32"),
        X_future=np.zeros((2, 5, 6), dtype="float32"),
        y=np.zeros((2, 3), dtype="float32"),
        y_exceed=np.zeros((2, 3), dtype="float32"),
        t_index=np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"),
        feature_cols=schema.FEATURE_COLUMNS,
        known_future_cols=schema.KNOWN_FUTURE_COLUMNS,
        horizon_names=windowing.HORIZON_NAMES_DEFAULT,
    )
    assert wt.X.shape == (2, 3, 4)
    assert wt.y_exceed.shape == (2, 3)


def test_make_supervised_shapes():
    n, L = 1600, 200
    df = _make_feature_df(n)
    wt = windowing.make_supervised(
        df,
        schema.FEATURE_COLUMNS,
        schema.KNOWN_FUTURE_COLUMNS,
        schema.TARGET,
        lookback=L,
        decoder_steps=144,
        horizon_steps=HORIZON_STEPS,
    )
    n_win = wt.X.shape[0]
    assert wt.X.shape == (n_win, L, len(schema.FEATURE_COLUMNS))
    assert wt.X_future.shape == (n_win, 144, len(schema.KNOWN_FUTURE_COLUMNS))
    assert wt.y.shape == (n_win, len(HORIZON_STEPS))
    assert wt.y_exceed.shape == (n_win, len(HORIZON_STEPS))
    assert wt.t_index.shape == (n_win,)
    # Dtypes per CONTRACTS.md §4.
    assert wt.X.dtype == np.float32
    assert wt.y.dtype == np.float32
    assert wt.t_index.dtype == np.dtype("datetime64[ns]")
    assert wt.horizon_names == list(HORIZON_STEPS.keys())


def test_make_supervised_no_future_leakage_alignment():
    n, L = 1600, 200
    df = _make_feature_df(n)
    wt = windowing.make_supervised(
        df,
        schema.FEATURE_COLUMNS,
        schema.KNOWN_FUTURE_COLUMNS,
        schema.TARGET,
        lookback=L,
        decoder_steps=144,
        horizon_steps=HORIZON_STEPS,
    )
    tcol = schema.FEATURE_COLUMNS.index("log_flux_e2")
    base = df["log_flux_e2"].to_numpy()
    i = 7
    t_pos = df.index.get_loc(pd.Timestamp(wt.t_index[i]))
    # X[i, -1] is exactly the value AT t (nothing past t leaks into X).
    assert np.isclose(wt.X[i, -1, tcol], base[t_pos])
    # X[i, 0] is the value at t-L+1 (window spans [t-L+1 .. t]).
    assert np.isclose(wt.X[i, 0, tcol], base[t_pos - L + 1])
    # y[i, h] is the target strictly in the future at t + horizon_steps[h].
    for hi, hsteps in enumerate(HORIZON_STEPS.values()):
        assert np.isclose(wt.y[i, hi], base[t_pos + hsteps])


def test_make_supervised_xfuture_is_strictly_future():
    n, L = 1200, 150
    df = _make_feature_df(n)
    wt = windowing.make_supervised(
        df,
        schema.FEATURE_COLUMNS,
        schema.KNOWN_FUTURE_COLUMNS,
        schema.TARGET,
        lookback=L,
        decoder_steps=144,
    )
    kf0 = schema.KNOWN_FUTURE_COLUMNS[0]
    col = df[kf0].to_numpy()
    i = 3
    t_pos = df.index.get_loc(pd.Timestamp(wt.t_index[i]))
    # X_future[i, 0] is the covariate at t+1 (first strictly-future step).
    assert np.isclose(wt.X_future[i, 0, 0], col[t_pos + 1])
    assert np.isclose(wt.X_future[i, -1, 0], col[t_pos + 144])


def test_make_supervised_y_exceed_binary_matches_threshold():
    n, L = 1000, 120
    df = _make_feature_df(n, ramp_target=True)
    wt = windowing.make_supervised(
        df,
        schema.FEATURE_COLUMNS,
        schema.KNOWN_FUTURE_COLUMNS,
        schema.TARGET,
        lookback=L,
        decoder_steps=144,
        horizon_steps=HORIZON_STEPS,
    )
    # Binary {0,1}.
    assert set(np.unique(wt.y_exceed).tolist()).issubset({0.0, 1.0})
    # Matches log-space threshold and the equivalent linear flux >= 1000 pfu.
    assert np.array_equal(wt.y_exceed, (wt.y >= LOG_HARSH).astype("float32"))
    assert np.array_equal(wt.y_exceed, (10.0**wt.y >= HARSH_PFU).astype("float32"))
    # The ramp crosses the threshold, so both classes are present.
    assert wt.y_exceed.min() == 0.0 and wt.y_exceed.max() == 1.0


def test_make_supervised_drops_long_gap_nan_windows():
    n, L = 1000, 100
    df = _make_feature_df(n)
    wt_full = windowing.make_supervised(
        df, schema.FEATURE_COLUMNS, schema.KNOWN_FUTURE_COLUMNS, schema.TARGET, lookback=L
    )
    df_gap = df.copy()
    df_gap.iloc[400, df_gap.columns.get_loc("vsw")] = np.nan
    wt_gap = windowing.make_supervised(
        df_gap, schema.FEATURE_COLUMNS, schema.KNOWN_FUTURE_COLUMNS, schema.TARGET, lookback=L
    )
    # A single NaN feature drops exactly the L windows whose lookback covers it.
    assert wt_full.X.shape[0] - wt_gap.X.shape[0] == L
    # No NaN survives in the kept windows.
    assert not np.isnan(wt_gap.X).any()
    assert not np.isnan(wt_gap.y).any()


def test_make_supervised_save_load_roundtrip(tmp_path):
    df = _make_feature_df(900)
    wt = windowing.make_supervised(
        df,
        schema.FEATURE_COLUMNS,
        schema.KNOWN_FUTURE_COLUMNS,
        schema.TARGET,
        lookback=120,
        decoder_steps=144,
    )
    path = tmp_path / "windows.npz"
    windowing.save_windows(wt, path)
    loaded = windowing.load_windows(path)
    assert np.array_equal(loaded.X, wt.X)
    assert np.array_equal(loaded.X_future, wt.X_future)
    assert np.array_equal(loaded.y, wt.y)
    assert np.array_equal(loaded.y_exceed, wt.y_exceed)
    assert np.array_equal(loaded.t_index, wt.t_index)
    assert loaded.feature_cols == wt.feature_cols
    assert loaded.known_future_cols == wt.known_future_cols
    assert loaded.horizon_names == wt.horizon_names


def test_chronological_split_embargo_respected_on_windows():
    # End-to-end: build windows, then split with an embargo >= lookback + max horizon.
    df = _make_feature_df(1600)
    L = 200
    wt = windowing.make_supervised(
        df, schema.FEATURE_COLUMNS, schema.KNOWN_FUTURE_COLUMNS, schema.TARGET, lookback=L
    )
    embargo = L + max(HORIZON_STEPS.values())
    train_idx, val_idx, test_idx = windowing.chronological_split(
        wt.t_index, train=0.7, val=0.15, embargo_steps=embargo
    )
    # Disjoint, ordered, and embargo gap honoured between consecutive segments.
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    assert len(all_idx) == len(set(all_idx.tolist()))
    if len(train_idx) and len(val_idx):
        assert val_idx.min() - train_idx.max() >= embargo
    if len(val_idx) and len(test_idx):
        assert test_idx.min() - val_idx.max() >= embargo
