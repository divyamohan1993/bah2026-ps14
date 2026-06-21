"""Tests for supervised windowing + chronological split (ps14.datasets.windowing).

``chronological_split`` is FULLY IMPLEMENTED and tested for leakage-free ordering +
embargo; ``make_supervised`` is a stub asserted to raise until implemented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ps14.datasets import schema, windowing


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


def test_make_supervised_stub_raises():
    idx = pd.date_range("2020-01-01", periods=50, freq="5min", name="time")
    df = pd.DataFrame({c: np.zeros(50) for c in schema.FEATURE_COLUMNS}, index=idx)
    for c in schema.KNOWN_FUTURE_COLUMNS:
        df[c] = 0.0
    with pytest.raises(NotImplementedError):
        windowing.make_supervised(
            df, schema.FEATURE_COLUMNS, schema.KNOWN_FUTURE_COLUMNS, schema.TARGET
        )
