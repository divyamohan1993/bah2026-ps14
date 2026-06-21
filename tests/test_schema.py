"""Real tests for the canonical schema + validators (ps14.datasets.schema)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ps14.constants import LOG_HARSH
from ps14.datasets import schema


def _minimal_merged(n: int = 20) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="5min", name="time")
    rng = np.random.default_rng(0)
    data = {col: rng.normal(size=n) for col in schema.MERGED_REQUIRED}
    df = pd.DataFrame(data, index=idx)
    df["mlt"] = rng.uniform(0, 24, size=n)
    df["kp"] = rng.uniform(0, 9, size=n)
    df["flux_e2"] = np.abs(df["flux_e2"]) + 0.1
    df["flux_seed"] = np.abs(df["flux_seed"]) + 0.1
    return df


def test_constants_present_and_consistent():
    assert LOG_HARSH == pytest.approx(3.0)
    assert schema.TARGET == "log_flux_e2"
    # Feature columns are unique and include the autoregressive target + a coupling fn.
    assert len(schema.FEATURE_COLUMNS) == len(set(schema.FEATURE_COLUMNS))
    assert "log_flux_e2" in schema.FEATURE_COLUMNS
    assert "newell" in schema.FEATURE_COLUMNS
    assert set(schema.KNOWN_FUTURE_COLUMNS) == {
        "tod_sin",
        "tod_cos",
        "doy_sin",
        "doy_cos",
        "mlt_sin",
        "mlt_cos",
    }


def test_validate_merged_accepts_minimal():
    df = _minimal_merged()
    problems = schema.validate_merged(df, raise_on_error=False)
    assert problems == []


def test_validate_merged_rejects_bad_index():
    df = _minimal_merged().reset_index(drop=True)  # drop the DatetimeIndex
    with pytest.raises(schema.SchemaError):
        schema.validate_merged(df)


def test_validate_merged_rejects_missing_column():
    df = _minimal_merged().drop(columns=["vsw"])
    problems = schema.validate_merged(df, raise_on_error=False)
    assert any("vsw" in p for p in problems)


def test_validate_merged_rejects_out_of_range_mlt_and_inf():
    df = _minimal_merged()
    df.loc[df.index[0], "mlt"] = 30.0
    df.loc[df.index[1], "vsw"] = np.inf
    problems = schema.validate_merged(df, raise_on_error=False)
    assert any("mlt" in p for p in problems)
    assert any("inf" in p for p in problems)


def test_validate_merged_non_uniform_index():
    df = _minimal_merged(10)
    # Drop a row in the middle to break uniformity.
    df = df.drop(df.index[5])
    problems = schema.validate_merged(df, raise_on_error=False)
    assert any("uniform" in p for p in problems)


def test_imputed_mask_validation():
    df = _minimal_merged()
    df["vsw_imputed"] = 0
    df.loc[df.index[0], "vsw_imputed"] = 1
    assert schema.validate_merged(df, raise_on_error=False) == []
    df.loc[df.index[1], "vsw_imputed"] = 2  # invalid
    problems = schema.validate_merged(df, raise_on_error=False)
    assert any("vsw_imputed" in p for p in problems)
