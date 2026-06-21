"""Canonical column/dtype contract + validation (the binding schema).

This module encodes CONTRACTS.md §2-3 as importable constants and validation functions
so every builder references one source of truth. The column lists and the validators are
FULLY IMPLEMENTED and unit-tested (see tests/test_schema.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ps14.constants import TARGET_COLUMN

# --------------------------------------------------------------------------------------
# Canonical MERGED dataframe columns (CONTRACTS.md §2). dtype is the expected pandas dtype.
# --------------------------------------------------------------------------------------
MERGED_DTYPES: dict[str, str] = {
    "flux_e2": "float64",
    "log_flux_e2": "float64",
    "flux_seed": "float64",
    "log_flux_seed": "float64",
    "vsw": "float64",
    "density": "float64",
    "pdyn": "float64",
    "bz_gsm": "float64",
    "bt": "float64",
    "ae": "float64",
    "al": "float64",
    "kp": "float64",
    "sym_h": "float64",
    "f107": "float64",
    "mlt": "float64",
    "longitude": "float64",
}
# Columns that MUST be present for the merged frame to be valid (sat_id is a category and
# the *_imputed masks are optional/dynamic, so they are validated separately).
MERGED_REQUIRED: list[str] = list(MERGED_DTYPES.keys())

# --------------------------------------------------------------------------------------
# Feature matrix columns (CONTRACTS.md §3). Lag/roll names mirror config defaults.
# --------------------------------------------------------------------------------------
OBSERVED_PAST_BASE: list[str] = [
    "log_flux_e2",
    "log_flux_seed",
    "vsw",
    "density",
    "pdyn",
    "bz_gsm",
    "bt",
    "ae",
    "al",
    "kp",
    "sym_h",
    "f107",
]
LAG_COLUMNS: list[str] = [
    "log_flux_e2_lag_1",
    "log_flux_e2_lag_6",
    "log_flux_e2_lag_72",
    "log_flux_e2_lag_288",
    "log_flux_e2_lag_576",
    "vsw_lag_288",
    "vsw_lag_576",
]
ROLLING_COLUMNS: list[str] = [
    "log_flux_e2_rollmean_12",
    "log_flux_e2_rollmean_72",
    "log_flux_e2_rollmean_288",
    "log_flux_e2_rollstd_72",
    "log_flux_e2_rollstd_288",
    "log_flux_e2_rollmin_72",
    "log_flux_e2_rollmax_72",
    "vsw_rollmean_576",
    "ae_rollmean_288",
]
COUPLING_COLUMNS: list[str] = ["vbs", "newell", "epsilon", "clock_angle", "r0_standoff"]

# Known-future covariates -> TFT decoder (cyclic calendar/geometry; CONTRACTS.md §3).
KNOWN_FUTURE_COLUMNS: list[str] = [
    "tod_sin",
    "tod_cos",
    "doy_sin",
    "doy_cos",
    "mlt_sin",
    "mlt_cos",
]
# Static covariates -> TFT static encoder.
STATIC_COLUMNS: list[str] = ["sat_id", "longitude"]

# The observed-past feature set fed to the model encoder (order is the tensor channel order).
FEATURE_COLUMNS: list[str] = OBSERVED_PAST_BASE + LAG_COLUMNS + ROLLING_COLUMNS + COUPLING_COLUMNS

# The model target column.
TARGET: str = TARGET_COLUMN


class SchemaError(ValueError):
    """Raised when a frame violates the canonical schema contract."""


def _check_time_index(df: pd.DataFrame, expected_freq: str | None = "5min") -> list[str]:
    """Return a list of human-readable problems with the time index (empty if OK)."""
    problems: list[str] = []
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        problems.append("index is not a DatetimeIndex")
        return problems
    if idx.name != "time":
        problems.append(f"index name is {idx.name!r}, expected 'time'")
    if not idx.is_monotonic_increasing:
        problems.append("index is not monotonically increasing")
    if idx.has_duplicates:
        problems.append("index has duplicate timestamps")
    if expected_freq is not None and len(idx) > 2:
        deltas = idx.to_series().diff().dropna().unique()
        expected = pd.Timedelta(expected_freq)
        if not (len(deltas) == 1 and deltas[0] == np.timedelta64(expected)):
            problems.append(f"index is not uniform at {expected_freq}")
    return problems


def validate_merged(
    df: pd.DataFrame, *, expected_freq: str | None = "5min", raise_on_error: bool = True
) -> list[str]:
    """Validate the canonical MERGED dataframe (CONTRACTS.md §2).

    Checks: DatetimeIndex named ``time`` that is sorted/unique/uniform; all
    ``MERGED_REQUIRED`` columns present; no ``+/-inf``; ``mlt in [0,24)``; ``kp in [0,9]``;
    ``*_imputed`` columns are 0/1.

    Parameters
    ----------
    df:
        Frame to validate.
    expected_freq:
        Expected uniform cadence (``None`` to skip the uniformity check).
    raise_on_error:
        If True, raise :class:`SchemaError` on the first batch of problems; else return
        the list of problems.

    Returns
    -------
    list[str]
        Problems found (empty if valid). Only returned when ``raise_on_error`` is False.
    """
    problems = _check_time_index(df, expected_freq)

    for col in MERGED_REQUIRED:
        if col not in df.columns:
            problems.append(f"missing required column {col!r}")
            continue
        if not np.issubdtype(df[col].dtype, np.floating):
            problems.append(f"column {col!r} dtype {df[col].dtype} is not floating")
        if np.isinf(df[col].to_numpy(dtype="float64", na_value=np.nan)).any():
            problems.append(f"column {col!r} contains +/-inf")

    if "mlt" in df.columns:
        mlt = df["mlt"].dropna()
        if ((mlt < 0) | (mlt >= 24)).any():
            problems.append("mlt outside [0, 24)")
    if "kp" in df.columns:
        kp = df["kp"].dropna()
        if ((kp < 0) | (kp > 9)).any():
            problems.append("kp outside [0, 9]")

    for col in df.columns:
        if col.endswith("_imputed"):
            vals = pd.unique(df[col].dropna())
            if not set(np.asarray(vals).tolist()).issubset({0, 1}):
                problems.append(f"mask column {col!r} not in {{0,1}}")

    if raise_on_error and problems:
        raise SchemaError("merged frame failed validation: " + "; ".join(problems))
    return problems


def validate_features(df: pd.DataFrame, *, raise_on_error: bool = True) -> list[str]:
    """Validate the FEATURE matrix (CONTRACTS.md §3).

    Checks that every column in ``FEATURE_COLUMNS + KNOWN_FUTURE_COLUMNS`` is present and
    floating, the target is present, and there are no ``+/-inf`` values.
    """
    problems = _check_time_index(df, expected_freq="5min")
    required = [*FEATURE_COLUMNS, *KNOWN_FUTURE_COLUMNS, TARGET]
    for col in required:
        if col not in df.columns:
            problems.append(f"missing feature column {col!r}")
            continue
        if not np.issubdtype(df[col].dtype, np.floating):
            problems.append(f"feature {col!r} dtype {df[col].dtype} is not floating")
        if np.isinf(df[col].to_numpy(dtype="float64", na_value=np.nan)).any():
            problems.append(f"feature {col!r} contains +/-inf")
    if raise_on_error and problems:
        raise SchemaError("feature frame failed validation: " + "; ".join(problems))
    return problems


__all__ = [
    "MERGED_DTYPES",
    "MERGED_REQUIRED",
    "OBSERVED_PAST_BASE",
    "LAG_COLUMNS",
    "ROLLING_COLUMNS",
    "COUPLING_COLUMNS",
    "KNOWN_FUTURE_COLUMNS",
    "STATIC_COLUMNS",
    "FEATURE_COLUMNS",
    "TARGET",
    "SchemaError",
    "validate_merged",
    "validate_features",
]
