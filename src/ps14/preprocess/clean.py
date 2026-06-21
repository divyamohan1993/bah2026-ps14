"""Cleaning: fill/valid masking, Hampel/MAD despiking, gap detection + interpolation.

Implements R5 §4.1-4.4. The Hampel despike and gap-aware interpolation are FULLY
IMPLEMENTED (small, well-defined, and unit-tested). Fill/valid masking for raw arrays
lives in :mod:`ps14.io.cdf_reader` (mask_invalid); here we operate on pandas Series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# MAD -> sigma consistency constant for Gaussian data (R5 §4.2).
MAD_TO_SIGMA: float = 1.4826


def hampel_filter(
    s: pd.Series,
    window: int = 7,
    n_sigma: float = 3.0,
    replace: str = "nan",
) -> tuple[pd.Series, pd.Series]:
    """Robust despiking via rolling median + MAD (R5 §4.1).

    A point is an outlier if it deviates from the centered rolling median by more than
    ``n_sigma * 1.4826 * MAD``. Preferred over mean +/- k*std because flux is heavy-tailed
    and bursty.

    Parameters
    ----------
    s:
        Input series (a single channel).
    window:
        Number of samples in the centered rolling window.
    n_sigma:
        MAD threshold multiplier.
    replace:
        ``"nan"`` -> set outliers to NaN (recommended; let gap logic decide) or
        ``"median"`` -> replace with the rolling median.

    Returns
    -------
    (filtered, outlier_mask):
        ``filtered`` is the despiked series; ``outlier_mask`` is a boolean Series marking
        detected outliers (NaNs are never flagged as outliers).
    """
    med = s.rolling(window, center=True, min_periods=1).median()
    abs_dev = (s - med).abs()
    mad = abs_dev.rolling(window, center=True, min_periods=1).median()
    threshold = n_sigma * MAD_TO_SIGMA * mad
    outliers = (abs_dev > threshold) & s.notna()

    out = s.copy()
    if replace == "median":
        out[outliers] = med[outliers]
    elif replace == "nan":
        out[outliers] = np.nan
    else:  # pragma: no cover - guarded by config validation
        raise ValueError(f"replace must be 'nan' or 'median', got {replace!r}")
    return out, outliers.fillna(False)


def nan_run_lengths(mask: pd.Series) -> pd.Series:
    """Length of each consecutive-NaN run, broadcast to every element of that run.

    Parameters
    ----------
    mask:
        Boolean Series that is True where the value is missing (NaN).

    Returns
    -------
    pd.Series
        Integer Series; for positions inside a NaN run, the run's total length; 0 for
        non-missing positions.
    """
    isna = mask.astype(bool)
    # Group consecutive NaNs together: the cumulative count of non-NaN positions is a
    # group id constant within each NaN run.
    group = (~isna).cumsum()
    run_len = isna.groupby(group).transform("sum")
    return run_len.where(isna, other=0).astype("int64")


def detect_gaps(s: pd.Series, max_gap_steps: int = 6) -> tuple[pd.Series, pd.Series]:
    """Classify missing data into short (interpolatable) vs long gaps.

    Parameters
    ----------
    s:
        Series already on the intended uniform grid (so missing = NaN positions).
    max_gap_steps:
        Maximum consecutive-NaN run length still considered a SHORT gap.

    Returns
    -------
    (short_gap_mask, long_gap_mask):
        Boolean Series: ``short_gap_mask`` marks NaNs in runs ``<= max_gap_steps``;
        ``long_gap_mask`` marks NaNs in longer runs (left NaN, windows dropped later).
    """
    isna = s.isna()
    run_len = nan_run_lengths(isna)
    short = isna & (run_len <= max_gap_steps)
    long = isna & (run_len > max_gap_steps)
    return short, long


def interpolate_short_gaps(
    s: pd.Series,
    max_gap_steps: int = 6,
    method: str = "time",
) -> tuple[pd.Series, pd.Series]:
    """Interpolate only short gaps; leave long gaps as NaN; flag imputed points (R5 §4.4).

    Parameters
    ----------
    s:
        Series on the uniform grid.
    max_gap_steps:
        Short-gap threshold (longer runs stay NaN — never fabricate dynamics).
    method:
        pandas interpolation method (``"time"`` recommended for plasma; avoid spline for
        spiky flux).

    Returns
    -------
    (filled, imputed_mask):
        ``filled`` has short gaps interpolated; ``imputed_mask`` (int8) is 1 where a value
        was filled. Long gaps remain NaN in ``filled``.
    """
    short, _long = detect_gaps(s, max_gap_steps=max_gap_steps)
    interp = s.interpolate(method=method, limit=max_gap_steps, limit_area="inside")
    out = s.copy()
    out[short] = interp[short]
    imputed = short.astype("int8")
    return out, imputed


__all__ = [
    "MAD_TO_SIGMA",
    "hampel_filter",
    "nan_run_lengths",
    "detect_gaps",
    "interpolate_short_gaps",
]
