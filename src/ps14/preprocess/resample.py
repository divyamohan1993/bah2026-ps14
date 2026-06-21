"""Resampling to a uniform cadence (default 5-min) — R5 §4.5.

Downsamples higher-rate inputs (Wind ~92 s, GOES 1 min) with a variable-appropriate
aggregation; for burst-sensitive flux, prefer aggregating in LINEAR space then taking
log, and optionally track the within-bin max. Upsampling obeys the short-gap rule.
"""

from __future__ import annotations

import pandas as pd


def to_uniform_grid(
    s: pd.Series,
    freq: str = "5min",
    agg: str = "mean",
) -> pd.Series:
    """Resample a series onto a regular ``freq`` grid (NaN where a bin is empty).

    Parameters
    ----------
    s:
        Irregular or higher-rate input series (UTC-indexed).
    freq:
        Target cadence string (e.g. ``"5min"``).
    agg:
        Aggregation within each bin (``"mean"`` for plasma, ``"max"`` to preserve flux
        bursts, ``"median"`` for robustness).

    Returns
    -------
    pd.Series
        Regular-cadence series; bins with no samples are NaN (gap logic handles them).
    """
    raise NotImplementedError(
        "TODO: getattr(s.resample(freq), agg)().asfreq(freq); flux should be aggregated in "
        "linear space before logging (R5 §4.5)."
    )


def resample_frame(
    df: pd.DataFrame,
    freq: str = "5min",
    agg_overrides: dict[str, str] | None = None,
    default_agg: str = "mean",
) -> pd.DataFrame:
    """Resample every column of a frame onto the uniform grid.

    Parameters
    ----------
    df:
        UTC-indexed multi-column frame.
    freq:
        Target cadence.
    agg_overrides:
        Per-column aggregation overrides (e.g. ``{"flux_e2": "max"}``).
    default_agg:
        Aggregation for columns not in ``agg_overrides``.

    Returns
    -------
    pd.DataFrame
        Regular-cadence frame aligned to a single canonical index.
    """
    raise NotImplementedError("TODO: per-column resample with overrides; concat onto one grid.")


__all__ = ["to_uniform_grid", "resample_frame"]
