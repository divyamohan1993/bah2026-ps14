"""Resampling to a uniform cadence (default 5-min) — R5 §4.5.

Downsamples higher-rate inputs (Wind ~92 s, GOES 1 min) with a variable-appropriate
aggregation; for burst-sensitive flux, prefer aggregating in LINEAR space then taking
log, and optionally track the within-bin max. Upsampling obeys the short-gap rule. The
resulting index is a gap-aware *uniform* DatetimeIndex (bins with no samples become NaN,
so the gap logic in :mod:`ps14.preprocess.clean` can classify short vs long gaps).
"""

from __future__ import annotations

import pandas as pd

# Columns whose physically-correct downsample aggregation differs from the plasma default
# (mean). Flux bursts are better preserved by aggregating in linear space and tracking the
# bin mean; here we use mean (the linear->log ordering is enforced by clean/transform).
DEFAULT_AGG: str = "mean"


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
    resampler = s.resample(freq)
    grid = getattr(resampler, agg)()
    grid = grid.asfreq(freq)
    grid.index.name = s.index.name
    return grid


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
        Regular-cadence frame aligned to a single canonical index; empty bins are NaN.
    """
    overrides = agg_overrides or {}
    resampler = df.resample(freq)
    agg_map = {col: overrides.get(col, default_agg) for col in df.columns}
    out = resampler.agg(agg_map)
    out = out.asfreq(freq)
    out = out[list(df.columns)]
    out.index.name = df.index.name
    return out


def resample_uniform(
    df: pd.DataFrame,
    cadence: str = "5min",
    agg: str = "mean",
    *,
    agg_overrides: dict[str, str] | None = None,
    mark_missing: bool = True,
) -> pd.DataFrame:
    """Resample a frame onto a gap-aware uniform ``cadence`` grid (CONTRACTS.md §2, R5 §4.5).

    Produces a monotonically increasing, unique, uniform DatetimeIndex. Bins that contain
    no samples become NaN. When ``mark_missing`` is True a single ``row_missing`` int8
    column flags rows introduced by the regular grid that had no underlying sample (so the
    gap logic / windowing can distinguish fabricated rows from real ones).

    Parameters
    ----------
    df:
        UTC-indexed (possibly irregular / higher-rate) frame.
    cadence:
        Target uniform cadence (e.g. ``"5min"``).
    agg:
        Default within-bin aggregation (``"mean"`` for plasma).
    agg_overrides:
        Per-column aggregation overrides (e.g. ``{"flux_e2": "max"}``).
    mark_missing:
        Add a ``row_missing`` int8 column marking introduced empty rows.

    Returns
    -------
    pd.DataFrame
        Uniform-cadence frame; empty bins NaN; optional ``row_missing`` flag.
    """
    out = resample_frame(df, freq=cadence, agg_overrides=agg_overrides, default_agg=agg)
    if mark_missing:
        # A row is "introduced/empty" if every original (non-flag) column is NaN there.
        data_cols = [c for c in out.columns if not c.endswith("_imputed")]
        if data_cols:
            empty = out[data_cols].isna().all(axis=1)
        else:  # pragma: no cover - degenerate frame with only masks
            empty = pd.Series(False, index=out.index)
        out["row_missing"] = empty.astype("int8")
    return out


def harmonize(
    df: pd.DataFrame,
    *,
    rename: dict[str, str] | None = None,
    dtypes: dict[str, str] | None = None,
    index_name: str = "time",
) -> pd.DataFrame:
    """Standardise column names / dtypes toward the canonical schema (R5 §4.0).

    A light, non-destructive harmoniser: renames columns, casts dtypes where requested,
    and normalises the time index name. Used to unify per-mission / per-era variable names
    (e.g. GOES-13/14/15 vs GOES-16/17/18 ``>2 MeV`` flux) into the canonical column set
    before merge.

    Parameters
    ----------
    df:
        Source frame (UTC-indexed).
    rename:
        ``{source_name: canonical_name}`` column renames.
    dtypes:
        ``{column: dtype}`` casts applied where the column exists.
    index_name:
        Canonical index name (default ``"time"``).

    Returns
    -------
    pd.DataFrame
        A copy with harmonised names / dtypes / index name.
    """
    out = df.copy()
    if rename:
        out = out.rename(columns=rename)
    if dtypes:
        for col, dtype in dtypes.items():
            if col in out.columns:
                out[col] = out[col].astype(dtype)
    out.index.name = index_name
    return out


__all__ = [
    "DEFAULT_AGG",
    "to_uniform_grid",
    "resample_frame",
    "resample_uniform",
    "harmonize",
]
