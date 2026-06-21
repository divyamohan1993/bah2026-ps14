"""L1 -> GEO time-alignment and GOES/OMNI merge (R5 §4.7, §5.1).

Joins the GEO flux (GOES) with the L1 driver block (OMNI/Wind) on the common 5-min grid.
When using OMNI_HRO (already bow-shock-nose time-shifted), no extra lag is applied; when
using raw Wind, a ballistic dx/Vsw shift is applied as a documented approximation
(:func:`ps14.utils.timeops.shift_l1_to_geo`). The output satisfies
:func:`ps14.datasets.schema.validate_merged` (CONTRACTS.md §2).
"""

from __future__ import annotations

import functools

import pandas as pd

from ps14.preprocess.resample import resample_uniform
from ps14.utils import timeops

# Map a friendly ``method`` to the underlying ``timeops.shift_l1_to_geo`` method.
_METHOD_ALIASES: dict[str, str] = {
    "omni": "omni_preshifted",
    "omni_preshifted": "omni_preshifted",
    "ballistic": "ballistic",
}


def merge_geo_l1(
    goes: pd.DataFrame,
    drivers: pd.DataFrame,
    *,
    method: str = "omni_preshifted",
    vsw_col: str = "vsw",
    how: str = "inner",
    cadence: str = "5min",
) -> pd.DataFrame:
    """Align GEO (GOES) and L1 (OMNI/Wind) onto one 5-min grid.

    Parameters
    ----------
    goes:
        GEO flux frame on the canonical grid (target + seed channels, MLT, static cols).
    drivers:
        L1 driver frame on the canonical grid (Vsw, density, Bz, indices, ...).
    method:
        ``"omni_preshifted"`` (no shift) or ``"ballistic"`` (apply per-sample dx/Vsw via
        :func:`ps14.utils.timeops.shift_l1_to_geo`).
    vsw_col:
        Speed column for the ballistic shift.
    how:
        Join policy (``"inner"`` recommended so only co-covered times remain).
    cadence:
        Uniform cadence used to re-grid the ballistic-shifted drivers.

    Returns
    -------
    pd.DataFrame
        The canonical MERGED dataframe (CONTRACTS.md §2) before feature engineering, with
        all ``*_imputed`` masks carried through.
    """
    shift_method = _METHOD_ALIASES.get(method, method)
    drivers_shifted = timeops.shift_l1_to_geo(
        drivers,
        method=shift_method,
        vsw_col=vsw_col,
        cadence=cadence if shift_method == "ballistic" else None,
    )
    assert isinstance(drivers_shifted, pd.DataFrame)  # return_lag not used here

    # Avoid duplicate columns when both frames carry the same key (e.g. a shared mask).
    overlap = goes.columns.intersection(drivers_shifted.columns)
    drivers_shifted = drivers_shifted.drop(columns=list(overlap))

    merged = goes.join(drivers_shifted, how=how)
    merged.index.name = "time"
    return merged


def align_l1_to_geo(
    geo_df: pd.DataFrame,
    l1_df: pd.DataFrame,
    *,
    method: str = "ballistic",
    cadence: str = "5min",
    vsw_col: str = "vsw",
    how: str = "inner",
) -> pd.DataFrame:
    """Time-shift L1 solar-wind to GEO and merge with the GOES target onto one grid.

    Thin, explicitly-named wrapper over :func:`merge_geo_l1` matching the build contract:
    ``method="ballistic"`` applies the physical L1->GEO propagation lag (Δx/Vsw, ~20-90 min)
    via :func:`ps14.utils.timeops.shift_l1_to_geo`; ``method="omni"`` treats the drivers as
    OMNI-preshifted (no extra lag). The returned frame satisfies
    :func:`ps14.datasets.schema.validate_merged`.

    Parameters
    ----------
    geo_df:
        GEO (GOES) flux/seed frame on the canonical uniform grid.
    l1_df:
        L1 solar-wind / IMF / index driver frame.
    method:
        ``"ballistic"`` | ``"omni"`` (alias of ``"omni_preshifted"``).
    cadence:
        Uniform cadence (default ``"5min"``).
    vsw_col:
        Speed column used by the ballistic shift.
    how:
        Join policy for the GEO/L1 merge.

    Returns
    -------
    pd.DataFrame
        The canonical MERGED dataframe (CONTRACTS.md §2).
    """
    return merge_geo_l1(
        geo_df,
        l1_df,
        method=method,
        vsw_col=vsw_col,
        how=how,
        cadence=cadence,
    )


def merge_sources(
    frames: dict[str, pd.DataFrame],
    *,
    cadence: str = "5min",
    how: str = "outer",
    resample: bool = False,
) -> pd.DataFrame:
    """Merge an arbitrary set of named source frames onto one uniform time grid.

    Joins all frames on the shared time index (optionally re-gridding each to ``cadence``
    first). Columns colliding across frames keep the first occurrence (sources should be
    pre-harmonised to the canonical names, so collisions are typically masks). The index is
    normalised to ``"time"``.

    Parameters
    ----------
    frames:
        Mapping ``name -> frame`` (each UTC-indexed). Iteration order defines precedence.
    cadence:
        Uniform cadence for optional re-gridding.
    how:
        Join policy across frames (``"outer"`` keeps the union of times).
    resample:
        If True, re-grid every frame to ``cadence`` before joining.

    Returns
    -------
    pd.DataFrame
        The unified frame on one ``time``-indexed grid.
    """
    if not frames:
        raise ValueError("merge_sources requires at least one frame")

    prepared: list[pd.DataFrame] = []
    seen: set[str] = set()
    for frame in frames.values():
        f = resample_uniform(frame, cadence=cadence, mark_missing=False) if resample else frame
        # Drop columns already contributed by an earlier (higher-precedence) source.
        keep = [c for c in f.columns if c not in seen]
        seen.update(keep)
        prepared.append(f[keep])

    merged = functools.reduce(lambda left, right: left.join(right, how=how), prepared)
    merged = merged.sort_index()
    merged.index.name = "time"
    return merged


__all__ = ["merge_geo_l1", "align_l1_to_geo", "merge_sources"]
