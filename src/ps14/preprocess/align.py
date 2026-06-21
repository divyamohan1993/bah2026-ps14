"""L1 -> GEO time-alignment and GOES/OMNI merge (R5 §4.7, §5.1).

Joins the GEO flux (GOES) with the L1 driver block (OMNI/Wind) on the common 5-min grid.
When using OMNI_HRO (already bow-shock-nose time-shifted), no extra lag is applied; when
using raw Wind, a ballistic dx/Vsw shift is applied as a documented approximation.
"""

from __future__ import annotations

import pandas as pd


def merge_geo_l1(
    goes: pd.DataFrame,
    drivers: pd.DataFrame,
    *,
    method: str = "omni_preshifted",
    vsw_col: str = "vsw",
    how: str = "inner",
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

    Returns
    -------
    pd.DataFrame
        The canonical MERGED dataframe (CONTRACTS.md §2) before feature engineering, with
        all ``*_imputed`` masks carried through.
    """
    raise NotImplementedError(
        "TODO: optionally shift drivers (timeops.shift_l1_to_geo) then "
        "goes.join(drivers, how=how); carry imputed masks; preserve canonical "
        "column set (CONTRACTS.md §2)."
    )


__all__ = ["merge_geo_l1"]
