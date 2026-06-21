"""Time operations: cyclic encodings, L1->GEO propagation lag, epoch helpers.

The cyclic encodings are fully implemented (small, well-defined, and used as
known-future covariates by the model). The L1->GEO ballistic lag is implemented as a
documented approximation; the recommended path is OMNI's pre-shifted product (R5 §5.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ps14.constants import L1_LEAD_MINUTES_MEAN

# Distance from L1 to Earth ~ 1.5e6 km (~235 R_E). Used for the ballistic lag.
L1_TO_EARTH_KM: float = 1.5e6


# --------------------------------------------------------------------------------------
# Cyclic (sin/cos) encodings — known-future calendar/geometry covariates (R3 §10).
# --------------------------------------------------------------------------------------


def cyclic_encode(values: np.ndarray | pd.Series, period: float) -> tuple[np.ndarray, np.ndarray]:
    """Encode a periodic quantity as (sin, cos) on the unit circle.

    Parameters
    ----------
    values:
        Raw periodic values (e.g. hour-of-day in [0, 24), day-of-year in [1, 366]).
    period:
        The period in the same units as ``values`` (e.g. 24 for hour-of-day).

    Returns
    -------
    (sin, cos):
        Two float arrays of the same length as ``values``.
    """
    arr = np.asarray(values, dtype="float64")
    angle = 2.0 * np.pi * arr / period
    return np.sin(angle), np.cos(angle)


def time_of_day_encoding(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """(sin, cos) of time-of-day (diurnal cycle, period 24 h)."""
    hours = index.hour + index.minute / 60.0 + index.second / 3600.0
    return cyclic_encode(np.asarray(hours, dtype="float64"), period=24.0)


def day_of_year_encoding(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """(sin, cos) of day-of-year (seasonal / equinox cycle, period ~365.25 d)."""
    doy = index.dayofyear + (index.hour + index.minute / 60.0) / 24.0
    return cyclic_encode(np.asarray(doy, dtype="float64"), period=365.25)


def mlt_encoding(mlt_hours: np.ndarray | pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(sin, cos) of magnetic local time (period 24 h) — the noon/midnight cycle."""
    return cyclic_encode(mlt_hours, period=24.0)


# --------------------------------------------------------------------------------------
# L1 -> GEO propagation lag (R5 §5.1).
# --------------------------------------------------------------------------------------


def ballistic_lag_minutes(
    vsw_kms: np.ndarray | pd.Series, distance_km: float = L1_TO_EARTH_KM
) -> np.ndarray:
    """Flat (ballistic) propagation lag from L1 to the target, ``dt = dx / Vsw``.

    Parameters
    ----------
    vsw_kms:
        Solar-wind radial speed [km/s].
    distance_km:
        Sun-Earth-line distance from the monitor to the target [km].

    Returns
    -------
    np.ndarray
        Lag in minutes per sample (typically ~30-70 min). Where ``vsw <= 0`` the lag
        falls back to the climatological mean (R1 §3).

    Notes
    -----
    This ignores phase-front tilt. Prefer the OMNI pre-shifted product when available
    (``preprocess.l1_to_geo.method == 'omni_preshifted'``).
    """
    v = np.asarray(vsw_kms, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        lag_min = (distance_km / v) / 60.0
    lag_min = np.where(np.isfinite(lag_min) & (v > 0), lag_min, L1_LEAD_MINUTES_MEAN)
    return lag_min


def shift_l1_to_geo(
    df: pd.DataFrame,
    *,
    method: str = "omni_preshifted",
    vsw_col: str = "vsw",
    target_distance_re: float = 0.0,
) -> pd.DataFrame:
    """Shift an L1 driver frame forward in time toward the magnetosphere.

    Parameters
    ----------
    df:
        L1 driver frame indexed by UTC time.
    method:
        ``'omni_preshifted'`` (no-op; OMNI already bow-shock-shifted) or ``'ballistic'``
        (apply a per-sample ``dx/Vsw`` shift using ``vsw_col``).
    vsw_col:
        Column holding solar-wind speed (for the ballistic method).
    target_distance_re:
        Extra bow-shock-nose -> GEO transit distance in Earth radii (optional).

    Returns
    -------
    pd.DataFrame
        The frame with a shifted index (and re-gridded to the original cadence by the
        caller). Raises for unknown methods.
    """
    raise NotImplementedError(
        "TODO: implement OMNI no-op pass-through and the ballistic per-sample shift; "
        "re-grid onto the canonical 5-min index after shifting (see CONTRACTS.md §2)."
    )


__all__ = [
    "cyclic_encode",
    "time_of_day_encoding",
    "day_of_year_encoding",
    "mlt_encoding",
    "ballistic_lag_minutes",
    "shift_l1_to_geo",
    "L1_TO_EARTH_KM",
]
