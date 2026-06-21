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


# Earth radius [km] — used to convert the optional bow-shock-nose -> GEO transit distance.
EARTH_RADIUS_KM: float = 6371.0


def shift_l1_to_geo(
    df: pd.DataFrame,
    *,
    method: str = "omni_preshifted",
    vsw_col: str = "vsw",
    target_distance_re: float = 0.0,
    cadence: str | None = None,
    return_lag: bool = False,
) -> pd.DataFrame | np.ndarray:
    """Shift an L1 driver frame forward in time toward the magnetosphere (R5 §5.1).

    Applies the physical L1 -> bow-shock-nose/GEO propagation lag so that each solar-wind
    sample is stamped with the (later) time it reaches the magnetosphere. The ballistic
    lag is ``dt = dx / Vsw`` (~20-90 min; reuses :func:`ballistic_lag_minutes`); the
    propagation distance is the L1 Sun-Earth-line distance plus an optional
    bow-shock-nose -> GEO transit term (``target_distance_re`` Earth radii).

    Parameters
    ----------
    df:
        L1 driver frame indexed by UTC time.
    method:
        ``'omni_preshifted'`` (no-op pass-through; OMNI HRO is already bow-shock-shifted)
        or ``'ballistic'``/``'omni'`` aliases. ``'ballistic'`` applies a per-sample
        ``dx/Vsw`` shift using ``vsw_col``; ``'omni'`` is treated as pre-shifted.
    vsw_col:
        Column holding solar-wind speed [km/s] (for the ballistic method).
    target_distance_re:
        Extra bow-shock-nose -> GEO transit distance in Earth radii (optional, added to the
        ~1.5e6 km L1 distance before dividing by Vsw).
    cadence:
        If given (e.g. ``"5min"``), the time-shifted ballistic series is re-gridded back
        onto a uniform grid at this cadence (mean within bin) so the result stays on the
        canonical index (CONTRACTS.md §2). Ignored for the pre-shifted no-op.
    return_lag:
        If True (ballistic only), return the per-row lag in **minutes** as an ``np.ndarray``
        instead of the shifted frame.

    Returns
    -------
    pd.DataFrame | np.ndarray
        The frame with a forward-shifted (and optionally re-gridded) index, or — when
        ``return_lag`` is True — the per-sample lag in minutes. Raises for unknown methods.
    """
    if method in ("omni_preshifted", "omni"):
        # OMNI HRO is already convected to the bow-shock nose (PFN method); no-op.
        return df.copy()
    if method != "ballistic":
        raise ValueError(
            f"method must be 'omni_preshifted', 'omni', or 'ballistic', got {method!r}"
        )

    if vsw_col not in df.columns:
        raise KeyError(f"ballistic shift requires speed column {vsw_col!r}")

    distance_km = L1_TO_EARTH_KM + float(target_distance_re) * EARTH_RADIUS_KM
    lag_min = ballistic_lag_minutes(df[vsw_col].to_numpy(), distance_km=distance_km)
    if return_lag:
        return lag_min

    shifted = df.copy()
    shifted.index = df.index + pd.to_timedelta(lag_min, unit="m")
    shifted = shifted[~shifted.index.duplicated(keep="last")].sort_index()

    if cadence is not None:
        shifted = shifted.resample(cadence).mean().asfreq(cadence)
        shifted.index.name = df.index.name

    return shifted


__all__ = [
    "cyclic_encode",
    "time_of_day_encoding",
    "day_of_year_encoding",
    "mlt_encoding",
    "ballistic_lag_minutes",
    "shift_l1_to_geo",
    "L1_TO_EARTH_KM",
    "EARTH_RADIUS_KM",
]
