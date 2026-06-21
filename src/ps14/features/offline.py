"""Offline (batch) feature engineering: lags, rolling stats, coupling functions, cyclic.

Builds the feature matrix (CONTRACTS.md §3) from the merged dataframe. The physics-based
solar-wind coupling functions (vBs, Newell dPhi/dt, epsilon, clock angle, Shue standoff)
are FULLY IMPLEMENTED (exact formulas from R3 §10 / Newell 2007); the lag/rolling/cyclic
assembly carries the contract in its signature/docstring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ps14.datasets import schema
from ps14.utils import timeops

# Physical constants for coupling functions / Shue model.
_MU0 = 1.25663706212e-6  # vacuum permeability
_PROTON_MASS_KG = 1.67262192369e-27


def clock_angle(by_nt: np.ndarray, bz_nt: np.ndarray) -> np.ndarray:
    """IMF clock angle ``theta_c = atan2(By, Bz)`` in radians (0 = north, pi = south)."""
    return np.arctan2(np.asarray(by_nt, dtype="float64"), np.asarray(bz_nt, dtype="float64"))


def vbs(vsw_kms: np.ndarray, bz_nt: np.ndarray) -> np.ndarray:
    """Half-wave-rectified dawn-dusk merging E-field ``v * Bs`` (R3 §10, Newell 2007).

    ``Bs = -Bz`` when ``Bz < 0`` else 0 (southward IMF only). Units: mV/m when ``v`` is
    km/s and ``B`` is nT (v[km/s] * B[nT] * 1e-3). The single best simple driver of Dst.
    """
    v = np.asarray(vsw_kms, dtype="float64")
    bz = np.asarray(bz_nt, dtype="float64")
    bs = np.where(bz < 0.0, -bz, 0.0)
    return v * bs * 1.0e-3


def newell_coupling(vsw_kms: np.ndarray, by_nt: np.ndarray, bz_nt: np.ndarray) -> np.ndarray:
    """Newell universal coupling ``dPhi/dt = v^(4/3) * B_T^(2/3) * sin^(8/3)(theta_c/2)``.

    The best universal solar-wind -> magnetosphere coupling function (Newell 2007, R3 §10).
    ``B_T = sqrt(By^2 + Bz^2)`` (transverse IMF). Returned in SI-ish arbitrary units;
    used as a feature so absolute scaling is unimportant (the scaler normalizes it).
    """
    v = np.asarray(vsw_kms, dtype="float64")
    by = np.asarray(by_nt, dtype="float64")
    bz = np.asarray(bz_nt, dtype="float64")
    b_t = np.hypot(by, bz)
    theta_c = np.arctan2(by, bz)
    return (
        np.power(v, 4.0 / 3.0)
        * np.power(b_t, 2.0 / 3.0)
        * np.power(np.abs(np.sin(theta_c / 2.0)), 8.0 / 3.0)
    )


def epsilon_coupling(vsw_kms: np.ndarray, bt_nt: np.ndarray, theta_c_rad: np.ndarray) -> np.ndarray:
    """Akasofu epsilon energy-input parameter ``~ v * B^2 * sin^4(theta_c/2)`` (R3 §10).

    Proportional form (arbitrary units); ``bt_nt`` is the IMF magnitude |B|.
    """
    v = np.asarray(vsw_kms, dtype="float64")
    b = np.asarray(bt_nt, dtype="float64")
    theta = np.asarray(theta_c_rad, dtype="float64")
    return v * np.square(b) * np.power(np.abs(np.sin(theta / 2.0)), 4.0)


def dynamic_pressure(density_cm3: np.ndarray, vsw_kms: np.ndarray) -> np.ndarray:
    """Solar-wind dynamic pressure ``Pdyn = rho * v^2`` in nPa (~ 1.6726e-6 * N * V^2)."""
    n = np.asarray(density_cm3, dtype="float64")
    v = np.asarray(vsw_kms, dtype="float64")
    # N[cm^-3] * mp[kg] * 1e6 m^-3/cm^-3 * (v[km/s]*1e3 m/s)^2 -> Pa, then *1e9 -> nPa.
    return n * _PROTON_MASS_KG * 1.0e6 * np.square(v * 1.0e3) * 1.0e9


def shue_standoff(pdyn_npa: np.ndarray, bz_nt: np.ndarray) -> np.ndarray:
    """Shue et al. (1997) magnetopause standoff distance ``r0`` in Earth radii.

    ``r0 = (10.22 + 1.29 * tanh(0.184 * (Bz + 8.14))) * Pdyn^(-1/6.6)`` — the dayside
    compression/shadowing geometry feature (R1 §4, ARCHITECTURE.md (d) #13).
    """
    p = np.asarray(pdyn_npa, dtype="float64")
    bz = np.asarray(bz_nt, dtype="float64")
    with np.errstate(invalid="ignore"):
        return (10.22 + 1.29 * np.tanh(0.184 * (bz + 8.14))) * np.power(p, -1.0 / 6.6)


def add_coupling_functions(df: pd.DataFrame) -> pd.DataFrame:
    """Append ``vbs, newell, epsilon, clock_angle, r0_standoff`` to the merged frame.

    Requires ``vsw, bz_gsm, bt`` and (for the full Newell form) a ``by_gsm`` column; if
    ``by_gsm`` is absent it is approximated as 0 (clock angle then collapses to Bz sign).
    Also (re)derives ``pdyn`` from ``density``/``vsw`` when missing. Uses the FULLY
    IMPLEMENTED physics helpers above (CONTRACTS.md §3).
    """
    out = df.copy()
    vsw = out["vsw"].to_numpy(dtype="float64")
    bz = out["bz_gsm"].to_numpy(dtype="float64")
    bt = out["bt"].to_numpy(dtype="float64")
    by = (
        out["by_gsm"].to_numpy(dtype="float64")
        if "by_gsm" in out.columns
        else np.zeros_like(vsw)
    )

    # (Re)derive dynamic pressure from density/vsw when missing or all-NaN.
    if "pdyn" not in out.columns or out["pdyn"].isna().all():
        if "density" in out.columns:
            out["pdyn"] = dynamic_pressure(out["density"].to_numpy(dtype="float64"), vsw)

    theta_c = clock_angle(by, bz)
    out["vbs"] = vbs(vsw, bz)
    out["newell"] = newell_coupling(vsw, by, bz)
    out["epsilon"] = epsilon_coupling(vsw, bt, theta_c)
    out["clock_angle"] = theta_c
    out["r0_standoff"] = shue_standoff(out["pdyn"].to_numpy(dtype="float64"), bz)
    return out


def add_lag_features(df: pd.DataFrame, column: str, lags_steps: list[int]) -> pd.DataFrame:
    """Append ``{column}_lag_{k}`` shifted features (no look-ahead: shift is positive).

    A positive ``shift(k)`` brings the value from ``k`` steps in the PAST to the current
    row, so the lag feature is knowable at time ``t`` (never future).
    """
    out = df.copy()
    for k in lags_steps:
        out[f"{column}_lag_{k}"] = out[column].shift(k)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    column: str,
    windows: list[int],
    stats: tuple[str, ...] = ("mean", "std", "min", "max"),
) -> pd.DataFrame:
    """Append rolling ``{column}_roll{stat}_{w}`` features over trailing windows.

    Rolling windows are computed on the trailing window ``[t-w+1 .. t]`` only (no
    ``center``), so no future value leaks in. ``min_periods=1`` keeps partial windows at
    the series head (still strictly past-only).
    """
    out = df.copy()
    for w in windows:
        roller = out[column].rolling(window=w, min_periods=1)
        for stat in stats:
            out[f"{column}_roll{stat}_{w}"] = getattr(roller, stat)()
    return out


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append the known-future cyclic encodings (CONTRACTS.md §3).

    Adds ``tod_sin/tod_cos`` (time-of-day), ``doy_sin/doy_cos`` (day-of-year), and
    ``mlt_sin/mlt_cos`` (from the ``mlt`` column) using :mod:`ps14.utils.timeops`.
    """
    out = df.copy()
    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex):  # pragma: no cover - guarded upstream
        raise TypeError("features require a DatetimeIndex")
    out["tod_sin"], out["tod_cos"] = timeops.time_of_day_encoding(idx)
    out["doy_sin"], out["doy_cos"] = timeops.day_of_year_encoding(idx)
    if "mlt" in out.columns:
        out["mlt_sin"], out["mlt_cos"] = timeops.mlt_encoding(out["mlt"].to_numpy())
    return out


def build_feature_matrix(df: pd.DataFrame, config=None) -> pd.DataFrame:
    """Assemble the full feature matrix from the merged dataframe (CONTRACTS.md §3).

    Produces **exactly** ``schema.FEATURE_COLUMNS`` (plus ``schema.KNOWN_FUTURE_COLUMNS``,
    the target, static columns, and any ``*_imputed`` masks) by:

    * coupling functions (``vbs, newell, epsilon, clock_angle, r0_standoff``);
    * lag features (``log_flux_e2`` lags {1,6,72,288,576}; ``vsw`` lags {288,576});
    * rolling stats (``log_flux_e2`` rollmean {12,72,288}, rollstd {72,288}, rollmin/max
      {72}; ``vsw`` rollmean 576; ``ae`` rollmean 288) on trailing windows only;
    * known-future cyclic encodings (``tod_*``, ``doy_*``, ``mlt_*``).

    All lag/rolling features use only PAST samples (no look-ahead), so every column is
    knowable at its row's time ``t`` (R5 §5.2). The result is validated by
    ``schema.validate_features``.

    Parameters
    ----------
    df:
        The canonical MERGED dataframe (validated by ``schema.validate_merged``).
    config:
        Optional ``FeaturesConfig``; unused for the canonical column set (the exact
        FEATURE_COLUMNS are fixed by the schema). Accepted for API symmetry.

    Returns
    -------
    pd.DataFrame
        Same time index, FEATURE_COLUMNS-complete, with propagated ``*_imputed`` masks.
    """
    out = df.copy()

    # 1) Coupling functions (also (re)derives pdyn if absent).
    out = add_coupling_functions(out)

    # 2) Lags: log_flux_e2 {1,6,72,288,576}, vsw {288,576} (CONTRACTS.md §3 exact set).
    out = add_lag_features(out, "log_flux_e2", [1, 6, 72, 288, 576])
    out = add_lag_features(out, "vsw", [288, 576])

    # 3) Rolling stats, computed per (column, window, stat) to match the exact schema:
    #    log_flux_e2: rollmean {12,72,288}, rollstd {72,288}, rollmin {72}, rollmax {72}.
    out = add_rolling_features(out, "log_flux_e2", [12], stats=("mean",))
    out = add_rolling_features(out, "log_flux_e2", [72], stats=("mean", "std", "min", "max"))
    out = add_rolling_features(out, "log_flux_e2", [288], stats=("mean", "std"))
    #    vsw rollmean 576 (2-day); ae rollmean 288 (1-day).
    out = add_rolling_features(out, "vsw", [576], stats=("mean",))
    out = add_rolling_features(out, "ae", [288], stats=("mean",))

    # 4) Known-future cyclic encodings (tod/doy/mlt sin & cos).
    out = add_cyclic_features(out)

    # Ensure exact dtypes for the engineered float columns (validate_features wants float).
    engineered = [
        *schema.LAG_COLUMNS,
        *schema.ROLLING_COLUMNS,
        *schema.COUPLING_COLUMNS,
        *schema.KNOWN_FUTURE_COLUMNS,
    ]
    for col in engineered:
        if col in out.columns:
            out[col] = out[col].astype("float64")

    return out


__all__ = [
    "clock_angle",
    "vbs",
    "newell_coupling",
    "epsilon_coupling",
    "dynamic_pressure",
    "shue_standoff",
    "add_coupling_functions",
    "add_lag_features",
    "add_rolling_features",
    "add_cyclic_features",
    "build_feature_matrix",
]
