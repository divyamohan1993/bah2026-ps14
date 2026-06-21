"""Physically-plausible synthetic space-weather data generator (offline backbone).

The system MUST be fully runnable with no real data and no network. This module
synthesizes a multi-year dataset that encodes the physics from R1 (so downstream models
learn the right structure) and **writes it to CDF** so the real ``cdf_reader`` path is
exercised end-to-end, plus a mirror Parquet for speed. See ARCHITECTURE.md (c.3).

Physics encoded (R1):
  * Vsw: quiet baseline + recurrent high-speed streams (~27 d) + stochastic CME shocks.
  * IMF Bz, density N, dynamic pressure Pdyn coupled to events.
  * AE/AL, Kp, SYM-H derived from a (Vsw, Bz) coupling function with response/recovery.
  * Sub-MeV seed population responding to AE (injection) on an hours timescale.
  * >2 MeV flux: leaky-integrator/impulse response of Vsw with a 1-2 day lag + 2-day
    running mean, plus seed contribution, plus diurnal (noon max ~1 dex) keyed to MLT,
    plus semiannual (equinox) term, minus prompt dropouts on Pdyn spikes / strong south Bz.
  * Log-normal flux; realistic NaN gaps (short + long), spikes, and per-variable
    FILLVAL/VALIDMIN/MAX so the CDF masking path has work to do.

The same generator backs the online replay source when the network is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SyntheticParams:
    """Knobs for the synthetic generator (mirror config/default.yaml -> synthetic)."""

    start: str = "2014-01-01"
    end: str = "2025-01-01"
    cadence: str = "5min"
    seed: int = 1993
    baseline_vsw_kms: float = 400.0
    hss_recurrence_days: float = 27.0
    n_cme_per_year: int = 12
    vsw_to_flux_lag_days: float = 1.5
    diurnal_amplitude_dex: float = 0.5
    gap_fraction: float = 0.03
    spike_fraction: float = 0.002
    longitude_deg: float = -137.0  # GOES-West-like
    sat_id: str = "SYN-GEO"


def generate_solar_wind(time_index: pd.DatetimeIndex, params: SyntheticParams) -> pd.DataFrame:
    """Generate coupled solar-wind drivers (Vsw, density, Bz, Pdyn) + indices.

    Returns a frame indexed by ``time_index`` with columns ``vsw, density, bz_gsm, bt,
    pdyn, ae, al, kp, sym_h, f107`` following the canonical units (CONTRACTS.md §2).
    HSS recurrence, CME shocks, and Ornstein-Uhlenbeck noise produce realistic structure.
    """
    raise NotImplementedError(
        "TODO: build quiet baseline + ~27 d HSS + Poisson CME shocks; couple density/Bz/Pdyn; "
        "derive AE/AL/Kp/SYM-H via a (Vsw, Bz) coupling fn with response/recovery (R1 §2)."
    )


def generate_flux(
    time_index: pd.DatetimeIndex,
    drivers: pd.DataFrame,
    params: SyntheticParams,
) -> pd.DataFrame:
    """Generate the >2 MeV target flux and the sub-MeV seed flux from drivers.

    The >2 MeV log-flux is a leaky-integrator/impulse response of Vsw with a 1-2 day lag
    plus a 2-day running mean, plus the seed contribution, plus diurnal (noon max) and
    semiannual terms, minus dropouts on Pdyn spikes / strong southward Bz. Output is
    log-normal in linear space.

    Returns a frame with ``flux_e2, log_flux_e2, flux_seed, log_flux_seed, mlt`` columns.
    """
    raise NotImplementedError(
        "TODO: seed = injection(AE) decayed; log_flux_e2 = convolve(Vsw_anomaly, lag kernel) "
        "+ a*seed + diurnal(mlt) + semiannual(doy) - dropouts(Pdyn,Bz) + lognormal noise (R1)."
    )


def inject_artifacts(
    df: pd.DataFrame, params: SyntheticParams, rng: np.random.Generator
) -> pd.DataFrame:
    """Inject realistic NaN gaps (short + long) and spikes for the cleaning path.

    Short gaps (<= a few samples) and long gaps (hours) are placed at random; a fraction
    of points are multiplied by large factors to create spikes the Hampel filter removes.
    """
    raise NotImplementedError(
        "TODO: randomly NaN short + long runs (~gap_fraction); spike ~spike_fraction of points."
    )


def write_cdf(
    df: pd.DataFrame,
    path: str | Path,
    *,
    kind: str,
    params: SyntheticParams,
) -> Path:
    """Write a synthetic frame to an ISTP-flavoured CDF (TT2000 epoch, fill/valid attrs).

    Parameters
    ----------
    df:
        UTC-indexed frame to write.
    path:
        Output ``.cdf`` path (under ``data/raw/synthetic/``).
    kind:
        ``"goes"`` (flux variables) or ``"omni"`` (driver variables) — controls variable
        names/attributes so the reader exercises the real discovery path.
    params:
        Generator parameters (for global attributes / provenance).

    Returns
    -------
    Path
        The written file path. Writes ``Epoch`` (TT2000), per-variable ``FILLVAL``,
        ``VALIDMIN``/``VALIDMAX``, ``UNITS``, ``DEPEND_0``, ``VAR_TYPE='data'``.
    """
    raise NotImplementedError(
        "TODO: cdflib.CDF(path, ...) writer; compute TT2000 via cdfepoch.compute or "
        "timestamp_to_tt2000; set ISTP attrs so list_data_variables/get_variable_meta "
        "resolve correctly (R5 §1.3)."
    )


def generate(
    params: SyntheticParams | None = None,
    *,
    out_dir: str | Path = "data/raw/synthetic",
    write_parquet: bool = True,
    write_cdf_files: bool = True,
) -> dict[str, Path]:
    """End-to-end synthetic generation: drivers -> flux -> artifacts -> CDF + parquet.

    Parameters
    ----------
    params:
        Generator parameters; defaults to :class:`SyntheticParams`.
    out_dir:
        Output directory for CDF/parquet artifacts.
    write_parquet, write_cdf_files:
        Toggle the two output forms (CDF exercises the reader; parquet is the fast path).

    Returns
    -------
    dict[str, Path]
        Map of artifact name -> path, e.g. ``{"goes_cdf": ..., "omni_cdf": ...,
        "merged_parquet": ...}``.

    Notes
    -----
    Deterministic given ``params.seed``. This is the function the ``make synth-data`` /
    ``ps14 synth-data`` entry point calls.
    """
    raise NotImplementedError(
        "TODO: build 5-min time_index; drivers=generate_solar_wind; flux=generate_flux; "
        "df=join+inject_artifacts; write_cdf(goes/omni) + parquet; return paths."
    )


def replay_stream(params: SyntheticParams | None = None) -> pd.DataFrame:
    """Produce a short, SWPC-shaped synthetic stream for offline serving/replay.

    Returns the most recent window of synthetic flux + solar wind shaped like the parsed
    SWPC feeds (CONTRACTS.md §7) so the serving path runs with no network.
    """
    raise NotImplementedError(
        "TODO: generate the trailing window and shape like SWPC parse output."
    )


__all__ = [
    "SyntheticParams",
    "generate_solar_wind",
    "generate_flux",
    "inject_artifacts",
    "write_cdf",
    "generate",
    "replay_stream",
]
