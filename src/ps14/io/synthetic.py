"""Physically-plausible synthetic space-weather data generator (offline backbone).

The system MUST be fully runnable with no real data and no network. This module
synthesizes a multi-year dataset that encodes the physics from R1 (so downstream models
learn the right structure) and **writes it to CDF** so the real ``cdf_reader`` path is
exercised end-to-end, plus a mirror Parquet for speed. See ARCHITECTURE.md (c.3).

Physics encoded (R1):
  * ``vsw``: quiet baseline + recurrent high-speed streams (~27 d corotating) + stochastic
    CME shocks (sharp rise to 600-800 km/s, exponential recovery) + Ornstein-Uhlenbeck
    noise.
  * IMF ``bz_gsm`` (mean ~0 with sustained southward excursions during disturbances),
    ``bt`` (|B|, ~5 nT baseline + enhancements), density ``N`` (~5 cm^-3 + shock
    compressions), dynamic pressure ``pdyn`` (~ 1.6726e-6 * N * Vsw^2), all coupled to
    events.
  * ``ae``/``al`` (substorm activity ~ Newell-like coupling of southward Bz * Vsw + noise;
    ae>0, al<0), ``kp`` (0-9), ``sym_h`` (negative during storms, ring-current injection +
    recovery), ``f107`` (slow solar-cycle variation 70-200 sfu).
  * ``flux_e2`` (>2 MeV): ``log10(flux)`` modelled as a 1-2 DAY LAGGED, smoothed response
    to Vsw (radial-diffusion / chorus acceleration), a DIURNAL term (sinusoid in MLT, peak
    near local noon), a semiannual (equinox) term, a seed-population contribution, minus
    storm DROPOUTS (magnetopause shadowing on high Pdyn / deep SYM-H), plus log-normal
    noise. Calibrated so quiet ~10^1-10^2 pfu and active periods exceed 1000 pfu often
    enough (~5-20% of samples) to give positive exceedance examples.
  * ``flux_seed`` (sub-MeV 40-475 keV): higher baseline, leads MeV by hours, driven by
    AE/substorm injections.

The same generator backs the online replay source when the network is unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ps14.constants import HARSH_PFU
from ps14.datasets import schema
from ps14.features.offline import dynamic_pressure, newell_coupling
from ps14.preprocess.transform import log10_floor

logger = logging.getLogger(__name__)

# Default log floor for the flux columns (mirrors config.preprocess.log_floor_pfu).
_LOG_FLOOR_PFU: float = 0.01

# ISTP fill value for the written CDFs (the conventional float FILLVAL, R5 §1.3).
_FILLVAL: float = -1.0e31


@dataclass
class SyntheticParams:
    """Knobs for the synthetic generator (mirror config/default.yaml -> synthetic)."""

    start: str = "2014-01-01"
    end: str = "2025-01-01"
    cadence: str = "5min"
    seed: int = 1993
    baseline_vsw_kms: float = 400.0
    hss_recurrence_days: float = 27.0
    n_cme_per_year: int = 36  # ~3/month: sporadic but frequent enough for short demos
    vsw_to_flux_lag_days: float = 1.5
    diurnal_amplitude_dex: float = 0.5
    gap_fraction: float = 0.03
    spike_fraction: float = 0.002
    longitude_deg: float = -137.0  # GOES-West-like
    sat_id: str = "SYN-GEO"


# --------------------------------------------------------------------------------------
# Low-level stochastic-process helpers
# --------------------------------------------------------------------------------------


def _ou_process(
    n: int, *, theta: float, sigma: float, rng: np.random.Generator, x0: float = 0.0
) -> np.ndarray:
    """Generate an Ornstein-Uhlenbeck (mean-reverting) noise series of length ``n``.

    ``dx = -theta * x * dt + sigma * dW`` with ``dt = 1`` step. ``theta`` controls the
    reversion rate (memory) and ``sigma`` the per-step innovation scale.
    """
    x = np.empty(n, dtype="float64")
    x[0] = x0
    noise = rng.standard_normal(n)
    keep = 1.0 - theta
    for i in range(1, n):
        x[i] = keep * x[i - 1] + sigma * noise[i]
    return x


def _causal_decay_kernel(tau_steps: float, n_steps: int) -> np.ndarray:
    """Return a normalised one-sided exponential-decay kernel of length ``n_steps``.

    Used as an impulse response for leaky-integrator responses (injection decay,
    ring-current recovery). Normalised to unit sum so it acts as a weighted average.
    """
    t = np.arange(n_steps, dtype="float64")
    k = np.exp(-t / max(tau_steps, 1e-6))
    return k / k.sum()


def _causal_convolve(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Causal convolution of ``x`` with ``kernel`` (kernel[0] weights the present sample).

    The output at index ``i`` depends only on ``x[:i+1]`` (no look-ahead), matching the
    physics that a response integrates *past* driving.
    """
    full = np.convolve(x, kernel, mode="full")
    return full[: len(x)]


# --------------------------------------------------------------------------------------
# Driver generation
# --------------------------------------------------------------------------------------


def generate_solar_wind(time_index: pd.DatetimeIndex, params: SyntheticParams) -> pd.DataFrame:
    """Generate coupled solar-wind drivers (Vsw, density, Bz, Pdyn) + indices.

    Returns a frame indexed by ``time_index`` with columns ``vsw, density, bz_gsm, bt,
    pdyn, ae, al, kp, sym_h, f107`` following the canonical units (CONTRACTS.md §2).
    HSS recurrence, CME shocks, and Ornstein-Uhlenbeck noise produce realistic structure.
    """
    rng = np.random.default_rng(params.seed)
    n = len(time_index)
    if n == 0:
        return pd.DataFrame(
            {c: pd.Series(dtype="float64") for c in schema.MERGED_REQUIRED}, index=time_index
        )

    steps_per_day = pd.Timedelta(days=1) / pd.Timedelta(params.cadence)
    days = (np.arange(n, dtype="float64")) / steps_per_day
    total_days = max(days[-1], 1.0)

    # --- recurrent high-speed streams (HSS): ~27-day corotating periodicity ----------
    # Half-wave-rectified sinusoid: fast wind for part of each recurrence, quiet otherwise.
    hss_phase = rng.uniform(0.0, 2.0 * np.pi)
    hss_raw = np.sin(2.0 * np.pi * days / params.hss_recurrence_days + hss_phase)
    hss = np.clip(hss_raw, 0.0, None) ** 1.3  # sharpen the stream onset
    hss_speed = 220.0 * hss  # peak HSS contribution ~+220 km/s

    # --- sporadic CME shocks: Poisson in time, sharp rise + exponential recovery ------
    n_cme = int(rng.poisson(params.n_cme_per_year * total_days / 365.25))
    cme_speed = np.zeros(n, dtype="float64")
    cme_density = np.zeros(n, dtype="float64")
    cme_bz = np.zeros(n, dtype="float64")
    cme_bt = np.zeros(n, dtype="float64")
    decay_steps = 1.5 * steps_per_day  # ~1.5-day recovery

    # Build the onset list. Always seed at least one event in the FIRST THIRD of the
    # window so its ~1.5-day lagged flux response peaks inside even a short (few-day)
    # span -- this guarantees positive exceedance examples for testability while staying
    # physically plausible (storms are common during active periods).
    onsets: list[int] = []
    if n > 1:
        first_third = max(int(0.34 * n), 1)
        onsets.append(int(rng.integers(0, first_third)))
    onsets.extend(int(rng.integers(0, n)) for _ in range(max(n_cme - 1, 0)))

    for k, onset in enumerate(onsets):
        # The guaranteed early event is biased strong so the lagged response clears the
        # 1000 pfu threshold; subsequent Poisson events span the usual range.
        if k == 0:
            amp = rng.uniform(320.0, 430.0)
        else:
            amp = rng.uniform(150.0, 420.0)  # shock speed jump -> peaks 600-800 km/s
        t_rel = np.arange(n - onset, dtype="float64")
        profile = np.exp(-t_rel / decay_steps)
        cme_speed[onset:] += amp * profile
        # Shock compression: density spike (15-45 cm^-3), enhanced |B|, southward Bz.
        cme_density[onset:] += rng.uniform(15.0, 45.0) * profile
        cme_bt[onset:] += rng.uniform(8.0, 22.0) * profile
        cme_bz[onset:] += -rng.uniform(6.0, 18.0) * profile  # sustained southward

    # --- bulk speed -------------------------------------------------------------------
    vsw_noise = _ou_process(n, theta=0.02, sigma=8.0, rng=rng)
    vsw = params.baseline_vsw_kms + hss_speed + cme_speed + vsw_noise
    vsw = np.clip(vsw, 250.0, 1000.0)

    # --- density: quiet baseline, anti-correlated with fast wind, + shock compression -
    density_noise = _ou_process(n, theta=0.05, sigma=0.6, rng=rng)
    density = 6.0 - 0.006 * (vsw - params.baseline_vsw_kms) + cme_density + density_noise
    density = np.clip(density, 0.3, 80.0)

    # --- IMF |B| and Bz ---------------------------------------------------------------
    bt_noise = np.abs(_ou_process(n, theta=0.05, sigma=0.8, rng=rng))
    bt = 4.5 + 0.6 * hss + cme_bt + bt_noise
    bt = np.clip(bt, 0.5, 60.0)

    # Bz: zero-mean OU fluctuation + sustained southward during HSS/CME disturbances.
    bz_noise = _ou_process(n, theta=0.08, sigma=1.6, rng=rng)
    bz_gsm = bz_noise + cme_bz - 1.2 * hss  # HSS gives moderate sustained southward bias
    bz_gsm = np.clip(bz_gsm, -1.05 * bt, 1.05 * bt)  # |Bz| cannot exceed |B| (approx)

    # --- dynamic pressure -------------------------------------------------------------
    pdyn = dynamic_pressure(density, vsw)

    # --- geomagnetic indices from a Newell-like coupling function ---------------------
    # Coupling drives substorm (AE) and storm (SYM-H) activity. Use By~0 here.
    by = np.zeros(n, dtype="float64")
    coupling = newell_coupling(vsw, by, bz_gsm)  # >= 0, peaks for fast + southward
    coupling_n = coupling / (np.median(coupling[coupling > 0]) + 1e-9)

    # AE: prompt substorm response (short memory) + noise; strictly positive.
    ae_kernel = _causal_decay_kernel(tau_steps=0.5 * steps_per_day, n_steps=int(2 * steps_per_day))
    ae_drive = _causal_convolve(coupling_n, ae_kernel)
    ae_noise = np.abs(_ou_process(n, theta=0.1, sigma=0.4, rng=rng))
    ae = 40.0 + 150.0 * ae_drive + 60.0 * ae_noise
    ae = np.clip(ae, 0.0, 3000.0)
    # AL is the (negative) lower envelope of the electrojet; track -AE with extra depth.
    al = -(0.55 * ae + 30.0 * np.abs(_ou_process(n, theta=0.1, sigma=0.3, rng=rng)))
    al = np.clip(al, -2500.0, 0.0)

    # SYM-H: ring-current injection (~ vBs) integrated with a recovery timescale.
    bs = np.clip(-bz_gsm, 0.0, None)
    injection = (vsw / 400.0) * bs  # proportional to dawn-dusk merging E-field
    sym_kernel = _causal_decay_kernel(tau_steps=1.2 * steps_per_day, n_steps=int(6 * steps_per_day))
    sym_h = -8.0 * _causal_convolve(injection, sym_kernel)
    sym_h += _ou_process(n, theta=0.05, sigma=1.0, rng=rng)
    sym_h = np.clip(sym_h, -600.0, 60.0)

    # Kp: coarse 3-hour global activity proxy on 0-9, smoothed coupling.
    kp_kernel = _causal_decay_kernel(tau_steps=0.5 * steps_per_day, n_steps=int(2 * steps_per_day))
    kp_drive = _causal_convolve(coupling_n, kp_kernel)
    kp = 1.5 + 2.2 * np.tanh(0.8 * kp_drive) + 0.6 * ae_noise
    kp = np.clip(kp, 0.0, 9.0)
    # Quantise to the nearest 1/3 (Kp is reported in thirds).
    kp = np.round(kp * 3.0) / 3.0

    # F10.7: slow solar-cycle variation (~11 yr) + a small noise term, 70-200 sfu.
    solar_cycle = 0.5 * (1.0 - np.cos(2.0 * np.pi * days / (11.0 * 365.25)))
    f107 = 70.0 + 130.0 * solar_cycle + 5.0 * _ou_process(n, theta=0.005, sigma=0.5, rng=rng)
    f107 = np.clip(f107, 65.0, 300.0)

    return pd.DataFrame(
        {
            "vsw": vsw,
            "density": density,
            "bz_gsm": bz_gsm,
            "bt": bt,
            "pdyn": pdyn,
            "ae": ae,
            "al": al,
            "kp": kp,
            "sym_h": sym_h,
            "f107": f107,
        },
        index=time_index,
    )


def _magnetic_local_time(time_index: pd.DatetimeIndex, longitude_deg: float) -> np.ndarray:
    """Approximate magnetic local time (hours, [0,24)) of a GEO sensor at ``longitude_deg``.

    MLT is local solar time at the sensor's magnetic footpoint; to first order it is UT
    plus the sensor longitude expressed in hours (15 deg = 1 h). Midnight in MLT occurs
    when the sensor is on the anti-sunward side.
    """
    ut_hours = (
        time_index.hour
        + time_index.minute / 60.0
        + time_index.second / 3600.0
    ).to_numpy(dtype="float64")
    mlt = (ut_hours + longitude_deg / 15.0) % 24.0
    return mlt


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
    rng = np.random.default_rng(params.seed + 1)  # independent stream from the drivers
    n = len(time_index)
    if n == 0:
        cols = ["flux_e2", "log_flux_e2", "flux_seed", "log_flux_seed", "mlt"]
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols}, index=time_index)

    steps_per_day = pd.Timedelta(days=1) / pd.Timedelta(params.cadence)
    days = np.arange(n, dtype="float64") / steps_per_day

    vsw = drivers["vsw"].to_numpy(dtype="float64")
    ae = drivers["ae"].to_numpy(dtype="float64")
    pdyn = drivers["pdyn"].to_numpy(dtype="float64")
    bz = drivers["bz_gsm"].to_numpy(dtype="float64")
    sym_h = drivers["sym_h"].to_numpy(dtype="float64")

    # --- seed (sub-MeV) population: prompt substorm injection, leads the MeV channel ---
    seed_kernel = _causal_decay_kernel(
        tau_steps=0.25 * steps_per_day, n_steps=int(2 * steps_per_day)
    )
    ae_norm = (ae - 40.0) / 200.0
    seed_drive = _causal_convolve(np.clip(ae_norm, 0.0, None), seed_kernel)
    log_seed = 2.7 + 1.1 * seed_drive  # baseline ~10^2.7 pfu, rises with injection
    log_seed += 0.12 * rng.standard_normal(n)

    # --- >2 MeV channel: lagged, smoothed response to Vsw -----------------------------
    # Build a delayed, broad acceleration kernel peaking at the configured lag (1-2 days).
    lag_steps = params.vsw_to_flux_lag_days * steps_per_day
    width_steps = 0.9 * steps_per_day
    ker_len = int(4 * steps_per_day)
    t_k = np.arange(ker_len, dtype="float64")
    accel_kernel = np.exp(-0.5 * ((t_k - lag_steps) / width_steps) ** 2)
    accel_kernel = accel_kernel / accel_kernel.sum()

    vsw_anom = (vsw - params.baseline_vsw_kms) / 200.0  # dimensionless speed anomaly
    accel = _causal_convolve(vsw_anom, accel_kernel)

    # Slow 2-day running mean term (radial-diffusion build-up memory, R1 §2).
    run_kernel = _causal_decay_kernel(tau_steps=2.0 * steps_per_day, n_steps=int(6 * steps_per_day))
    vsw_run = _causal_convolve(np.clip(vsw_anom, 0.0, None), run_kernel)

    # Diurnal term: peak near local noon (MLT ~ 12 h), ~diurnal_amplitude_dex in log10.
    mlt = _magnetic_local_time(time_index, params.longitude_deg)
    diurnal = params.diurnal_amplitude_dex * np.cos(2.0 * np.pi * (mlt - 12.0) / 24.0)

    # Semiannual (equinox) modulation, ~x2 in linear -> ~0.3 dex, peaks at equinoxes.
    doy = (
        time_index.dayofyear + (time_index.hour + time_index.minute / 60.0) / 24.0
    ).to_numpy(dtype="float64")
    semiannual = 0.15 * np.cos(4.0 * np.pi * (doy - 80.0) / 365.25)

    # Storm dropouts: prompt depletion when Pdyn spikes (shadowing) or SYM-H goes deep.
    pdyn_excess = np.clip(pdyn - 6.0, 0.0, None)
    sym_depth = np.clip(-sym_h - 30.0, 0.0, None)
    bs = np.clip(-bz, 0.0, None)
    dropout = 0.9 * np.tanh(0.25 * pdyn_excess) + 0.012 * sym_depth + 0.03 * bs
    # Dropouts persist for a few hours then recover (leaky integrator on the loss term).
    drop_kernel = _causal_decay_kernel(
        tau_steps=0.4 * steps_per_day, n_steps=int(1 * steps_per_day)
    )
    dropout = _causal_convolve(dropout, drop_kernel)

    # Seed coupling: an elevated seed population feeds the MeV channel (lagged a few h).
    seed_couple_kernel = _causal_decay_kernel(
        tau_steps=0.5 * steps_per_day, n_steps=int(2 * steps_per_day)
    )
    seed_couple = _causal_convolve(np.clip(seed_drive, 0.0, None), seed_couple_kernel)

    # Slowly-varying quiet baseline tied to the solar-cycle (more relativistic e- at
    # solar max declining phase); keep it modest so quiet periods sit at 10^1-10^2 pfu.
    solar_cycle = 0.5 * (1.0 - np.cos(2.0 * np.pi * days / (11.0 * 365.25)))
    baseline = 1.3 + 0.3 * solar_cycle

    # Acceleration drive (Vsw lag response + 2-day running mean + seed coupling). The
    # response SATURATES (acceleration efficiency rolls off at extreme driving) via a
    # tanh, so the >2 MeV flux peaks near 10^4-10^4.5 pfu rather than running away.
    drive = 2.2 * accel + 1.2 * vsw_run + 0.55 * seed_couple
    drive_amp = 2.2  # max additive log10 contribution from the acceleration drive
    drive_sat = drive_amp * np.tanh(drive / drive_amp)

    log_flux = baseline + drive_sat + diurnal + semiannual - dropout
    # Log-normal observational noise (heavy-ish tail handled downstream by Hampel).
    log_flux += 0.13 * rng.standard_normal(n)

    flux_e2 = np.power(10.0, log_flux)
    flux_seed = np.power(10.0, log_seed)

    # Re-floor through the canonical transform so log columns match the contract exactly.
    log_flux_e2 = np.asarray(log10_floor(flux_e2, floor=_LOG_FLOOR_PFU), dtype="float64")
    log_flux_seed = np.asarray(log10_floor(flux_seed, floor=_LOG_FLOOR_PFU), dtype="float64")

    return pd.DataFrame(
        {
            "flux_e2": flux_e2,
            "log_flux_e2": log_flux_e2,
            "flux_seed": flux_seed,
            "log_flux_seed": log_flux_seed,
            "mlt": mlt,
        },
        index=time_index,
    )


def generate_dataset(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    cadence_min: int = 5,
    *,
    seed: int = 0,
    longitude_deg: float = 83.0,
    sat_id: str = "SYN-GEO",
    with_gaps: bool = True,
    with_spikes: bool = True,
) -> pd.DataFrame:
    """Generate a canonical MERGED synthetic dataframe (CONTRACTS.md §2).

    Produces a physically-plausible, reproducible 5-min (or ``cadence_min``-min) dataset
    that conforms exactly to ``ps14.datasets.schema.validate_merged``: a UTC
    ``DatetimeIndex`` named ``"time"`` and every required column with the right dtype.

    Parameters
    ----------
    start, end:
        Inclusive-start/exclusive-end UTC bounds (anything ``pd.date_range`` accepts).
    cadence_min:
        Grid cadence in minutes (default 5, matching the project grid).
    seed:
        Seeds the deterministic RNG; identical seeds yield identical frames.
    longitude_deg:
        GEO sensor longitude in degrees (drives the MLT diurnal phase). Default 83 deg E
        (an Indian-longitude GEO slot).
    sat_id:
        Source-satellite identifier stored as a ``category`` column.
    with_gaps:
        If True, inject short + long NaN gaps into the raw science channels (long gaps
        stay NaN per the contract; the ``*_imputed`` masks remain 0 because nothing has
        been interpolated yet -- imputation happens downstream in ``preprocess``).
    with_spikes:
        If True, inject sparse multiplicative spikes the Hampel filter must remove.

    Returns
    -------
    pd.DataFrame
        Frame satisfying ``schema.validate_merged`` with ``flux_e2`` exceeding
        ``HARSH_PFU`` (1000 pfu) for a realistic minority of samples.
    """
    cadence = f"{cadence_min}min"
    params = SyntheticParams(
        start=str(start),
        end=str(end),
        cadence=cadence,
        seed=seed,
        longitude_deg=float(longitude_deg),
        sat_id=sat_id,
    )
    time_index = pd.date_range(start=start, end=end, freq=cadence, name="time")
    # date_range is inclusive of both ends; drop the closing endpoint for a half-open grid.
    if len(time_index) > 1 and pd.Timestamp(end) == time_index[-1]:
        time_index = time_index[:-1]
    time_index = pd.DatetimeIndex(time_index, name="time").as_unit("ns")

    drivers = generate_solar_wind(time_index, params)
    flux = generate_flux(time_index, drivers, params)

    df = pd.concat([flux, drivers], axis=1)
    df["longitude"] = float(longitude_deg)
    df["sat_id"] = pd.Categorical([sat_id] * len(df))

    # Enforce canonical column dtypes for the required science columns.
    for col in schema.MERGED_REQUIRED:
        df[col] = df[col].astype("float64")

    # Add zero-valued *_imputed masks for the dynamic science channels (nothing imputed
    # yet; ``longitude`` is static and never gapped, so it gets no mask).
    mask_targets = [c for c in schema.MERGED_REQUIRED if c != "longitude"]
    for col in mask_targets:
        df[f"{col}_imputed"] = np.zeros(len(df), dtype="int8")

    if with_spikes:
        df = _inject_spikes(df, params, np.random.default_rng(seed + 7))
    if with_gaps:
        df = _inject_gaps(df, params, np.random.default_rng(seed + 11))

    # Order columns: required science columns, then static sat_id, then imputed masks.
    mask_cols = [f"{c}_imputed" for c in mask_targets]
    ordered = [*schema.MERGED_REQUIRED, "sat_id", *mask_cols]
    df = df[ordered]
    df.index.name = "time"
    return df


# --------------------------------------------------------------------------------------
# Realism artifacts
# --------------------------------------------------------------------------------------


# Channels that receive injected spikes/gaps (the measured science channels, not derived
# static metadata or the log mirrors, which are recomputed downstream).
_ARTIFACT_COLUMNS: tuple[str, ...] = (
    "flux_e2",
    "flux_seed",
    "vsw",
    "density",
    "bz_gsm",
    "bt",
    "ae",
    "al",
    "sym_h",
)


def _inject_spikes(
    df: pd.DataFrame, params: SyntheticParams, rng: np.random.Generator
) -> pd.DataFrame:
    """Multiply a sparse random fraction of points by large factors (Hampel targets)."""
    out = df.copy()
    n = len(out)
    for col in _ARTIFACT_COLUMNS:
        if col not in out.columns:
            continue
        n_spikes = int(params.spike_fraction * n)
        if n_spikes <= 0:
            continue
        idx = rng.integers(0, n, size=n_spikes)
        factors = rng.uniform(3.0, 8.0, size=n_spikes)
        signs = rng.choice([-1.0, 1.0], size=n_spikes)
        vals = out[col].to_numpy(dtype="float64").copy()
        # For sign-definite channels (flux/density/bt/ae) spike upward; else random sign.
        if col in ("flux_e2", "flux_seed", "density", "bt", "ae"):
            vals[idx] = vals[idx] * factors
        else:
            vals[idx] = vals[idx] + signs * factors * (np.nanstd(vals) + 1e-9)
        out[col] = vals
    # Recompute the log mirrors so they remain consistent with the spiked linear flux.
    out["log_flux_e2"] = np.asarray(log10_floor(out["flux_e2"], floor=_LOG_FLOOR_PFU))
    out["log_flux_seed"] = np.asarray(log10_floor(out["flux_seed"], floor=_LOG_FLOOR_PFU))
    return out


def _inject_gaps(
    df: pd.DataFrame, params: SyntheticParams, rng: np.random.Generator
) -> pd.DataFrame:
    """Null out random short + long intervals of the science channels (left as NaN).

    Long gaps stay NaN per the contract; the ``*_imputed`` masks are untouched (0)
    because imputation is a downstream ``preprocess`` responsibility.
    """
    out = df.copy()
    n = len(out)
    if n == 0:
        return out
    target_nan = int(params.gap_fraction * n)
    if target_nan <= 0:
        return out

    steps_per_day = pd.Timedelta(days=1) / pd.Timedelta(params.cadence)
    short_max = max(int(steps_per_day / 48), 2)  # ~30 min
    long_max = max(int(steps_per_day / 2), short_max + 1)  # ~12 h

    log_for = {"flux_e2": "log_flux_e2", "flux_seed": "log_flux_seed"}

    for col in _ARTIFACT_COLUMNS:
        if col not in out.columns:
            continue
        vals = out[col].to_numpy(dtype="float64").copy()
        placed = 0
        guard = 0
        while placed < target_nan and guard < target_nan * 4:
            guard += 1
            if rng.random() < 0.7:
                length = int(rng.integers(1, short_max + 1))
            else:
                length = int(rng.integers(short_max + 1, long_max + 1))
            onset = int(rng.integers(0, n))
            sl = slice(onset, min(onset + length, n))
            vals[sl] = np.nan
            placed += sl.stop - sl.start
        out[col] = vals
        # Keep the log mirror consistent (NaN where linear is NaN).
        if col in log_for:
            lcol = log_for[col]
            lvals = out[lcol].to_numpy(dtype="float64").copy()
            lvals[np.isnan(vals)] = np.nan
            out[lcol] = lvals
    return out


def inject_artifacts(
    df: pd.DataFrame,
    params: SyntheticParams | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Inject realistic NaN gaps (short + long) and spikes for the cleaning path.

    Short gaps (<= a few samples) and long gaps (hours) are placed at random; a fraction
    of points are multiplied by large factors to create spikes the Hampel filter removes.
    This is provided as a standalone helper for the CDF-write / preprocess demo when a
    clean frame was produced via ``generate_dataset(..., with_gaps=False)``.
    """
    params = params or SyntheticParams()
    rng = rng or np.random.default_rng(params.seed)
    out = _inject_spikes(df, params, rng)
    out = _inject_gaps(out, params, rng)
    return out


# --------------------------------------------------------------------------------------
# CDF / Parquet writers
# --------------------------------------------------------------------------------------

# Variable groupings for the two ISTP files written from a merged frame.
GOES_VARIABLES: tuple[str, ...] = ("flux_e2", "flux_seed", "mlt")
OMNI_VARIABLES: tuple[str, ...] = (
    "vsw",
    "density",
    "bz_gsm",
    "bt",
    "ae",
    "al",
    "kp",
    "sym_h",
    "f107",
)

# Per-variable ISTP attributes for the writer (UNITS / valid range / human description).
_VAR_ATTRS: dict[str, dict[str, object]] = {
    "flux_e2": {"UNITS": "cm^-2 s^-1 sr^-1", "VALIDMIN": 0.0, "VALIDMAX": 1.0e8,
                "CATDESC": ">2 MeV integral electron flux at GEO"},
    "flux_seed": {"UNITS": "cm^-2 s^-1 sr^-1", "VALIDMIN": 0.0, "VALIDMAX": 1.0e9,
                  "CATDESC": "sub-MeV seed electron flux"},
    "mlt": {"UNITS": "hours", "VALIDMIN": 0.0, "VALIDMAX": 24.0,
            "CATDESC": "magnetic local time of the GEO sensor"},
    "vsw": {"UNITS": "km/s", "VALIDMIN": 100.0, "VALIDMAX": 1200.0,
            "CATDESC": "solar-wind bulk speed"},
    "density": {"UNITS": "cm^-3", "VALIDMIN": 0.0, "VALIDMAX": 200.0,
                "CATDESC": "solar-wind proton density"},
    "bz_gsm": {"UNITS": "nT", "VALIDMIN": -100.0, "VALIDMAX": 100.0,
               "CATDESC": "IMF Bz (GSM)"},
    "bt": {"UNITS": "nT", "VALIDMIN": 0.0, "VALIDMAX": 100.0, "CATDESC": "IMF magnitude |B|"},
    "ae": {"UNITS": "nT", "VALIDMIN": 0.0, "VALIDMAX": 5000.0,
           "CATDESC": "auroral electrojet index"},
    "al": {"UNITS": "nT", "VALIDMIN": -5000.0, "VALIDMAX": 500.0, "CATDESC": "AL index"},
    "kp": {"UNITS": " ", "VALIDMIN": 0.0, "VALIDMAX": 9.0, "CATDESC": "planetary K-index"},
    "sym_h": {"UNITS": "nT", "VALIDMIN": -1000.0, "VALIDMAX": 200.0,
              "CATDESC": "SYM-H (high-res Dst)"},
    "f107": {"UNITS": "sfu", "VALIDMIN": 50.0, "VALIDMAX": 400.0,
             "CATDESC": "F10.7 solar radio flux"},
}


def _datetime_to_tt2000(index: pd.DatetimeIndex) -> np.ndarray:
    """Convert a UTC ``DatetimeIndex`` to a leap-aware ``CDF_TIME_TT2000`` int64 array.

    Uses ``cdflib.cdfepoch.compute_tt2000`` from broken-down calendar components, which
    round-trips exactly through ``cdfepoch.to_datetime`` (verified across the 2015/2016
    leap seconds). ``timestamp_to_tt2000`` is avoided because its float input convention
    does not match Unix seconds.
    """
    import cdflib  # noqa: PLC0415  (lazy: only needed when writing CDFs)

    idx = pd.DatetimeIndex(index).as_unit("ns")
    ns = idx.to_numpy().astype("datetime64[ns]").view("int64")
    sec, rem = np.divmod(ns, 1_000_000_000)
    ms, rem = np.divmod(rem, 1_000_000)
    us, nsr = np.divmod(rem, 1_000)
    comps = [
        [t.year, t.month, t.day, t.hour, t.minute, t.second, int(a), int(b), int(c)]
        for t, a, b, c in zip(idx, ms, us, nsr)
    ]
    return np.asarray(cdflib.cdfepoch.compute_tt2000(comps), dtype="int64")


def write_cdf(
    df: pd.DataFrame,
    path: str | Path,
    *,
    kind: str,
    params: SyntheticParams | None = None,
) -> Path:
    """Write a synthetic frame to an ISTP-flavoured CDF (TT2000 epoch, fill/valid attrs).

    Parameters
    ----------
    df:
        UTC-indexed frame to write (a merged frame; only the relevant columns are taken).
    path:
        Output ``.cdf`` path (under ``data/raw/synthetic/``).
    kind:
        ``"goes"`` (flux variables) or ``"omni"`` (driver variables) -- controls variable
        names/attributes so the reader exercises the real discovery path.
    params:
        Generator parameters (for global attributes / provenance).

    Returns
    -------
    Path
        The written file path. Writes ``Epoch`` (TT2000), per-variable ``FILLVAL``,
        ``VALIDMIN``/``VALIDMAX``, ``UNITS``, ``DEPEND_0``, ``VAR_TYPE='data'``.
    """
    from cdflib import cdfwrite  # noqa: PLC0415  (lazy: only needed when writing CDFs)

    params = params or SyntheticParams()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    if kind == "goes":
        variables = [c for c in GOES_VARIABLES if c in df.columns]
    elif kind == "omni":
        variables = [c for c in OMNI_VARIABLES if c in df.columns]
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown CDF kind {kind!r} (expected 'goes' or 'omni')")

    index = pd.DatetimeIndex(df.index)
    epoch = _datetime_to_tt2000(index)

    writer = cdfwrite.CDF(str(out_path), cdf_spec={"Majority": "row_major"}, delete=True)
    writer.write_globalattrs(
        {
            "Project": {0: "BAH-2026 PS-14"},
            "Source_name": {0: params.sat_id},
            "Data_type": {0: kind},
            "Logical_source": {0: f"SYNTHETIC_{kind.upper()}"},
            "Generation_date": {0: pd.Timestamp.now("UTC").isoformat()},
            "TEXT": {0: "Synthetic physically-plausible dataset (ps14.io.synthetic)."},
        }
    )

    writer.write_var(
        {
            "Variable": "Epoch",
            "Data_Type": cdfwrite.CDF.CDF_TIME_TT2000,
            "Num_Elements": 1,
            "Rec_Vary": True,
            "Dim_Sizes": [],
            "Var_Type": "zVariable",
        },
        var_attrs={"VAR_TYPE": "support_data", "UNITS": "ns", "CATDESC": "UTC epoch (TT2000)"},
        var_data=epoch,
    )

    for var in variables:
        attrs = dict(_VAR_ATTRS.get(var, {}))
        attrs.update(
            {
                "VAR_TYPE": "data",
                "DEPEND_0": "Epoch",
                "FILLVAL": _FILLVAL,
                "FIELDNAM": var,
                "DISPLAY_TYPE": "time_series",
            }
        )
        data = df[var].to_numpy(dtype="float64").copy()
        data[np.isnan(data)] = _FILLVAL  # write FILLVAL where the frame had NaN gaps
        writer.write_var(
            {
                "Variable": var,
                "Data_Type": cdfwrite.CDF.CDF_DOUBLE,
                "Num_Elements": 1,
                "Rec_Vary": True,
                "Dim_Sizes": [],
                "Var_Type": "zVariable",
            },
            var_attrs=attrs,
            var_data=data,
        )

    writer.close()
    logger.info("wrote synthetic %s CDF -> %s (%d records)", kind, out_path, len(index))
    return out_path


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Write the merged frame to Parquet (the fast mirror of the CDF, CONTRACTS.md §8)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    logger.info("wrote synthetic parquet -> %s (%d records)", out_path, len(df))
    return out_path


def make_demo_dataset(
    out_dir: str | Path = "data/raw/synthetic",
    *,
    years: float = 1.0,
    start: str = "2014-01-01",
    cadence_min: int = 5,
    seed: int = 1993,
    longitude_deg: float = 83.0,
    sat_id: str = "SYN-GEO",
    write_cdf_files: bool = True,
    write_parquet_file: bool = True,
) -> dict[str, Path]:
    """Generate + persist a demo dataset: GOES-like + OMNI-like CDFs and a merged parquet.

    Parameters
    ----------
    out_dir:
        Output directory for the artifacts.
    years:
        Length of the generated span in years (from ``start``).
    start:
        Start timestamp (UTC).
    cadence_min, seed, longitude_deg, sat_id:
        Passed through to :func:`generate_dataset`.
    write_cdf_files:
        Write the GOES/OMNI CDFs (exercises ``cdf_reader``). Falls back gracefully to
        parquet-only with a warning if the cdflib writer is unavailable.
    write_parquet_file:
        Write the merged Parquet mirror.

    Returns
    -------
    dict[str, Path]
        Map of artifact name -> path, e.g. ``{"goes_cdf": ..., "omni_cdf": ...,
        "merged_parquet": ...}``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    end = (pd.Timestamp(start) + pd.Timedelta(days=365.25 * years)).isoformat()

    df = generate_dataset(
        start,
        end,
        cadence_min=cadence_min,
        seed=seed,
        longitude_deg=longitude_deg,
        sat_id=sat_id,
        with_gaps=True,
        with_spikes=True,
    )
    params = SyntheticParams(
        start=str(start), end=str(end), cadence=f"{cadence_min}min", seed=seed,
        longitude_deg=float(longitude_deg), sat_id=sat_id,
    )

    artifacts: dict[str, Path] = {}
    if write_parquet_file:
        artifacts["merged_parquet"] = write_parquet(df, out_dir / "merged.parquet")

    if write_cdf_files:
        try:
            artifacts["goes_cdf"] = write_cdf(df, out_dir / "synthetic_goes.cdf",
                                              kind="goes", params=params)
            artifacts["omni_cdf"] = write_cdf(df, out_dir / "synthetic_omni.cdf",
                                              kind="omni", params=params)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "CDF writing failed (%s); continuing with parquet only. "
                "CDF output is optional but recommended (cdf_reader round-trip).",
                exc,
            )

    return artifacts


def generate(
    params: SyntheticParams | None = None,
    *,
    out_dir: str | Path = "data/raw/synthetic",
    write_parquet: bool = True,
    write_cdf_files: bool = True,
) -> dict[str, Path]:
    """End-to-end synthetic generation: drivers -> flux -> artifacts -> CDF + parquet.

    Thin wrapper over :func:`make_demo_dataset` that honours a :class:`SyntheticParams`
    (the ``make synth-data`` / ``ps14 synth-data`` entry point).

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
        Map of artifact name -> path.

    Notes
    -----
    Deterministic given ``params.seed``.
    """
    params = params or SyntheticParams()
    years = (pd.Timestamp(params.end) - pd.Timestamp(params.start)) / pd.Timedelta(days=365.25)
    cadence_min = int(pd.Timedelta(params.cadence) / pd.Timedelta(minutes=1))
    return make_demo_dataset(
        out_dir,
        years=float(years),
        start=params.start,
        cadence_min=cadence_min,
        seed=params.seed,
        longitude_deg=params.longitude_deg,
        sat_id=params.sat_id,
        write_cdf_files=write_cdf_files,
        write_parquet_file=write_parquet,
    )


def replay_stream(
    params: SyntheticParams | None = None, *, window_hours: float = 24.0
) -> pd.DataFrame:
    """Produce a short, SWPC-shaped synthetic stream for offline serving/replay.

    Returns the most recent ``window_hours`` of synthetic flux + solar wind shaped like
    the parsed SWPC feeds (CONTRACTS.md §7) so the serving path runs with no network.

    Parameters
    ----------
    params:
        Generator parameters; defaults to :class:`SyntheticParams`.
    window_hours:
        Length of the trailing window to return.

    Returns
    -------
    pd.DataFrame
        A merged-schema frame for the trailing window (the serving path can map the
        columns it needs: ``flux_e2``, ``vsw``, ``bz_gsm``, ``kp``, ``mlt``).
    """
    params = params or SyntheticParams()
    end = pd.Timestamp.now("UTC").tz_convert(None).floor(params.cadence)
    start = end - pd.Timedelta(hours=window_hours)
    cadence_min = int(pd.Timedelta(params.cadence) / pd.Timedelta(minutes=1))
    return generate_dataset(
        start.isoformat(),
        end.isoformat(),
        cadence_min=cadence_min,
        seed=params.seed,
        longitude_deg=params.longitude_deg,
        sat_id=params.sat_id,
        with_gaps=False,
        with_spikes=False,
    )


def exceedance_fraction(df: pd.DataFrame, threshold_pfu: float = HARSH_PFU) -> float:
    """Fraction of non-NaN ``flux_e2`` samples that exceed ``threshold_pfu`` (diagnostic)."""
    flux = df["flux_e2"].to_numpy(dtype="float64")
    valid = flux[~np.isnan(flux)]
    if valid.size == 0:
        return 0.0
    return float((valid >= threshold_pfu).mean())


__all__ = [
    "SyntheticParams",
    "generate_solar_wind",
    "generate_flux",
    "generate_dataset",
    "inject_artifacts",
    "write_cdf",
    "write_parquet",
    "make_demo_dataset",
    "generate",
    "replay_stream",
    "exceedance_fraction",
    "GOES_VARIABLES",
    "OMNI_VARIABLES",
]
