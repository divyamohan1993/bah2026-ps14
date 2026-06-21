"""Project-wide constants: channels, thresholds, horizons, units.

These are the single source of truth for magic numbers used across modules. Import
from here rather than hard-coding. See CONTRACTS.md for the binding meanings.
"""

from __future__ import annotations

import math
from typing import Final

# --------------------------------------------------------------------------------------
# Cadence / grid
# --------------------------------------------------------------------------------------
CADENCE: Final[str] = "5min"
CADENCE_MINUTES: Final[int] = 5
STEPS_PER_HOUR: Final[int] = 60 // CADENCE_MINUTES  # 12
STEPS_PER_DAY: Final[int] = STEPS_PER_HOUR * 24  # 288

# --------------------------------------------------------------------------------------
# Forecast horizons (in 5-minute steps). nowcast = 8 steps = 40 min (30-45 min band).
# Read off a single 12 h MIMO decoder (decoder_steps = 144). See ARCHITECTURE.md (e).
# --------------------------------------------------------------------------------------
HORIZON_STEPS: Final[dict[str, int]] = {
    "nowcast": 8,  # 40 min
    "6h": 6 * STEPS_PER_HOUR,  # 72
    "12h": 12 * STEPS_PER_HOUR,  # 144
}
HORIZON_NAMES: Final[list[str]] = list(HORIZON_STEPS.keys())
HORIZON_LEAD_MINUTES: Final[dict[str, int]] = {
    name: steps * CADENCE_MINUTES for name, steps in HORIZON_STEPS.items()
}
DECODER_STEPS: Final[int] = HORIZON_STEPS["12h"]  # full MIMO decoder length
DEFAULT_LOOKBACK_STEPS: Final[int] = 4 * STEPS_PER_DAY  # 1152 = 4 days (R1 §8)

# --------------------------------------------------------------------------------------
# Thresholds (the "harsh" radiation alert) — R1 §6, R3 §0.
# --------------------------------------------------------------------------------------
HARSH_PFU: Final[float] = 1000.0  # NOAA SWPC >2 MeV alert threshold
LOG_HARSH: Final[float] = math.log10(HARSH_PFU)  # 3.0 in log10 space
SUSTAINED_PERIODS: Final[int] = 3  # >= 3 consecutive 5-min readings
DAILY_FLUENCE_ALERT: Final[float] = 1.0e9  # significant daily fluence (electrons/cm^2/day/sr)

# --------------------------------------------------------------------------------------
# Units (read UNITS from the CDF when present; these are the canonical strings).
# --------------------------------------------------------------------------------------
UNIT_FLUX: Final[str] = "cm^-2 s^-1 sr^-1"  # pfu for integral electron flux
UNIT_SPEED: Final[str] = "km/s"
UNIT_DENSITY: Final[str] = "cm^-3"
UNIT_PRESSURE: Final[str] = "nPa"
UNIT_BFIELD: Final[str] = "nT"
UNIT_INDEX: Final[str] = "nT"

# --------------------------------------------------------------------------------------
# Channels / target naming. The >2 MeV integral channel is the target ("E2" lineage).
# --------------------------------------------------------------------------------------
TARGET_FLUX_COLUMN: Final[str] = "flux_e2"  # linear pfu
TARGET_COLUMN: Final[str] = "log_flux_e2"  # log10 pfu — the model target
SEED_FLUX_COLUMN: Final[str] = "flux_seed"  # sub-MeV precursor channel (linear)
SEED_LOG_COLUMN: Final[str] = "log_flux_seed"

# Physical lag of the Vsw -> flux acceleration response (R1 §2): 1-2 days.
VSW_FLUX_LAG_DAYS: Final[tuple[float, float]] = (1.0, 2.0)
# L1 -> magnetosphere propagation lead (R1 §3): ~20-90 min, mean ~47 min.
L1_LEAD_MINUTES_RANGE: Final[tuple[float, float]] = (20.0, 90.0)
L1_LEAD_MINUTES_MEAN: Final[float] = 47.0

# --------------------------------------------------------------------------------------
# Default quantiles for the regression head (pinball loss).
# --------------------------------------------------------------------------------------
QUANTILES: Final[tuple[float, ...]] = (0.1, 0.5, 0.9)
MEDIAN_QUANTILE: Final[float] = 0.5

__all__ = [
    "CADENCE",
    "CADENCE_MINUTES",
    "STEPS_PER_HOUR",
    "STEPS_PER_DAY",
    "HORIZON_STEPS",
    "HORIZON_NAMES",
    "HORIZON_LEAD_MINUTES",
    "DECODER_STEPS",
    "DEFAULT_LOOKBACK_STEPS",
    "HARSH_PFU",
    "LOG_HARSH",
    "SUSTAINED_PERIODS",
    "DAILY_FLUENCE_ALERT",
    "UNIT_FLUX",
    "UNIT_SPEED",
    "UNIT_DENSITY",
    "UNIT_PRESSURE",
    "UNIT_BFIELD",
    "UNIT_INDEX",
    "TARGET_FLUX_COLUMN",
    "TARGET_COLUMN",
    "SEED_FLUX_COLUMN",
    "SEED_LOG_COLUMN",
    "VSW_FLUX_LAG_DAYS",
    "L1_LEAD_MINUTES_RANGE",
    "L1_LEAD_MINUTES_MEAN",
    "QUANTILES",
    "MEDIAN_QUANTILE",
]
