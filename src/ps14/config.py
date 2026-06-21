"""Configuration loading + validation (pydantic).

Loads ``config/default.yaml`` (and optionally a per-model YAML merged over ``model:``)
into a validated :class:`Settings` object. Import the schema everywhere rather than
indexing raw dicts, so a config change is a typed, single-point edit.

See ARCHITECTURE.md (j) for the documented schema and config/default.yaml for values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------------------
# Section models. Defaults mirror config/default.yaml; pydantic validates types/ranges.
# --------------------------------------------------------------------------------------


class PathsConfig(BaseModel):
    """Filesystem locations for data, models, and reports."""

    data_raw: Path = Path("data/raw")
    data_interim: Path = Path("data/interim")
    data_processed: Path = Path("data/processed")
    models: Path = Path("models")
    reports: Path = Path("reports")


class HampelConfig(BaseModel):
    window: int = 7
    n_sigma: float = 3.0
    replace: str = "nan"  # 'nan' | 'median'


class GapConfig(BaseModel):
    max_gap_steps: int = 6  # 30 min at 5-min cadence; longer stays NaN


class L1ToGeoConfig(BaseModel):
    method: str = "omni_preshifted"  # 'omni_preshifted' | 'ballistic'
    target_distance_re: float = 0.0


class PreprocessConfig(BaseModel):
    cadence: str = "5min"
    hampel: HampelConfig = Field(default_factory=HampelConfig)
    gaps: GapConfig = Field(default_factory=GapConfig)
    log_floor_pfu: float = 0.01
    l1_to_geo: L1ToGeoConfig = Field(default_factory=L1ToGeoConfig)


class FeaturesConfig(BaseModel):
    lookback_steps: int = 1152  # 4 days
    lags_steps: list[int] = Field(default_factory=lambda: [1, 6, 72, 288, 576])
    roll_windows: list[int] = Field(default_factory=lambda: [12, 72, 288, 576])
    coupling: list[str] = Field(default_factory=lambda: ["vbs", "newell", "epsilon"])
    difference_target: bool = False


class ModelConfig(BaseModel):
    name: str = "tft"
    horizons_steps: dict[str, int] = Field(
        default_factory=lambda: {"nowcast": 8, "6h": 72, "12h": 144}
    )
    decoder_steps: int = 144
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    # Per-model hyperparameters merged from config/model/<name>.yaml live here, untyped.
    params: dict[str, Any] = Field(default_factory=dict)


class ThresholdsConfig(BaseModel):
    harsh_pfu: float = 1000.0
    sustained_periods: int = 3
    exceedance_loss_weight: float = 1.0


class SplitConfig(BaseModel):
    mode: str = "chronological"
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15
    embargo_steps: int = 1296


class OnnxConfig(BaseModel):
    threads_intra: int = 2
    threads_inter: int = 1
    opt_level: str = "all"


class ServingConfig(BaseModel):
    refresh_seconds: int = 60
    poll: dict[str, int] = Field(
        default_factory=lambda: {"electrons": 60, "solar_wind": 60, "kp": 600}
    )
    cache: str = "dict"  # 'dict' | 'redis'
    redis_url: str = "redis://localhost:6379/0"
    cache_decimals: int = 3
    onnx: OnnxConfig = Field(default_factory=OnnxConfig)
    source: str = "synthetic"  # 'synthetic' | 'swpc'
    host: str = "0.0.0.0"
    port: int = 8000


class DataConfig(BaseModel):
    sources: dict[str, str] = Field(default_factory=dict)
    time_range: dict[str, str] = Field(default_factory=dict)
    satellites: list[str] = Field(default_factory=list)


class SyntheticConfig(BaseModel):
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


class Settings(BaseModel):
    """Top-level validated configuration. Build with :func:`load_config`."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    seed: int = 1993
    log_level: str = "INFO"


# --------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------

DEFAULT_CONFIG_PATH: Path = Path("config/default.yaml")
MODEL_CONFIG_DIR: Path = Path("config/model")


def _read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dict (empty dict if the file is empty)."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def load_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    merge_model_yaml: bool = True,
) -> Settings:
    """Load + validate the YAML configuration into a :class:`Settings`.

    Parameters
    ----------
    path:
        Path to the top-level YAML (default ``config/default.yaml``).
    merge_model_yaml:
        If True and ``config/model/<model.name>.yaml`` exists, merge it into
        ``settings.model.params`` so per-model hyperparameters are available.

    Returns
    -------
    Settings
        A validated configuration object.
    """
    raw = _read_yaml(path)
    settings = Settings(**raw)

    if merge_model_yaml:
        model_yaml = MODEL_CONFIG_DIR / f"{settings.model.name}.yaml"
        if model_yaml.exists():
            settings.model.params = _read_yaml(model_yaml)

    return settings


__all__ = [
    "Settings",
    "PathsConfig",
    "DataConfig",
    "SyntheticConfig",
    "PreprocessConfig",
    "HampelConfig",
    "GapConfig",
    "L1ToGeoConfig",
    "FeaturesConfig",
    "ModelConfig",
    "ThresholdsConfig",
    "SplitConfig",
    "ServingConfig",
    "OnnxConfig",
    "load_config",
    "DEFAULT_CONFIG_PATH",
]
