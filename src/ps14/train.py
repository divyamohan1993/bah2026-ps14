"""Training orchestration: load windows -> split -> scale (train-only) -> fit -> save.

Config-driven entry point for any :class:`~ps14.models.base.Forecaster` selected by name.
Enforces the leakage-critical contract: chronological split with purge/embargo, and a
per-channel scaler fit on the TRAIN split only (R3 §8e, R5 §4.9). The deep-learning models
receive the scaled encoder tensors; the scale-robust baselines (tree / linear / climatology)
are trained on the raw tensors. Deterministic seeds are set from ``config.seed``.

The model dispatch table is :data:`MODEL_REGISTRY` (``name -> class``); the CLI ``train``
stage calls :func:`main`.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ps14.config import Settings, load_config
from ps14.datasets import windowing
from ps14.models.base import Forecaster
from ps14.models.baselines import (
    Climatology,
    LightGBMForecaster,
    Persistence,
    REFMForecaster,
)
from ps14.models.foundation import FoundationForecaster
from ps14.models.nhits import NHiTSForecaster
from ps14.models.tft import TFTForecaster
from ps14.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from ps14.datasets.windowing import WindowTensors

logger = get_logger("ps14.train")

# --------------------------------------------------------------------------------------
# Model registry (CONTRACTS.md §5). config.model.name -> Forecaster subclass.
# --------------------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, type[Forecaster]] = {
    "persistence": Persistence,
    "climatology": Climatology,
    "lightgbm": LightGBMForecaster,
    "refm": REFMForecaster,
    "tft": TFTForecaster,
    "nhits": NHiTSForecaster,
    "foundation": FoundationForecaster,
}

# Models that benefit from train-only standardization of the encoder tensor.
_SCALE_MODELS = frozenset({"tft", "nhits"})


def set_seeds(seed: int) -> None:
    """Set Python / NumPy / (optional) torch seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - deliberately seed the global legacy RNG too
    try:  # torch is optional; only seed it if present.
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@dataclass
class TensorScaler:
    """Per-channel standardizer for the ``[N, L, F]`` encoder tensor (fit on TRAIN only).

    ``X_future`` (cyclic, already in ``[-1, 1]``) and the targets are left unscaled, so the
    model still predicts in native log10 space and the metric suite needs no inverse step.
    """

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> TensorScaler:
        """Fit per-channel mean/std over the ``(N, L)`` axes of ``X`` (NaN-aware)."""
        X = np.asarray(X, dtype="float64")
        mean = np.nanmean(X, axis=(0, 1))
        std = np.nanstd(X, axis=(0, 1))
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean.astype("float32"), std=std.astype("float32"))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize ``X`` channel-wise; NaNs are replaced with 0 (the scaled mean)."""
        X = np.asarray(X, dtype="float32")
        out = (X - self.mean[None, None, :]) / self.std[None, None, :]
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

    def save(self, path: str | Path) -> None:
        """Persist the scaler statistics (CONTRACTS.md §8)."""
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str | Path) -> TensorScaler:
        """Load the scaler statistics from ``path``."""
        data = np.load(path)
        return cls(mean=data["mean"], std=data["std"])


def _model_params(config: Settings, name: str) -> dict:
    """Resolve hyperparameters for ``name``, independent of ``config.model.name``.

    When ``name`` matches the configured model we reuse the already-merged
    ``config.model.params``; otherwise we read ``config/model/<file>.yaml`` directly so
    requesting a *different* model (e.g. scoring the whole baseline tier) never leaks the
    configured model's hyperparameters. The baseline tier (persistence/climatology/lightgbm/
    refm) all live in ``baseline.yaml`` under their own sub-key.
    """
    from ps14.config import MODEL_CONFIG_DIR, _read_yaml

    if name == config.model.name.lower() and config.model.params:
        raw = dict(config.model.params)
    else:
        is_baseline = name in {"persistence", "climatology", "lightgbm", "refm"}
        file_stem = "baseline" if is_baseline else name
        yaml_path = MODEL_CONFIG_DIR / f"{file_stem}.yaml"
        raw = _read_yaml(yaml_path) if yaml_path.exists() else {}
    # For the baseline tier, descend into the model's own sub-section.
    if name in {"persistence", "climatology", "lightgbm", "refm"} and name in raw:
        return dict(raw[name])
    return raw


def build_model(config: Settings, model_name: str | None = None) -> Forecaster:
    """Instantiate the forecaster named by ``model_name`` (or ``config.model.name``).

    Per-model hyperparameters are resolved by :func:`_model_params`. Decoder/lookback come
    from the config where the model accepts them.
    """
    name = (model_name or config.model.name).lower()
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)}.")
    cls = MODEL_REGISTRY[name]
    params = _model_params(config, name)
    decoder_steps = int(config.model.decoder_steps)

    if name in {"tft", "nhits", "foundation"}:
        return cls(params=params, decoder_steps=decoder_steps, seed=config.seed)  # type: ignore[call-arg]
    if name == "lightgbm":
        # Drop non-LightGBM bookkeeping keys (e.g. an 'enabled' flag from baseline.yaml).
        lgbm = {k: v for k, v in params.items() if k != "enabled"}
        return LightGBMForecaster(params=lgbm)
    if name == "climatology":
        return Climatology(
            n_tod_bins=params.get("n_tod_bins", 24),
            n_kp_bins=params.get("n_kp_bins", 10),
            n_doy_bins=params.get("n_doy_bins", 24),
        )
    if name == "refm":
        return REFMForecaster(recent_steps=params.get("vsw_history_steps", 288))
    return cls()  # persistence


def _resolve_windows(
    config: Settings,
    windows_path: str | Path | None,
    windows: WindowTensors | None,
) -> WindowTensors:
    """Return the window tensors from an in-memory object or by loading the NPZ."""
    if windows is not None:
        return windows
    path = Path(windows_path) if windows_path else (config.paths.data_processed / "windows.npz")
    logger.info("Loading window tensors from %s", path)
    return windowing.load_windows(path)


def train(
    config: Settings,
    model_name: str | None = None,
    windows_path: str | Path | None = None,
    *,
    windows: WindowTensors | None = None,
    save: bool = True,
) -> Forecaster:
    """Run a full training pass for one configured model.

    Steps:
      1. load the window tensors (:func:`windowing.load_windows`) or use ``windows``;
      2. chronological split with embargo (:func:`windowing.chronological_split`);
      3. fit a per-channel scaler on TRAIN only and transform all splits for DL models;
      4. ``model = build_model(...)``; ``model.fit(train..., val=val...)``;
      5. save the model (and scaler) under ``config.paths.models``.

    Parameters
    ----------
    config:
        Validated :class:`~ps14.config.Settings`.
    model_name:
        Override for ``config.model.name`` (e.g. when scoring the baseline tier).
    windows_path:
        Optional explicit path to ``windows.npz`` (defaults to the processed-data dir).
    windows:
        Optional in-memory :class:`~ps14.datasets.windowing.WindowTensors` (skips loading).
    save:
        Persist the fitted model + scaler when True.

    Returns
    -------
    Forecaster
        The fitted model.
    """
    name = (model_name or config.model.name).lower()
    set_seeds(config.seed)
    wt = _resolve_windows(config, windows_path, windows)

    train_idx, val_idx, _ = windowing.chronological_split(
        wt.t_index,
        train=config.split.train,
        val=config.split.val,
        embargo_steps=config.split.embargo_steps,
    )
    if train_idx.size == 0:
        raise ValueError("Chronological split produced an empty TRAIN set; check split/embargo.")
    logger.info("Training %s on %d train / %d val windows", name, train_idx.size, val_idx.size)

    Xtr, Xf_tr, ytr, ye_tr = (
        wt.X[train_idx],
        wt.X_future[train_idx],
        wt.y[train_idx],
        wt.y_exceed[train_idx],
    )
    has_val = val_idx.size > 0
    if has_val:
        Xva, Xf_va, yva, ye_va = (
            wt.X[val_idx],
            wt.X_future[val_idx],
            wt.y[val_idx],
            wt.y_exceed[val_idx],
        )

    scaler: TensorScaler | None = None
    if name in _SCALE_MODELS:
        scaler = TensorScaler.fit(Xtr)
        Xtr = scaler.transform(Xtr)
        if has_val:
            Xva = scaler.transform(Xva)

    model = build_model(config, model_name=name)
    val_tuple = (Xva, Xf_va, yva, ye_va) if has_val else None
    model.fit(Xtr, Xf_tr, ytr, ye_tr, val=val_tuple)

    if save:
        models_dir = Path(config.paths.models)
        models_dir.mkdir(parents=True, exist_ok=True)
        suffix = _artifact_suffix(name)
        model_path = models_dir / f"{name}{suffix}"
        model.save(model_path)
        logger.info("Saved %s model to %s", name, model_path)
        if scaler is not None:
            scaler_path = models_dir / f"scaler_{name}.npz"
            scaler.save(scaler_path)
            logger.info("Saved TRAIN-only scaler to %s", scaler_path)

    return model


def _artifact_suffix(name: str) -> str:
    """Filename suffix per model artifact format (CONTRACTS.md §8)."""
    if name == "persistence":
        return ".npz"
    if name == "climatology":
        return ".json"
    return ".joblib"


def main(config_path: str | Path = "config/default.yaml", model_name: str | None = None) -> int:
    """CLI entry point: load config and run :func:`train`."""
    config = load_config(config_path)
    train(config, model_name=model_name)
    return 0


__all__ = [
    "MODEL_REGISTRY",
    "TensorScaler",
    "build_model",
    "train",
    "set_seeds",
    "main",
]
