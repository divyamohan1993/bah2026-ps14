"""Training orchestration: load windows -> split -> scale (train-only) -> fit -> export.

Ties the pipeline together for any :class:`~ps14.models.base.Forecaster` selected by
``config.model.name``. Enforces the leakage-critical contract: chronological split with
purge/embargo, scalers fit on TRAIN only (R3 §8e, R5 §4.9).
"""

from __future__ import annotations

from pathlib import Path

from ps14.config import Settings
from ps14.models.base import Forecaster


def build_model(config: Settings) -> Forecaster:
    """Instantiate the forecaster named by ``config.model.name``.

    Dispatch: ``persistence|climatology|lightgbm|refm`` -> baselines; ``tft`` ->
    DualHeadTFT; ``nhits`` -> NHiTSForecaster; ``foundation`` -> FoundationForecaster.
    Per-model hyperparameters come from ``config.model.params``.
    """
    raise NotImplementedError(
        "TODO: map config.model.name -> the right Forecaster subclass, passing config.model.params."
    )


def train(config: Settings) -> Forecaster:
    """Full training run for the configured model.

    Steps:
      1. load ``data/processed/windows.npz`` (:func:`windowing.load_windows`);
      2. chronological split with embargo (:func:`windowing.chronological_split`);
      3. fit a scaler on TRAIN only and transform all splits (:mod:`preprocess.transform`);
      4. ``model = build_model(config)``; ``model.fit(train..., val=val...)``;
      5. save the model (+ ONNX export for DL models) and the scaler.

    Returns
    -------
    Forecaster
        The fitted model.
    """
    raise NotImplementedError(
        "TODO: orchestrate load -> split -> train-only scaling -> build_model -> "
        "fit -> save/export (R3 §8e, R5 §4.9; ARCHITECTURE.md (k) phase 5)."
    )


def main(config_path: str | Path = "config/default.yaml") -> None:
    """CLI entry point: load config and run :func:`train`."""
    raise NotImplementedError("TODO: cfg = load_config(config_path); train(cfg).")


__all__ = ["build_model", "train", "main"]
