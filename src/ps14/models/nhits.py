"""Backup model A: N-HiTS / TiDE direct multi-horizon forecaster (R3 §11).

Fast, simple, strong direct multi-horizon; ~50x faster than transformers and a robust
fallback if the TFT overfits the limited data. Uses quantile loss for uncertainty and an
auxiliary exceedance head. Built on neuralforecast/darts (or pytorch-forecasting N-HiTS).
Requires the ``[dl]`` extra.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ps14.constants import HORIZON_NAMES, QUANTILES
from ps14.models.base import Forecaster


class NHiTSForecaster(Forecaster):
    """N-HiTS (or TiDE via ``variant``) implementing the :class:`Forecaster` contract."""

    name = "nhits"
    horizon_names = HORIZON_NAMES

    def __init__(
        self, params: dict | None = None, quantiles: tuple[float, ...] = QUANTILES
    ) -> None:
        self.params = params or {}
        self.variant = self.params.get("variant", "nhits")  # 'nhits' | 'tide'
        self.quantiles = quantiles
        self._model = None

    def fit(self, X, X_future, y, y_exceed, *, val=None) -> NHiTSForecaster:  # noqa: D102
        raise NotImplementedError(
            "TODO: construct NHITS/TiDE (neuralforecast or pytorch-forecasting) with a quantile "
            "loss over self.quantiles + an exceedance head; train with early stopping (R3 §11)."
        )

    def predict(self, X, X_future) -> np.ndarray:  # noqa: D102
        raise NotImplementedError("TODO: return P50 -> [N, n_h].")

    def predict_quantiles(self, X, X_future) -> dict[float, np.ndarray]:  # noqa: D102
        raise NotImplementedError("TODO: return {tau: [N, n_h]}.")

    def predict_proba_exceed(self, X, X_future) -> np.ndarray:  # noqa: D102
        raise NotImplementedError("TODO: exceedance head -> [N, n_h] in [0,1].")

    def export_onnx(self, path: str | Path) -> Path:
        """Export to ONNX for serving (R4 §3.1)."""
        raise NotImplementedError("TODO: torch.onnx.export of the trained N-HiTS/TiDE.")

    def save(self, path: str | Path) -> None:  # noqa: D102
        raise NotImplementedError("TODO: persist the trained model.")

    @classmethod
    def load(cls, path: str | Path) -> NHiTSForecaster:  # noqa: D102
        raise NotImplementedError("TODO: load into a new NHiTSForecaster.")


__all__ = ["NHiTSForecaster"]
