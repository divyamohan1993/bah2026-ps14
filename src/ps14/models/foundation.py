"""Backup model B: fine-tuned time-series foundation model (R3 §4/§11).

Few-shot route needing only ~months of data. Options:
  * **Chronos-Bolt** (Amazon, T5, Apache-2.0) — native quantiles + covariates, fast.
  * **TimesFM+Cov** (Google) — validated on this exact target (R^2 ~ 0.90 @ 6 h).
Pair with the exceedance head to fix the known storm-peak underestimation / onset lag
(R3 §4 caveat). Requires the ``[dl]`` extra (and the model weights / AutoGluon-TS).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ps14.constants import HORIZON_NAMES, QUANTILES
from ps14.models.base import Forecaster


class FoundationForecaster(Forecaster):
    """Fine-tuned foundation-model forecaster implementing the :class:`Forecaster` contract.

    Parameters
    ----------
    params:
        Hyperparameters (``backbone`` in {``"chronos_bolt"``, ``"timesfm"``},
        fine-tune steps, context length, covariate handling, ...).
    """

    name = "foundation"
    horizon_names = HORIZON_NAMES

    def __init__(
        self, params: dict | None = None, quantiles: tuple[float, ...] = QUANTILES
    ) -> None:
        self.params = params or {}
        self.backbone = self.params.get("backbone", "chronos_bolt")
        self.quantiles = quantiles
        self._model = None

    def fit(self, X, X_future, y, y_exceed, *, val=None) -> FoundationForecaster:  # noqa: D102
        raise NotImplementedError(
            "TODO: load the pretrained backbone (Chronos-Bolt / TimesFM); few-shot fine-tune on "
            "the target series with covariates; attach an exceedance head/post-processor (R3 §4)."
        )

    def predict(self, X, X_future) -> np.ndarray:  # noqa: D102
        raise NotImplementedError("TODO: median forecast -> [N, n_h].")

    def predict_quantiles(self, X, X_future) -> dict[float, np.ndarray]:  # noqa: D102
        raise NotImplementedError("TODO: native quantiles (Chronos-Bolt) -> {tau: [N, n_h]}.")

    def predict_proba_exceed(self, X, X_future) -> np.ndarray:  # noqa: D102
        raise NotImplementedError("TODO: exceedance head/post-processor -> [N, n_h] in [0,1].")

    def save(self, path: str | Path) -> None:  # noqa: D102
        raise NotImplementedError("TODO: persist fine-tuned weights + config.")

    @classmethod
    def load(cls, path: str | Path) -> FoundationForecaster:  # noqa: D102
        raise NotImplementedError("TODO: load into a new FoundationForecaster.")


__all__ = ["FoundationForecaster"]
