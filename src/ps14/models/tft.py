"""Primary model: dual-head Temporal Fusion Transformer (TFT).

Quantile-regression head (P10/P50/P90 of log10 flux) + focal-BCE exceedance head
(P(flux >= 1000 pfu)) per horizon, direct multi-horizon (MIMO), with native known-future
covariates and an interpretable variable-selection network (R3 §3c/§11, ARCHITECTURE.md
(e.1)). Built on pytorch-forecasting + PyTorch-Lightning. Requires the ``[dl]`` extra.

The combined loss (R3 §6-7):
    L = sum_h sum_tau pinball_tau(y_h, q_{tau,h})
        + lambda * sum_h focal_BCE(1[y_h >= LOG_HARSH], p_h)
optionally with inverse-frequency / output-weighting on the regression term to fight
storm-peak underestimation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ps14.constants import HORIZON_NAMES, QUANTILES
from ps14.models.base import Forecaster


class DualHeadTFT(Forecaster):
    """Dual-head TFT forecaster implementing the :class:`Forecaster` contract.

    Parameters
    ----------
    params:
        Hyperparameters merged from ``config/model/tft.yaml`` (hidden_size, lstm_layers,
        attention_head_size, dropout, learning_rate, focal_gamma, exceedance_weight, ...).
    quantiles:
        Output quantiles for the regression head.
    """

    name = "tft-dualhead"
    horizon_names = HORIZON_NAMES

    def __init__(
        self, params: dict | None = None, quantiles: tuple[float, ...] = QUANTILES
    ) -> None:
        self.params = params or {}
        self.quantiles = quantiles
        self._model = None  # the underlying LightningModule / TFT
        self._trainer = None

    def fit(self, X, X_future, y, y_exceed, *, val=None) -> DualHeadTFT:  # noqa: D102
        raise NotImplementedError(
            "TODO: wrap the tensors in a pytorch_forecasting TimeSeriesDataSet with "
            "static/known-future/observed-past channels; build TemporalFusionTransformer with "
            "QuantileLoss + a custom focal-BCE exceedance head; train via a Lightning Trainer "
            "with early stopping (ARCHITECTURE.md (e.1))."
        )

    def predict(self, X, X_future) -> np.ndarray:  # noqa: D102
        raise NotImplementedError("TODO: return the P50 quantile -> [N, n_h].")

    def predict_quantiles(self, X, X_future) -> dict[float, np.ndarray]:  # noqa: D102
        raise NotImplementedError("TODO: return {tau: [N, n_h]} from the quantile head.")

    def predict_proba_exceed(self, X, X_future) -> np.ndarray:  # noqa: D102
        raise NotImplementedError("TODO: sigmoid of the exceedance head -> [N, n_h] in [0,1].")

    def variable_importances(self) -> dict[str, np.ndarray]:  # convenience, not in ABC
        """Return the TFT variable-selection weights (answers 'important drivers')."""
        raise NotImplementedError(
            "TODO: expose interpret_output() encoder/decoder variable weights."
        )

    def export_onnx(self, path: str | Path) -> Path:
        """Export the trained model to ONNX for the serving path (R4 §3.1)."""
        raise NotImplementedError(
            "TODO: model.eval(); torch.onnx.export with dynamic_axes for batch; opset pinned; "
            "write to models/<name>.onnx (R4 §3.1)."
        )

    def save(self, path: str | Path) -> None:  # noqa: D102
        raise NotImplementedError("TODO: save the Lightning checkpoint + hyperparameters.")

    @classmethod
    def load(cls, path: str | Path) -> DualHeadTFT:  # noqa: D102
        raise NotImplementedError("TODO: load checkpoint into a new DualHeadTFT.")


__all__ = ["DualHeadTFT"]
