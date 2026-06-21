"""The :class:`Forecaster` abstract base class (the model contract, CONTRACTS.md §5).

Every model — baseline, TFT, N-HiTS, foundation — implements this interface so the
training, evaluation, and serving code is model-agnostic. Array shapes follow
CONTRACTS.md §4:

    X:        [N, L, F]      encoder features
    X_future: [N, H, F_kf]   known-future covariates
    y:        [N, n_h]       log10 flux at named horizons
    y_exceed: [N, n_h]       1[flux >= HARSH_PFU]

``predict`` returns the median (P50) in log10 space; convert to linear pfu with ``10**y``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ps14.constants import HORIZON_NAMES


class Forecaster(ABC):
    """Abstract multi-horizon, dual-output forecaster.

    Attributes
    ----------
    name:
        Short model identifier (used in report filenames / payloads).
    horizon_names:
        Ordered named horizons; defaults to ``constants.HORIZON_NAMES``.
    """

    name: str = "forecaster"
    horizon_names: list[str] = HORIZON_NAMES

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        X_future: np.ndarray,
        y: np.ndarray,
        y_exceed: np.ndarray,
        *,
        val: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> Forecaster:
        """Fit the model on the training tensors.

        Parameters
        ----------
        X, X_future, y, y_exceed:
            Training arrays (shapes per CONTRACTS.md §4).
        val:
            Optional ``(X, X_future, y, y_exceed)`` validation tuple for early stopping.

        Returns
        -------
        Forecaster
            ``self`` (fitted), to allow chaining.
        """

    @abstractmethod
    def predict(self, X: np.ndarray, X_future: np.ndarray) -> np.ndarray:
        """Return the median (P50) log10-flux forecast, shape ``[N, n_h]``."""

    @abstractmethod
    def predict_quantiles(self, X: np.ndarray, X_future: np.ndarray) -> dict[float, np.ndarray]:
        """Return quantile forecasts ``{tau: [N, n_h]}`` of log10 flux.

        Models without a probabilistic head return a degenerate dict where every quantile
        equals :meth:`predict` (so the evaluation harness stays uniform).
        """

    @abstractmethod
    def predict_proba_exceed(self, X: np.ndarray, X_future: np.ndarray) -> np.ndarray:
        """Return P(flux >= HARSH_PFU) per horizon, shape ``[N, n_h]`` in ``[0, 1]``.

        Models without an exceedance head return a 0/1 step derived from the point
        forecast crossing the threshold.
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the fitted model (format is model-specific; CONTRACTS.md §8)."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> Forecaster:
        """Load a persisted model of this class."""


__all__ = ["Forecaster"]
