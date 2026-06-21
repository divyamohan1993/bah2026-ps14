"""Baseline tier: persistence, climatology, LightGBM, REFM-style linear filter.

These are the mandatory skill references every model must beat (R3 §1, ARCHITECTURE.md
(e.2)). :class:`Persistence` is FULLY IMPLEMENTED (trivial, gives an immediately usable
reference + a concrete example of the Forecaster contract); the others are contract stubs.

A subtlety of the windowed contract: ``X`` is ``[N, L, F]`` and the autoregressive target
``log_flux_e2`` is one of the ``F`` channels. Persistence therefore reads the most recent
encoder step of that channel (``X[:, -1, target_channel]``) and repeats it across horizons.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ps14.constants import HORIZON_NAMES, LOG_HARSH
from ps14.datasets import schema
from ps14.models.base import Forecaster

# Channel index of the autoregressive target within FEATURE_COLUMNS (the encoder X).
_TARGET_CHANNEL = schema.FEATURE_COLUMNS.index(schema.TARGET)


class Persistence(Forecaster):
    """Persistence: ``yhat(t+h) = y(t)`` for every horizon (R3 §1).

    Deceptively strong at the nowcast because the target is highly autocorrelated; it is
    the reference skill must beat at short horizons. No fitting required.
    """

    name = "persistence"
    horizon_names = HORIZON_NAMES

    def __init__(self, target_channel: int = _TARGET_CHANNEL) -> None:
        self.target_channel = target_channel

    def fit(self, X, X_future, y, y_exceed, *, val=None) -> Persistence:  # noqa: D102
        return self

    def predict(self, X: np.ndarray, X_future: np.ndarray) -> np.ndarray:  # noqa: D102
        last = np.asarray(X)[:, -1, self.target_channel]  # [N] last observed log-flux
        n_h = len(self.horizon_names)
        return np.repeat(last[:, None], n_h, axis=1).astype("float32")

    def predict_quantiles(self, X, X_future) -> dict[float, np.ndarray]:  # noqa: D102
        point = self.predict(X, X_future)
        return {0.1: point, 0.5: point, 0.9: point}

    def predict_proba_exceed(self, X, X_future) -> np.ndarray:  # noqa: D102
        return (self.predict(X, X_future) >= LOG_HARSH).astype("float32")

    def save(self, path: str | Path) -> None:  # noqa: D102
        np.savez(path, target_channel=self.target_channel, name=self.name)

    @classmethod
    def load(cls, path: str | Path) -> Persistence:  # noqa: D102
        data = np.load(path, allow_pickle=True)
        return cls(target_channel=int(data["target_channel"]))


class Climatology(Forecaster):
    """Diurnal climatology: mean log-flux per (local-time, Kp, day-of-year) bin (R3 §1).

    Captures the strong diurnal + seasonal structure; the long-horizon reference and the
    serving fallback LUT. Fit on TRAIN only. Requires the cyclic/Kp context, which it
    recovers from the known-future covariates and the encoder.
    """

    name = "climatology"
    horizon_names = HORIZON_NAMES

    def __init__(self, n_tod_bins: int = 24, n_kp_bins: int = 10, n_doy_bins: int = 24) -> None:
        self.n_tod_bins = n_tod_bins
        self.n_kp_bins = n_kp_bins
        self.n_doy_bins = n_doy_bins
        self._lut: np.ndarray | None = None

    def fit(self, X, X_future, y, y_exceed, *, val=None) -> Climatology:  # noqa: D102
        raise NotImplementedError(
            "TODO: bin training samples by (tod, kp, doy) and store mean/quantiles of y per "
            "bin in self._lut; persist as the climatology LUT (CONTRACTS.md §7)."
        )

    def predict(self, X, X_future) -> np.ndarray:  # noqa: D102
        raise NotImplementedError(
            "TODO: look up the per-bin mean log-flux for each sample's horizon."
        )

    def predict_quantiles(self, X, X_future) -> dict[float, np.ndarray]:  # noqa: D102
        raise NotImplementedError("TODO: return per-bin P10/P50/P90 from the LUT.")

    def predict_proba_exceed(self, X, X_future) -> np.ndarray:  # noqa: D102
        raise NotImplementedError("TODO: per-bin empirical exceedance frequency, or step from P50.")

    def save(self, path: str | Path) -> None:  # noqa: D102
        raise NotImplementedError("TODO: np.savez(path, lut=self._lut, ...).")

    @classmethod
    def load(cls, path: str | Path) -> Climatology:  # noqa: D102
        raise NotImplementedError("TODO: load LUT into a new Climatology.")


class LightGBMForecaster(Forecaster):
    """LightGBM direct multi-horizon baseline + feature-importance oracle (R3 §1).

    The strongest non-DL baseline. One model per horizon (direct); flattens the encoder
    window into a tabular row of lag/rolling/coupling features. Quantile heads are fit per
    tau separately. Requires the ``[dl]`` extra (lightgbm).
    """

    name = "lightgbm"
    horizon_names = HORIZON_NAMES

    def __init__(self, params: dict | None = None) -> None:
        self.params = params or {}
        self._models: dict[str, object] = {}  # horizon -> fitted booster (P50) + per-tau

    def fit(self, X, X_future, y, y_exceed, *, val=None) -> LightGBMForecaster:  # noqa: D102
        raise NotImplementedError(
            "TODO: flatten X (+ last X_future) to tabular; train one LGBMRegressor per horizon "
            "(and per quantile via objective='quantile'); store in self._models (R3 §1)."
        )

    def predict(self, X, X_future) -> np.ndarray:  # noqa: D102
        raise NotImplementedError("TODO: stack per-horizon P50 predictions -> [N, n_h].")

    def predict_quantiles(self, X, X_future) -> dict[float, np.ndarray]:  # noqa: D102
        raise NotImplementedError("TODO: per-tau, per-horizon predictions -> {tau: [N, n_h]}.")

    def predict_proba_exceed(self, X, X_future) -> np.ndarray:  # noqa: D102
        raise NotImplementedError(
            "TODO: optional LGBMClassifier on the exceedance label, or step from P50."
        )

    def feature_importance(self) -> dict[str, np.ndarray]:  # convenience, not in ABC
        """Return per-horizon feature importances (answers 'important drivers')."""
        raise NotImplementedError("TODO: expose booster.feature_importance() per horizon.")

    def save(self, path: str | Path) -> None:  # noqa: D102
        raise NotImplementedError("TODO: joblib.dump(self._models, path).")

    @classmethod
    def load(cls, path: str | Path) -> LightGBMForecaster:  # noqa: D102
        raise NotImplementedError("TODO: joblib.load -> new LightGBMForecaster.")


class RefmLinearFilter(Forecaster):
    """NOAA-REFM-style linear prediction filter on Vsw history (R3 §1, R4 §3.4).

    The literal operational benchmark: a ridge linear filter mapping a window of L1
    solar-wind speed to the daily-ahead flux/fluence. PE collapses past ~1 day — that is
    the benchmark and the opportunity.
    """

    name = "refm"
    horizon_names = HORIZON_NAMES

    def __init__(self, vsw_history_steps: int = 8640, ridge_alpha: float = 1.0) -> None:
        self.vsw_history_steps = vsw_history_steps
        self.ridge_alpha = ridge_alpha
        self._models: dict[str, object] = {}

    def fit(self, X, X_future, y, y_exceed, *, val=None) -> RefmLinearFilter:  # noqa: D102
        raise NotImplementedError(
            "TODO: extract the Vsw channel history from X; fit a Ridge per horizon mapping "
            "Vsw-window -> y; store in self._models (R4 §3.4)."
        )

    def predict(self, X, X_future) -> np.ndarray:  # noqa: D102
        raise NotImplementedError("TODO: apply per-horizon Ridge to the Vsw window -> [N, n_h].")

    def predict_quantiles(self, X, X_future) -> dict[float, np.ndarray]:  # noqa: D102
        raise NotImplementedError("TODO: residual-based intervals or degenerate quantiles.")

    def predict_proba_exceed(self, X, X_future) -> np.ndarray:  # noqa: D102
        raise NotImplementedError("TODO: step from the point forecast crossing LOG_HARSH.")

    def save(self, path: str | Path) -> None:  # noqa: D102
        raise NotImplementedError("TODO: joblib.dump(self._models, path).")

    @classmethod
    def load(cls, path: str | Path) -> RefmLinearFilter:  # noqa: D102
        raise NotImplementedError("TODO: joblib.load -> new RefmLinearFilter.")


__all__ = [
    "Persistence",
    "Climatology",
    "LightGBMForecaster",
    "RefmLinearFilter",
]
