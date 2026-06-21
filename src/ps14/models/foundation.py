"""Backup model B: zero/few-shot time-series foundation model (R3 §4/§11).

Wraps **Chronos-Bolt** (Amazon, T5, Apache-2.0) via the ``chronos-forecasting`` package for
the P50 / quantile path (ARCHITECTURE.md (e.3)). Chronos-Bolt is a patch-based forecaster
that emits multi-step quantiles directly, needs no training to run zero-shot, and fine-tunes
cheaply. All heavy imports are **lazy** so ``import ps14`` works without the dependency; the
class stays importable and the methods raise an informative error if it is absent.

Adapter
-------
Chronos is univariate: it forecasts the autoregressive target from its own context. For each
window we pass the encoder history of ``log_flux_e2`` (``X[:, :, target_channel]``) as the
context and request a ``decoder_steps``-step forecast, then slice the named-horizon offsets
(``HORIZON_STEPS``) back out — mirroring the MIMO read-off used by the TFT/N-HiTS wrappers.
``fit`` is a no-op for the zero-shot path (the pretrained weights are the model); a documented
hook is left for few-shot fine-tuning. The exceedance probability is derived from the
predicted quantile spread (Gaussian tail above ``LOG_HARSH``), consistent with the other
quantile models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ps14.constants import HORIZON_NAMES, HORIZON_STEPS, LOG_HARSH, QUANTILES
from ps14.datasets import schema
from ps14.models.base import Forecaster

_DEP_HINT = (
    "FoundationForecaster requires the optional 'chronos-forecasting' package "
    "(and torch). Install it (pip install chronos-forecasting) to use the foundation route."
)

_TARGET_CHANNEL = schema.FEATURE_COLUMNS.index(schema.TARGET)


def _require_chronos() -> Any:
    """Import-guard returning the Chronos pipeline class, else an informative error."""
    try:
        from chronos import BaseChronosPipeline  # type: ignore

        return BaseChronosPipeline
    except ImportError:  # pragma: no cover - exercised only when dep missing
        try:
            from chronos import ChronosPipeline  # type: ignore

            return ChronosPipeline
        except ImportError as exc:
            raise ImportError(_DEP_HINT) from exc


def _quantile_sigma(q10: np.ndarray, q90: np.ndarray) -> np.ndarray:
    """Estimate a Gaussian sigma from the P10/P90 spread (z90 - z10 ~= 2.5631)."""
    from scipy.stats import norm

    z_spread = float(norm.ppf(0.9) - norm.ppf(0.1))
    sigma = (np.asarray(q90, dtype="float64") - np.asarray(q10, dtype="float64")) / z_spread
    return np.maximum(sigma, 1e-6)


class FoundationForecaster(Forecaster):
    """Chronos-Bolt foundation-model forecaster (lazy backend).

    Parameters
    ----------
    params:
        ``model_name`` (HF id, default ``"amazon/chronos-bolt-base"``), ``device_map``
        (``"cpu"`` | ``"cuda"`` | ``"auto"``), ``context_length`` (max context steps fed to
        the model; the encoder tail is used), and an optional ``num_samples`` for sampling
        backbones.
    quantiles:
        Output quantile levels (default ``constants.QUANTILES``).
    decoder_steps:
        Forecast horizon ``H`` (defaults to the 12 h horizon = 144 steps).
    """

    name = "foundation"
    horizon_names = HORIZON_NAMES

    DEFAULT_PARAMS: dict[str, Any] = {
        "model_name": "amazon/chronos-bolt-base",
        "device_map": "cpu",
        "context_length": 512,
        "num_samples": 20,
    }

    def __init__(
        self,
        params: dict | None = None,
        quantiles: tuple[float, ...] = QUANTILES,
        *,
        decoder_steps: int = HORIZON_STEPS["12h"],
        target_channel: int = _TARGET_CHANNEL,
    ) -> None:
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.backbone = self.params["model_name"]
        self.quantiles = tuple(float(q) for q in quantiles)
        self.decoder_steps = int(decoder_steps)
        self.target_channel = int(target_channel)
        self._pipeline: Any = None

    def _ensure_pipeline(self) -> Any:
        """Lazily construct (and cache) the Chronos pipeline."""
        if self._pipeline is None:
            import torch

            pipeline_cls = _require_chronos()
            dtype = torch.bfloat16 if str(self.params["device_map"]) != "cpu" else torch.float32
            self._pipeline = pipeline_cls.from_pretrained(
                self.params["model_name"],
                device_map=self.params["device_map"],
                torch_dtype=dtype,
            )
        return self._pipeline

    def fit(self, X, X_future, y, y_exceed, *, val=None) -> FoundationForecaster:  # noqa: D102
        # Zero-shot path: the pretrained backbone *is* the model — loading it validates the
        # dependency is present. (Few-shot fine-tuning would slot in here on X/y.)
        self._ensure_pipeline()
        return self

    def _context(self, X: np.ndarray) -> Any:
        """Build the list of per-window context tensors from the encoder target history."""
        import torch

        X = np.asarray(X, dtype="float32")
        hist = X[:, :, self.target_channel]
        ctx = int(self.params["context_length"])
        if hist.shape[1] > ctx:
            hist = hist[:, -ctx:]
        return [torch.tensor(row, dtype=torch.float32) for row in hist]

    def _forecast_quantiles(self, X: np.ndarray) -> np.ndarray:
        """Return the full decoder quantiles ``[N, H, n_quantiles]`` from Chronos."""
        pipeline = self._ensure_pipeline()
        context = self._context(X)
        levels = list(self.quantiles)
        # Chronos-Bolt exposes predict_quantiles(context, prediction_length, quantile_levels).
        if hasattr(pipeline, "predict_quantiles"):
            q_tensor, _ = pipeline.predict_quantiles(
                context=context,
                prediction_length=self.decoder_steps,
                quantile_levels=levels,
            )
            arr = q_tensor.detach().cpu().numpy()  # [N, H, n_quantiles]
        else:  # sampling backbone fallback
            samples = pipeline.predict(
                context, self.decoder_steps, num_samples=int(self.params["num_samples"])
            )
            samples = samples.detach().cpu().numpy()  # [N, num_samples, H]
            arr = np.quantile(samples, levels, axis=1).transpose(1, 2, 0)  # [N, H, n_q]
        return arr

    def predict(self, X, X_future) -> np.ndarray:  # noqa: D102
        q = self.predict_quantiles(X, X_future)
        return q[0.5] if 0.5 in q else q[self.quantiles[len(self.quantiles) // 2]]

    def predict_quantiles(self, X, X_future) -> dict[float, np.ndarray]:  # noqa: D102
        raw = self._forecast_quantiles(np.asarray(X, dtype="float32"))
        taus = list(self.quantiles)
        order = np.argsort(taus)
        stacked = np.stack(
            [
                np.column_stack([raw[:, HORIZON_STEPS[h] - 1, qi] for h in self.horizon_names])
                for qi in range(len(taus))
            ],
            axis=-1,
        )  # [N, n_h, Q]
        stacked = np.sort(stacked[..., order], axis=-1)
        out: dict[float, np.ndarray] = {}
        for i, oi in enumerate(order):
            out[taus[oi]] = stacked[..., i].astype("float32")
        return out

    def predict_proba_exceed(self, X, X_future) -> np.ndarray:  # noqa: D102
        from scipy.stats import norm

        q = self.predict_quantiles(X, X_future)
        taus = sorted(self.quantiles)
        lo = q.get(0.1, q[taus[0]])
        mid = q.get(0.5, q[taus[len(taus) // 2]])
        hi = q.get(0.9, q[taus[-1]])
        sigma = _quantile_sigma(lo, hi)
        proba = norm.sf(LOG_HARSH, loc=np.asarray(mid, dtype="float64"), scale=sigma)
        return np.clip(proba, 0.0, 1.0).astype("float32")

    def save(self, path: str | Path) -> None:  # noqa: D102
        import joblib

        joblib.dump(
            {
                "params": self.params,
                "quantiles": list(self.quantiles),
                "decoder_steps": self.decoder_steps,
                "target_channel": self.target_channel,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> FoundationForecaster:  # noqa: D102
        import joblib

        state = joblib.load(path)
        return cls(
            params=state["params"],
            quantiles=tuple(state["quantiles"]),
            decoder_steps=state["decoder_steps"],
            target_channel=state["target_channel"],
        )


__all__ = ["FoundationForecaster"]
