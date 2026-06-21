"""Backup model A: N-HiTS direct multi-horizon forecaster (R3 §11).

Fast, simple, strong direct multi-horizon; ~50x faster than transformers and a robust
fallback if the TFT overfits the limited data (ARCHITECTURE.md (e.3)). Built on
``pytorch-forecasting``'s :class:`NHiTS` with a native ``QuantileLoss`` for P10/P50/P90.
All heavy imports are **lazy** so ``import ps14`` works without the ``[dl]`` extra; the
class stays importable and :meth:`fit`/:meth:`load` raise an informative error if the extra
is absent.

Like :class:`~ps14.models.tft.TFTForecaster`, the ``[N, L, F]`` window tensors are unrolled
into a pytorch-forecasting long dataframe (encoder past + ``H``-step decoder), the full
decoder is predicted, and the named-horizon offsets (``HORIZON_STEPS``) are sliced back out.
N-HiTS consumes only the target series (it does not take per-channel known/observed reals the
way the TFT does), so the encoder feature window enters through the autoregressive target.
The exceedance probability is derived from the predicted quantile spread (Gaussian tail above
``LOG_HARSH``), mirroring the documented TFT route.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ps14.constants import HORIZON_NAMES, HORIZON_STEPS, LOG_HARSH, QUANTILES
from ps14.datasets import schema
from ps14.models.base import Forecaster

_DL_HINT = (
    "NHiTSForecaster requires the optional deep-learning stack "
    "(torch, pytorch-lightning, pytorch-forecasting). Install the project '[dl]' extra."
)


def _require_dl() -> None:
    """Import-guard: raise an informative error if the DL stack is unavailable."""
    try:
        import pytorch_forecasting  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only when deps missing
        raise ImportError(_DL_HINT) from exc


def _import_lightning() -> Any:
    """Import the Lightning ``pytorch`` module across the lightning/pytorch_lightning split."""
    try:
        import lightning.pytorch as pl  # type: ignore
    except ImportError:  # pragma: no cover
        import pytorch_lightning as pl  # type: ignore
    return pl


def _quantile_sigma(q10: np.ndarray, q90: np.ndarray) -> np.ndarray:
    """Estimate a Gaussian sigma from the P10/P90 spread (z90 - z10 ~= 2.5631)."""
    from scipy.stats import norm

    z_spread = float(norm.ppf(0.9) - norm.ppf(0.1))
    sigma = (np.asarray(q90, dtype="float64") - np.asarray(q10, dtype="float64")) / z_spread
    return np.maximum(sigma, 1e-6)


class NHiTSForecaster(Forecaster):
    """N-HiTS forecaster implementing the :class:`Forecaster` contract (lazy DL backend).

    Parameters
    ----------
    params:
        Hyperparameters merged over the defaults (mirrors ``config/model/nhits.yaml`` →
        ``arch`` + ``train``). Recognized keys include ``hidden_size``, ``dropout``,
        ``max_epochs``, ``batch_size``, ``learning_rate``, ``early_stopping_patience``,
        ``accelerator``.
    quantiles:
        Output quantile levels (default ``constants.QUANTILES``).
    lookback:
        Encoder length ``L`` (inferred at ``fit`` when ``None``).
    decoder_steps:
        Decoder length ``H`` (defaults to the 12 h horizon = 144 steps).
    seed:
        Deterministic seed.
    """

    name = "nhits"
    horizon_names = HORIZON_NAMES

    DEFAULT_PARAMS: dict[str, Any] = {
        "hidden_size": 64,
        "dropout": 0.1,
        "max_epochs": 30,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "early_stopping_patience": 8,
        "accelerator": "auto",
    }

    def __init__(
        self,
        params: dict | None = None,
        quantiles: tuple[float, ...] = QUANTILES,
        *,
        lookback: int | None = None,
        decoder_steps: int = HORIZON_STEPS["12h"],
        seed: int = 1993,
    ) -> None:
        merged = dict(self.DEFAULT_PARAMS)
        if params:
            flat = dict(params)
            for sub in ("arch", "train"):
                if isinstance(params.get(sub), dict):
                    flat.update(params[sub])
            merged.update({k: v for k, v in flat.items() if k in self.DEFAULT_PARAMS})
        self.params = merged
        self.variant = (params or {}).get("variant", "nhits")
        self.quantiles = tuple(float(q) for q in quantiles)
        self.lookback = lookback
        self.decoder_steps = int(decoder_steps)
        self.seed = int(seed)
        self._model: Any = None
        self._trainer: Any = None
        self._training_dataset: Any = None
        self._feature_cols = list(schema.FEATURE_COLUMNS)
        self._known_future_cols = list(schema.KNOWN_FUTURE_COLUMNS)
        self._target = schema.TARGET

    def _to_long_frame(self, X: np.ndarray, X_future: np.ndarray, y: np.ndarray | None) -> Any:
        """Unroll ``[N, L, F]`` window tensors into a pytorch-forecasting long dataframe."""
        import pandas as pd

        X = np.asarray(X, dtype="float32")
        X_future = np.asarray(X_future, dtype="float32")
        n, lookback, _ = X.shape
        h = self.decoder_steps
        total = lookback + h
        target_idx = self._feature_cols.index(self._target)

        from ps14.models.tft import _decoder_target_trajectory

        last_obs = X[:, -1, target_idx]
        future_target = np.repeat(last_obs[:, None], h, axis=1).astype("float32")
        if y is not None:
            future_target = _decoder_target_trajectory(
                last_obs, np.asarray(y, dtype="float32"), self.horizon_names, h
            )

        frame: dict[str, np.ndarray] = {
            "series": np.repeat(np.arange(n), total).astype("int64"),
            "time_idx": np.tile(np.arange(total), n).astype("int64"),
        }
        enc_target = X[:, :, target_idx]
        frame[self._target] = (
            np.concatenate([enc_target, future_target], axis=1).reshape(-1).astype("float32")
        )
        # N-HiTS supports known-future reals (covariates) but not unknown reals.
        for c_idx, col in enumerate(self._known_future_cols):
            dec = X_future[:, :h, c_idx]
            enc = np.repeat(dec[:, :1], lookback, axis=1)
            frame[col] = np.concatenate([enc, dec], axis=1).reshape(-1).astype("float32")

        df = pd.DataFrame(frame)
        df["series"] = df["series"].astype(str).astype("category")
        return df

    def fit(self, X, X_future, y, y_exceed, *, val=None) -> NHiTSForecaster:  # noqa: D102
        _require_dl()
        from pytorch_forecasting import NHiTS, TimeSeriesDataSet
        from pytorch_forecasting.data import EncoderNormalizer
        from pytorch_forecasting.metrics import QuantileLoss

        pl = _import_lightning()
        pl.seed_everything(self.seed, workers=True)

        X = np.asarray(X, dtype="float32")
        self.lookback = X.shape[1] if self.lookback is None else self.lookback
        df = self._to_long_frame(X, X_future, y)
        max_idx = int(df["time_idx"].max())

        dataset = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target=self._target,
            group_ids=["series"],
            max_encoder_length=self.lookback,
            max_prediction_length=self.decoder_steps,
            time_varying_unknown_reals=[self._target],
            time_varying_known_reals=list(self._known_future_cols),
            # Encoder-only normalization: identical scale at train and predict (no decoder leak).
            target_normalizer=EncoderNormalizer(),
            allow_missing_timesteps=True,
            add_relative_time_idx=False,
        )
        self._training_dataset = dataset
        train_loader = dataset.to_dataloader(
            train=True, batch_size=int(self.params["batch_size"]), num_workers=0
        )

        val_loader = None
        callbacks: list[Any] = []
        if val is not None:
            from lightning.pytorch.callbacks import EarlyStopping  # type: ignore

            vX, vXf, vy, _ = val
            vdf = self._to_long_frame(np.asarray(vX, dtype="float32"), vXf, vy)
            vdf["time_idx"] = vdf["time_idx"].clip(upper=max_idx)
            val_ds = TimeSeriesDataSet.from_dataset(
                dataset, vdf, predict=False, stop_randomization=True
            )
            val_loader = val_ds.to_dataloader(
                train=False, batch_size=int(self.params["batch_size"]), num_workers=0
            )
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=int(self.params["early_stopping_patience"]),
                    mode="min",
                )
            )

        self._model = NHiTS.from_dataset(
            dataset,
            hidden_size=int(self.params["hidden_size"]),
            dropout=float(self.params["dropout"]),
            loss=QuantileLoss(quantiles=list(self.quantiles)),
            learning_rate=float(self.params["learning_rate"]),
        )
        self._trainer = pl.Trainer(
            max_epochs=int(self.params["max_epochs"]),
            accelerator=str(self.params["accelerator"]),
            devices=1,
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            callbacks=callbacks,
            deterministic=True,
        )
        self._trainer.fit(self._model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        return self

    def _require_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("NHiTSForecaster is not fitted; call fit() first.")

    def _raw_quantiles(self, X: np.ndarray, X_future: np.ndarray) -> np.ndarray:
        from pytorch_forecasting import TimeSeriesDataSet

        df = self._to_long_frame(np.asarray(X, dtype="float32"), X_future, None)
        ds = TimeSeriesDataSet.from_dataset(
            self._training_dataset, df, predict=True, stop_randomization=True
        )
        loader = ds.to_dataloader(
            train=False, batch_size=int(self.params["batch_size"]), num_workers=0
        )
        raw = self._model.predict(loader, mode="quantiles")
        return raw.detach().cpu().numpy() if hasattr(raw, "detach") else np.asarray(raw)

    def predict(self, X, X_future) -> np.ndarray:  # noqa: D102
        self._require_fitted()
        q = self.predict_quantiles(X, X_future)
        return q[0.5] if 0.5 in q else q[self.quantiles[len(self.quantiles) // 2]]

    def predict_quantiles(self, X, X_future) -> dict[float, np.ndarray]:  # noqa: D102
        self._require_fitted()
        raw = self._raw_quantiles(X, X_future)
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
        self._require_fitted()
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
        self._require_fitted()
        import joblib

        path = Path(path)
        ckpt = path.with_suffix(".ckpt")
        self._trainer.save_checkpoint(str(ckpt))
        joblib.dump(
            {
                "params": self.params,
                "variant": self.variant,
                "quantiles": list(self.quantiles),
                "lookback": self.lookback,
                "decoder_steps": self.decoder_steps,
                "seed": self.seed,
                "checkpoint": str(ckpt),
                "training_dataset": self._training_dataset,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> NHiTSForecaster:  # noqa: D102
        _require_dl()
        import joblib
        from pytorch_forecasting import NHiTS

        state = joblib.load(path)
        obj = cls(
            params={"variant": state.get("variant", "nhits"), **state["params"]},
            quantiles=tuple(state["quantiles"]),
            lookback=state["lookback"],
            decoder_steps=state["decoder_steps"],
            seed=state["seed"],
        )
        obj._training_dataset = state["training_dataset"]
        obj._model = NHiTS.load_from_checkpoint(state["checkpoint"])
        return obj


__all__ = ["NHiTSForecaster"]
