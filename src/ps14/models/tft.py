"""Primary model: dual-purpose Temporal Fusion Transformer (TFT).

Wraps ``pytorch-forecasting``'s :class:`TemporalFusionTransformer` with a native
``QuantileLoss`` over P10/P50/P90 for direct multi-horizon log-flux forecasting
(ARCHITECTURE.md (e.1), R3 §3c). All heavy imports (``torch``, ``lightning``,
``pytorch_forecasting``) are **lazy** — performed inside the methods that need them — so
``import ps14`` works in the core environment without the ``[dl]`` extra. If the extra is
absent the class is still importable and :meth:`fit`/:meth:`load` raise an informative error.

Window-tensor -> long-dataframe adapter
---------------------------------------
``pytorch-forecasting`` consumes a *long* dataframe, not the ``[N, L, F]`` window tensors of
CONTRACTS.md §4. We therefore unroll each window into a per-timestep long frame:

* a ``series`` id per window (the window index ``i``);
* a ``time_idx`` running ``0 .. L+H-1`` covering encoder (``L`` past steps) then decoder
  (``H`` future steps);
* the ``F`` encoder feature columns over the encoder span, held flat over the decoder span
  -> ``time_varying_unknown_reals``;
* the ``F_kf`` known-future cyclic columns over the whole span (encoder values reconstructed
  as a flat hold of the first decoder value) -> ``time_varying_known_reals``;
* the target ``log_flux_e2``: the true encoder series over the past and the per-step future
  target over the decoder, built so the named horizons land on the right ``time_idx``.

``max_encoder_length = L`` and ``max_prediction_length = H``. After predicting the full
``H``-step decoder we slice the named-horizon offsets (``HORIZON_STEPS``) back out, exactly
mirroring the MIMO read-off in ARCHITECTURE.md (e.1).

Exceedance probability
----------------------
The base TFT here has a single **quantile-regression head**. The per-horizon exceedance
probability ``P(log_flux >= LOG_HARSH)`` is derived from the predicted quantiles by fitting a
Gaussian to (P10, P50, P90) per sample — ``sigma ~= (P90 - P10) / (z90 - z10)`` — and reading
its upper tail at ``LOG_HARSH``. This keeps the head count minimal while yielding a monotone,
spread-consistent probability. The architecture's optional second focal-BCE classification
head (ARCHITECTURE.md (e.1)) can be layered on later as a custom multitask metric; this
wrapper documents and implements the quantile-derived route so the
:class:`~ps14.models.base.Forecaster` contract is fully met without torch in the core env.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ps14.constants import HORIZON_NAMES, HORIZON_STEPS, LOG_HARSH, QUANTILES
from ps14.datasets import schema
from ps14.models.base import Forecaster

_DL_HINT = (
    "TFTForecaster requires the optional deep-learning stack "
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


class TFTForecaster(Forecaster):
    """Temporal Fusion Transformer forecaster (lazy DL backend).

    Parameters
    ----------
    params:
        Hyperparameters merged over the defaults (mirrors ``config/model/tft.yaml`` →
        ``arch`` + ``train``): ``hidden_size``, ``lstm_layers``, ``attention_head_size``,
        ``hidden_continuous_size``, ``dropout``, ``max_epochs``, ``batch_size``,
        ``learning_rate``, ``gradient_clip_val``, ``early_stopping_patience``,
        ``accelerator``.
    quantiles:
        Output quantile levels (default ``constants.QUANTILES`` = (0.1, 0.5, 0.9)).
    lookback:
        Encoder length ``L`` (inferred from the data at ``fit`` when ``None``).
    decoder_steps:
        Decoder length ``H`` (defaults to the 12 h horizon = 144 steps).
    seed:
        Deterministic seed.
    """

    name = "tft"
    horizon_names = HORIZON_NAMES

    DEFAULT_PARAMS: dict[str, Any] = {
        "hidden_size": 32,
        "lstm_layers": 1,
        "attention_head_size": 2,
        "hidden_continuous_size": 16,
        "dropout": 0.1,
        "max_epochs": 30,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "gradient_clip_val": 0.1,
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
        # Accept a nested {"arch": {...}, "train": {...}} config too (config/model/tft.yaml).
        if params:
            flat = dict(params)
            for sub in ("arch", "train"):
                if isinstance(params.get(sub), dict):
                    flat.update(params[sub])
            merged.update({k: v for k, v in flat.items() if k in self.DEFAULT_PARAMS})
        self.params = merged
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

    # ---- adapter ---------------------------------------------------------------------
    def _to_long_frame(self, X: np.ndarray, X_future: np.ndarray, y: np.ndarray | None) -> Any:
        """Unroll ``[N, L, F]`` window tensors into a pytorch-forecasting long dataframe."""
        import pandas as pd

        X = np.asarray(X, dtype="float32")
        X_future = np.asarray(X_future, dtype="float32")
        n, lookback, _ = X.shape
        h = self.decoder_steps
        total = lookback + h
        target_idx = self._feature_cols.index(self._target)

        future_target = np.repeat(X[:, -1:, target_idx], h, axis=1).astype("float32")
        if y is not None:
            y = np.asarray(y, dtype="float32")
            for j, h_name in enumerate(self.horizon_names):
                future_target[:, HORIZON_STEPS[h_name] - 1] = y[:, j]

        frame: dict[str, np.ndarray] = {
            "series": np.repeat(np.arange(n), total).astype("int64"),
            "time_idx": np.tile(np.arange(total), n).astype("int64"),
        }
        enc_target = X[:, :, target_idx]
        frame[self._target] = (
            np.concatenate([enc_target, future_target], axis=1).reshape(-1).astype("float32")
        )
        for c_idx, col in enumerate(self._feature_cols):
            enc = X[:, :, c_idx]
            dec = np.repeat(enc[:, -1:], h, axis=1)
            frame[col] = np.concatenate([enc, dec], axis=1).reshape(-1).astype("float32")
        for c_idx, col in enumerate(self._known_future_cols):
            dec = X_future[:, :h, c_idx]
            enc = np.repeat(dec[:, :1], lookback, axis=1)
            frame[col] = np.concatenate([enc, dec], axis=1).reshape(-1).astype("float32")

        df = pd.DataFrame(frame)
        df["series"] = df["series"].astype(str).astype("category")
        return df

    # ---- contract --------------------------------------------------------------------
    def fit(self, X, X_future, y, y_exceed, *, val=None) -> TFTForecaster:  # noqa: D102
        _require_dl()
        import torch
        from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer
        from pytorch_forecasting.metrics import QuantileLoss

        pl = _import_lightning()
        pl.seed_everything(self.seed, workers=True)
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:  # pragma: no cover - older torch
            pass

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
            time_varying_unknown_reals=[self._target, *self._feature_cols],
            time_varying_known_reals=list(self._known_future_cols),
            target_normalizer=GroupNormalizer(groups=["series"]),
            allow_missing_timesteps=True,
            add_relative_time_idx=True,
            add_target_scales=True,
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

        self._model = TemporalFusionTransformer.from_dataset(
            dataset,
            hidden_size=int(self.params["hidden_size"]),
            lstm_layers=int(self.params["lstm_layers"]),
            attention_head_size=int(self.params["attention_head_size"]),
            hidden_continuous_size=int(self.params["hidden_continuous_size"]),
            dropout=float(self.params["dropout"]),
            output_size=len(self.quantiles),
            loss=QuantileLoss(quantiles=list(self.quantiles)),
            learning_rate=float(self.params["learning_rate"]),
        )
        self._trainer = pl.Trainer(
            max_epochs=int(self.params["max_epochs"]),
            accelerator=str(self.params["accelerator"]),
            devices=1,
            gradient_clip_val=float(self.params["gradient_clip_val"]),
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
            raise RuntimeError("TFTForecaster is not fitted; call fit() first.")

    def _raw_quantiles(self, X: np.ndarray, X_future: np.ndarray) -> np.ndarray:
        """Return the full decoder quantile predictions ``[N, H, n_quantiles]``."""
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
        raw = self._raw_quantiles(X, X_future)  # [N, H, Q]
        taus = list(self.quantiles)
        order = np.argsort(taus)
        stacked = np.stack(
            [
                np.column_stack([raw[:, HORIZON_STEPS[h] - 1, qi] for h in self.horizon_names])
                for qi in range(len(taus))
            ],
            axis=-1,
        )  # [N, n_h, Q]
        # Sort along the quantile axis (in ascending-tau order) to prevent crossing.
        stacked = stacked[..., order]
        stacked = np.sort(stacked, axis=-1)
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

    def variable_importances(self) -> dict[str, np.ndarray]:
        """Return the TFT variable-selection weights (answers 'important drivers').

        pytorch-forecasting exposes these via ``model.interpret_output(raw_predictions)`` on
        a *raw* prediction batch (``mode="raw"``). Use the fitted ``self._model`` directly:
        ``raw = self._model.predict(loader, mode="raw")`` then
        ``self._model.interpret_output(raw, reduction="sum")``.
        """
        self._require_fitted()
        raise NotImplementedError(
            "variable_importances requires a raw prediction batch; run "
            "self._model.predict(loader, mode='raw') then "
            "self._model.interpret_output(raw, reduction='sum') (pytorch-forecasting)."
        )

    # ---- persistence -----------------------------------------------------------------
    def save(self, path: str | Path) -> None:  # noqa: D102
        self._require_fitted()
        import joblib

        path = Path(path)
        ckpt = path.with_suffix(".ckpt")
        self._trainer.save_checkpoint(str(ckpt))
        joblib.dump(
            {
                "params": self.params,
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
    def load(cls, path: str | Path) -> TFTForecaster:  # noqa: D102
        _require_dl()
        import joblib
        from pytorch_forecasting import TemporalFusionTransformer

        state = joblib.load(path)
        obj = cls(
            params=state["params"],
            quantiles=tuple(state["quantiles"]),
            lookback=state["lookback"],
            decoder_steps=state["decoder_steps"],
            seed=state["seed"],
        )
        obj._training_dataset = state["training_dataset"]
        obj._model = TemporalFusionTransformer.load_from_checkpoint(state["checkpoint"])
        return obj


# Backwards-compatible alias: the original scaffold named the primary class ``DualHeadTFT``.
DualHeadTFT = TFTForecaster

__all__ = ["TFTForecaster", "DualHeadTFT"]
