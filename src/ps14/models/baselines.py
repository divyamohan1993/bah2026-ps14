"""Baseline tier: persistence, climatology, LightGBM, REFM-style linear filter.

These are the mandatory skill references every model must beat (R3 §1, ARCHITECTURE.md
(e.2)). :class:`Persistence` is the trivial, immediately usable reference and a concrete
worked example of the :class:`~ps14.models.base.Forecaster` contract; the remaining
baselines are implemented here:

* :class:`Climatology` — diurnal (+ seasonal) climatology learned per local-time / Kp /
  day-of-year bin from TRAIN (pure numpy/pandas).
* :class:`LightGBMForecaster` — gradient-boosted trees on the *flattened* window features,
  one model per horizon (lazy ``import lightgbm``); the strongest non-DL baseline and a
  feature-importance oracle.
* :class:`REFMForecaster` — a NOAA-REFM-style linear prediction filter on recent
  solar-wind speed + flux persistence (pure numpy / scikit-learn ``LinearRegression``).

A subtlety of the windowed contract: ``X`` is ``[N, L, F]`` and the autoregressive target
``log_flux_e2`` is one of the ``F`` channels. Persistence therefore reads the most recent
encoder step of that channel (``X[:, -1, target_channel]``) and repeats it across horizons.
Climatology and REFM recover their drivers (cyclic context / Vsw history) from the encoder
and the known-future tensor in the same channel-indexed way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ps14.constants import HORIZON_NAMES, HORIZON_STEPS, LOG_HARSH, QUANTILES
from ps14.datasets import schema
from ps14.models.base import Forecaster

# Channel index of the autoregressive target within FEATURE_COLUMNS (the encoder X).
_TARGET_CHANNEL = schema.FEATURE_COLUMNS.index(schema.TARGET)
# Channel index of the solar-wind speed within FEATURE_COLUMNS (REFM driver).
_VSW_CHANNEL = schema.FEATURE_COLUMNS.index("vsw")

# Known-future channel indices (CONTRACTS.md §3 / schema.KNOWN_FUTURE_COLUMNS order):
# ["tod_sin", "tod_cos", "doy_sin", "doy_cos", "mlt_sin", "mlt_cos"].
_KF = {name: i for i, name in enumerate(schema.KNOWN_FUTURE_COLUMNS)}


def _angle_to_unit_bin(sin_v: np.ndarray, cos_v: np.ndarray, n_bins: int) -> np.ndarray:
    """Map a (sin, cos) cyclic pair back to an integer bin in ``[0, n_bins)``.

    ``atan2(sin, cos)`` recovers the phase in ``(-pi, pi]``; we wrap it to ``[0, 2pi)`` and
    quantize into ``n_bins`` equal arcs (so a full diurnal/seasonal cycle is binned).
    """
    phase = np.arctan2(sin_v, cos_v)  # (-pi, pi]
    frac = (phase / (2.0 * np.pi)) % 1.0  # [0, 1)
    idx = np.floor(frac * n_bins).astype("int64")
    return np.clip(idx, 0, n_bins - 1)


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
    """Diurnal (+ seasonal) climatology: per-(local-time, Kp, day-of-year) bin statistics.

    Captures the strong diurnal + seasonal structure of the >2 MeV flux (R1 §0, R3 §1); the
    long-horizon skill reference and the serving fallback LUT (CONTRACTS.md §7). Pure
    numpy/pandas — no optional deps.

    For each training sample and named horizon we read the **known-future** cyclic
    covariates *at the target time* (``X_future[:, horizon_step - 1, :]``) to derive the
    (time-of-day, day-of-year, magnetic-local-time) bins, and the **Kp** value from the last
    encoder step. We then store, per ``(horizon, tod_bin, kp_bin, doy_bin, mlt_bin)`` cell,
    the mean log-flux, the empirical quantiles, and the exceedance frequency. Prediction is a
    constant-time lookup; empty cells fall back to the per-horizon global statistics.
    """

    name = "climatology"
    horizon_names = HORIZON_NAMES

    def __init__(
        self,
        n_tod_bins: int = 24,
        n_kp_bins: int = 10,
        n_doy_bins: int = 24,
        n_mlt_bins: int = 1,
        quantiles: tuple[float, ...] = QUANTILES,
        *,
        kp_channel: int | None = None,
    ) -> None:
        self.n_tod_bins = int(n_tod_bins)
        self.n_kp_bins = int(n_kp_bins)
        self.n_doy_bins = int(n_doy_bins)
        self.n_mlt_bins = int(n_mlt_bins)
        self.quantiles = tuple(float(q) for q in quantiles)
        # Kp lives in the encoder (observed-past); default to its FEATURE_COLUMNS index.
        self.kp_channel = (
            schema.FEATURE_COLUMNS.index("kp") if kp_channel is None else int(kp_channel)
        )
        # LUTs: horizon -> dict keyed by flat bin index. Filled in fit().
        self._mean: dict[str, dict[int, float]] = {}
        self._quant: dict[str, dict[int, list[float]]] = {}
        self._exceed: dict[str, dict[int, float]] = {}
        self._global_mean: dict[str, float] = {}
        self._global_quant: dict[str, list[float]] = {}
        self._global_exceed: dict[str, float] = {}

    # ---- bin helpers -----------------------------------------------------------------
    def _flat_bin(
        self, tod: np.ndarray, kp: np.ndarray, doy: np.ndarray, mlt: np.ndarray
    ) -> np.ndarray:
        """Combine the four sub-bins into a single flat index."""
        return (
            ((tod * self.n_kp_bins + kp) * self.n_doy_bins + doy) * self.n_mlt_bins + mlt
        ).astype("int64")

    def _bins_for_horizon(self, X: np.ndarray, X_future: np.ndarray, h_name: str) -> np.ndarray:
        """Compute the flat climatology bin per sample for one named horizon."""
        X = np.asarray(X, dtype="float64")
        X_future = np.asarray(X_future, dtype="float64")
        step = HORIZON_STEPS[h_name]
        h_idx = min(step - 1, X_future.shape[1] - 1)  # known-future row at the target time
        kf = X_future[:, h_idx, :]
        tod = _angle_to_unit_bin(kf[:, _KF["tod_sin"]], kf[:, _KF["tod_cos"]], self.n_tod_bins)
        doy = _angle_to_unit_bin(kf[:, _KF["doy_sin"]], kf[:, _KF["doy_cos"]], self.n_doy_bins)
        if self.n_mlt_bins > 1:
            mlt = _angle_to_unit_bin(kf[:, _KF["mlt_sin"]], kf[:, _KF["mlt_cos"]], self.n_mlt_bins)
        else:
            mlt = np.zeros(X.shape[0], dtype="int64")
        kp_val = X[:, -1, self.kp_channel]  # last observed Kp
        kp = np.clip(np.floor(np.nan_to_num(kp_val, nan=0.0)), 0, self.n_kp_bins - 1).astype(
            "int64"
        )
        return self._flat_bin(tod, kp, doy, mlt)

    # ---- contract --------------------------------------------------------------------
    def fit(self, X, X_future, y, y_exceed, *, val=None) -> Climatology:  # noqa: D102
        import pandas as pd

        y = np.asarray(y, dtype="float64")
        y_exceed = np.asarray(y_exceed, dtype="float64")
        for j, h_name in enumerate(self.horizon_names):
            bins = self._bins_for_horizon(X, X_future, h_name)
            yj = y[:, j]
            ej = y_exceed[:, j]
            finite = np.isfinite(yj)
            frame = pd.DataFrame({"bin": bins[finite], "y": yj[finite], "e": ej[finite]})
            grouped = frame.groupby("bin")
            self._mean[h_name] = grouped["y"].mean().to_dict()
            self._exceed[h_name] = grouped["e"].mean().to_dict()
            quant_map: dict[int, list[float]] = {}
            for b, sub in grouped["y"]:
                quant_map[int(b)] = [float(np.quantile(sub.to_numpy(), q)) for q in self.quantiles]
            self._quant[h_name] = quant_map
            # Per-horizon global fallbacks for empty cells.
            self._global_mean[h_name] = float(np.mean(yj[finite])) if finite.any() else 0.0
            self._global_exceed[h_name] = float(np.mean(ej[finite])) if finite.any() else 0.0
            self._global_quant[h_name] = (
                [float(np.quantile(yj[finite], q)) for q in self.quantiles]
                if finite.any()
                else [0.0 for _ in self.quantiles]
            )
        return self

    def _require_fitted(self) -> None:
        if not self._mean:
            raise RuntimeError("Climatology is not fitted; call fit() first.")

    def predict(self, X, X_future) -> np.ndarray:  # noqa: D102
        self._require_fitted()
        n = np.asarray(X).shape[0]
        out = np.empty((n, len(self.horizon_names)), dtype="float32")
        for j, h_name in enumerate(self.horizon_names):
            bins = self._bins_for_horizon(X, X_future, h_name)
            lut = self._mean[h_name]
            default = self._global_mean[h_name]
            out[:, j] = [lut.get(int(b), default) for b in bins]
        return out

    def predict_quantiles(self, X, X_future) -> dict[float, np.ndarray]:  # noqa: D102
        self._require_fitted()
        n = np.asarray(X).shape[0]
        n_q = len(self.quantiles)
        out = {q: np.empty((n, len(self.horizon_names)), dtype="float32") for q in self.quantiles}
        for j, h_name in enumerate(self.horizon_names):
            bins = self._bins_for_horizon(X, X_future, h_name)
            lut = self._quant[h_name]
            default = self._global_quant[h_name]
            rows = np.asarray([lut.get(int(b), default) for b in bins], dtype="float32").reshape(
                n, n_q
            )
            # Enforce monotone non-crossing quantiles per row.
            rows = np.sort(rows, axis=1)
            for qi, q in enumerate(self.quantiles):
                out[q][:, j] = rows[:, qi]
        return out

    def predict_proba_exceed(self, X, X_future) -> np.ndarray:  # noqa: D102
        self._require_fitted()
        n = np.asarray(X).shape[0]
        out = np.empty((n, len(self.horizon_names)), dtype="float32")
        for j, h_name in enumerate(self.horizon_names):
            bins = self._bins_for_horizon(X, X_future, h_name)
            lut = self._exceed[h_name]
            default = self._global_exceed[h_name]
            out[:, j] = [lut.get(int(b), default) for b in bins]
        return np.clip(out, 0.0, 1.0)

    def _state(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_tod_bins": self.n_tod_bins,
            "n_kp_bins": self.n_kp_bins,
            "n_doy_bins": self.n_doy_bins,
            "n_mlt_bins": self.n_mlt_bins,
            "quantiles": list(self.quantiles),
            "kp_channel": self.kp_channel,
            "mean": {h: {str(k): v for k, v in d.items()} for h, d in self._mean.items()},
            "quant": {h: {str(k): v for k, v in d.items()} for h, d in self._quant.items()},
            "exceed": {h: {str(k): v for k, v in d.items()} for h, d in self._exceed.items()},
            "global_mean": self._global_mean,
            "global_quant": self._global_quant,
            "global_exceed": self._global_exceed,
        }

    def save(self, path: str | Path) -> None:  # noqa: D102
        Path(path).write_text(json.dumps(self._state()), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Climatology:  # noqa: D102
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(
            n_tod_bins=state["n_tod_bins"],
            n_kp_bins=state["n_kp_bins"],
            n_doy_bins=state["n_doy_bins"],
            n_mlt_bins=state.get("n_mlt_bins", 1),
            quantiles=tuple(state["quantiles"]),
            kp_channel=state["kp_channel"],
        )
        obj._mean = {h: {int(k): v for k, v in d.items()} for h, d in state["mean"].items()}
        obj._quant = {h: {int(k): v for k, v in d.items()} for h, d in state["quant"].items()}
        obj._exceed = {h: {int(k): v for k, v in d.items()} for h, d in state["exceed"].items()}
        obj._global_mean = state["global_mean"]
        obj._global_quant = state["global_quant"]
        obj._global_exceed = state["global_exceed"]
        return obj


class LightGBMForecaster(Forecaster):
    """LightGBM direct multi-horizon baseline + feature-importance oracle (R3 §1).

    The strongest non-DL baseline. One regressor per ``(horizon, quantile)`` plus one
    binary classifier per horizon for the exceedance head. The encoder window ``X`` is
    flattened to a compact tabular row — the **last** step of every channel plus a few
    lag/rolling summaries (mean/std/min/max/first) of each channel over the window — and the
    last known-future row is appended. Requires the ``[dl]`` extra (``lightgbm``), imported
    lazily so ``import ps14`` works without it.
    """

    name = "lightgbm"
    horizon_names = HORIZON_NAMES

    #: Default LightGBM hyperparameters (overridable via ``params``); mirror baseline.yaml.
    DEFAULT_PARAMS: dict[str, Any] = {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "random_state": 1993,
        "n_jobs": -1,
        "verbose": -1,
    }

    def __init__(
        self, params: dict | None = None, quantiles: tuple[float, ...] = QUANTILES
    ) -> None:
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.quantiles = tuple(float(q) for q in quantiles)
        # horizon -> {"q": {tau: booster}, "clf": booster|None}
        self._models: dict[str, dict[str, Any]] = {}
        self._feature_names: list[str] = []

    # ---- feature flattening ----------------------------------------------------------
    def _flatten(self, X: np.ndarray, X_future: np.ndarray) -> np.ndarray:
        """Flatten ``[N, L, F]`` + last known-future row into a compact ``[N, D]`` table."""
        X = np.asarray(X, dtype="float64")
        X_future = np.asarray(X_future, dtype="float64")
        feats = [
            X[:, -1, :],  # last encoder step of every channel
            X[:, 0, :],  # first encoder step (coarse trend anchor)
            np.nanmean(X, axis=1),
            np.nanstd(X, axis=1),
            np.nanmin(X, axis=1),
            np.nanmax(X, axis=1),
            X_future[:, -1, :],  # known-future covariates at the furthest decoder step
        ]
        flat = np.concatenate(feats, axis=1)
        return np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_feature_names(self) -> list[str]:
        f = schema.FEATURE_COLUMNS
        kf = schema.KNOWN_FUTURE_COLUMNS
        names: list[str] = []
        names += [f"{c}_last" for c in f]
        names += [f"{c}_first" for c in f]
        names += [f"{c}_mean" for c in f]
        names += [f"{c}_std" for c in f]
        names += [f"{c}_min" for c in f]
        names += [f"{c}_max" for c in f]
        names += [f"{c}_kf" for c in kf]
        return names

    def fit(self, X, X_future, y, y_exceed, *, val=None) -> LightGBMForecaster:  # noqa: D102
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as exc:  # pragma: no cover - exercised only when dep missing
            raise ImportError(
                "LightGBMForecaster requires the optional 'lightgbm' dependency "
                "(install the project '[dl]' extra)."
            ) from exc

        Xf = self._flatten(X, X_future)
        y = np.asarray(y, dtype="float64")
        y_exceed = np.asarray(y_exceed, dtype="float64")
        self._feature_names = self._build_feature_names()
        eval_set = None
        if val is not None:
            vX, vXf, vy, vye = val
            Vf = self._flatten(vX, vXf)
            vy = np.asarray(vy, dtype="float64")
            vye = np.asarray(vye, dtype="float64")

        base = {k: v for k, v in self.params.items() if k not in {"objective", "alpha", "metric"}}
        for j, h_name in enumerate(self.horizon_names):
            q_models: dict[float, Any] = {}
            for tau in self.quantiles:
                reg = LGBMRegressor(objective="quantile", alpha=tau, **base)
                if val is not None:
                    eval_set = [(Vf, vy[:, j])]
                    reg.fit(Xf, y[:, j], eval_set=eval_set)
                else:
                    reg.fit(Xf, y[:, j])
                q_models[tau] = reg
            # Exceedance classifier (only if both classes are present in TRAIN).
            clf: Any = None
            labels = y_exceed[:, j]
            if np.unique(labels[np.isfinite(labels)]).size >= 2:
                clf = LGBMClassifier(objective="binary", **base)
                if val is not None:
                    clf.fit(Xf, labels, eval_set=[(Vf, vye[:, j])])
                else:
                    clf.fit(Xf, labels)
            self._models[h_name] = {"q": q_models, "clf": clf}
        return self

    def _require_fitted(self) -> None:
        if not self._models:
            raise RuntimeError("LightGBMForecaster is not fitted; call fit() first.")

    def predict(self, X, X_future) -> np.ndarray:  # noqa: D102
        q = self.predict_quantiles(X, X_future)
        median = 0.5 if 0.5 in q else self.quantiles[len(self.quantiles) // 2]
        return q[median]

    def predict_quantiles(self, X, X_future) -> dict[float, np.ndarray]:  # noqa: D102
        self._require_fitted()
        Xf = self._flatten(X, X_future)
        n = Xf.shape[0]
        out = {
            tau: np.empty((n, len(self.horizon_names)), dtype="float32") for tau in self.quantiles
        }
        for j, h_name in enumerate(self.horizon_names):
            preds = np.column_stack(
                [self._models[h_name]["q"][tau].predict(Xf) for tau in self.quantiles]
            )
            preds = np.sort(preds, axis=1)  # avoid quantile crossing
            for qi, tau in enumerate(self.quantiles):
                out[tau][:, j] = preds[:, qi]
        return out

    def predict_proba_exceed(self, X, X_future) -> np.ndarray:  # noqa: D102
        self._require_fitted()
        Xf = self._flatten(X, X_future)
        n = Xf.shape[0]
        out = np.empty((n, len(self.horizon_names)), dtype="float32")
        median = self.predict(X, X_future)
        for j, h_name in enumerate(self.horizon_names):
            clf = self._models[h_name]["clf"]
            if clf is None:
                out[:, j] = (median[:, j] >= LOG_HARSH).astype("float32")
            else:
                proba = clf.predict_proba(Xf)
                # Probability of the positive (exceedance) class.
                pos = list(clf.classes_).index(1) if 1 in clf.classes_ else proba.shape[1] - 1
                out[:, j] = proba[:, pos]
        return np.clip(out, 0.0, 1.0)

    def feature_importance(self) -> dict[str, np.ndarray]:
        """Return per-horizon P50 feature importances (answers 'important drivers')."""
        self._require_fitted()
        median = 0.5 if 0.5 in self.quantiles else self.quantiles[len(self.quantiles) // 2]
        return {
            h_name: np.asarray(self._models[h_name]["q"][median].feature_importances_)
            for h_name in self.horizon_names
        }

    def save(self, path: str | Path) -> None:  # noqa: D102
        joblib.dump(
            {
                "models": self._models,
                "params": self.params,
                "quantiles": list(self.quantiles),
                "feature_names": self._feature_names,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> LightGBMForecaster:  # noqa: D102
        state = joblib.load(path)
        obj = cls(params=state["params"], quantiles=tuple(state["quantiles"]))
        obj._models = state["models"]
        obj._feature_names = state.get("feature_names", [])
        return obj


class REFMForecaster(Forecaster):
    """NOAA-REFM-style linear prediction filter (R3 §1, R4 §3.4).

    The literal operational benchmark: a linear model mapping recent solar-wind speed (and
    flux persistence) to the next-horizon log-flux. We summarize the ``vsw`` channel of the
    encoder window into a small set of statistics (last value, window mean, recent mean,
    trend) and append the last observed log-flux (persistence anchor), then fit one
    ``LinearRegression`` per horizon (pure numpy / scikit-learn). PE is expected to collapse
    past ~1 day — that is precisely the benchmark and the opportunity.

    Prediction intervals come from the per-horizon training residual spread (Gaussian
    quantile offsets); the exceedance probability is the Gaussian tail above ``LOG_HARSH``.
    """

    name = "refm"
    horizon_names = HORIZON_NAMES

    def __init__(
        self,
        recent_steps: int = 288,
        quantiles: tuple[float, ...] = QUANTILES,
        *,
        vsw_channel: int = _VSW_CHANNEL,
        target_channel: int = _TARGET_CHANNEL,
    ) -> None:
        self.recent_steps = int(recent_steps)
        self.quantiles = tuple(float(q) for q in quantiles)
        self.vsw_channel = int(vsw_channel)
        self.target_channel = int(target_channel)
        self._models: dict[str, Any] = {}
        self._resid_std: dict[str, float] = {}

    def _design(self, X: np.ndarray) -> np.ndarray:
        """Build the REFM linear-filter design matrix from the Vsw + flux history."""
        X = np.asarray(X, dtype="float64")
        vsw = X[:, :, self.vsw_channel]
        k = min(self.recent_steps, vsw.shape[1])
        recent = vsw[:, -k:]
        vsw_last = vsw[:, -1]
        vsw_mean = np.nanmean(vsw, axis=1)
        vsw_recent_mean = np.nanmean(recent, axis=1)
        vsw_recent_max = np.nanmax(recent, axis=1)
        vsw_trend = vsw[:, -1] - vsw[:, 0]
        flux_last = X[:, -1, self.target_channel]  # persistence anchor
        design = np.column_stack(
            [vsw_last, vsw_mean, vsw_recent_mean, vsw_recent_max, vsw_trend, flux_last]
        )
        return np.nan_to_num(design, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X, X_future, y, y_exceed, *, val=None) -> REFMForecaster:  # noqa: D102
        from sklearn.linear_model import LinearRegression

        D = self._design(X)
        y = np.asarray(y, dtype="float64")
        for j, h_name in enumerate(self.horizon_names):
            yj = y[:, j]
            finite = np.isfinite(yj)
            model = LinearRegression()
            model.fit(D[finite], yj[finite])
            self._models[h_name] = model
            resid = yj[finite] - model.predict(D[finite])
            self._resid_std[h_name] = float(np.std(resid)) if resid.size else 0.0
        return self

    def _require_fitted(self) -> None:
        if not self._models:
            raise RuntimeError("REFMForecaster is not fitted; call fit() first.")

    def predict(self, X, X_future) -> np.ndarray:  # noqa: D102
        self._require_fitted()
        D = self._design(X)
        out = np.column_stack([self._models[h].predict(D) for h in self.horizon_names])
        return out.astype("float32")

    def predict_quantiles(self, X, X_future) -> dict[float, np.ndarray]:  # noqa: D102
        from scipy.stats import norm

        median = self.predict(X, X_future)
        out: dict[float, np.ndarray] = {}
        for tau in self.quantiles:
            z = float(norm.ppf(tau))
            offsets = np.asarray(
                [z * self._resid_std[h] for h in self.horizon_names], dtype="float32"
            )
            out[tau] = (median + offsets[None, :]).astype("float32")
        # Guarantee monotone non-crossing across the provided quantiles.
        taus = sorted(self.quantiles)
        stacked = np.stack([out[t] for t in taus], axis=-1)
        stacked = np.sort(stacked, axis=-1)
        for i, t in enumerate(taus):
            out[t] = stacked[..., i]
        return out

    def predict_proba_exceed(self, X, X_future) -> np.ndarray:  # noqa: D102
        from scipy.stats import norm

        median = self.predict(X, X_future)
        out = np.empty_like(median)
        for j, h_name in enumerate(self.horizon_names):
            sigma = self._resid_std[h_name]
            if sigma <= 0.0:
                out[:, j] = (median[:, j] >= LOG_HARSH).astype("float32")
            else:
                out[:, j] = norm.sf(LOG_HARSH, loc=median[:, j], scale=sigma)
        return np.clip(out, 0.0, 1.0).astype("float32")

    def save(self, path: str | Path) -> None:  # noqa: D102
        joblib.dump(
            {
                "models": self._models,
                "resid_std": self._resid_std,
                "recent_steps": self.recent_steps,
                "quantiles": list(self.quantiles),
                "vsw_channel": self.vsw_channel,
                "target_channel": self.target_channel,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> REFMForecaster:  # noqa: D102
        state = joblib.load(path)
        obj = cls(
            recent_steps=state["recent_steps"],
            quantiles=tuple(state["quantiles"]),
            vsw_channel=state["vsw_channel"],
            target_channel=state["target_channel"],
        )
        obj._models = state["models"]
        obj._resid_std = state["resid_std"]
        return obj


# Backwards-compatible alias: the original scaffold named this class ``RefmLinearFilter``.
RefmLinearFilter = REFMForecaster


__all__ = [
    "Persistence",
    "Climatology",
    "LightGBMForecaster",
    "REFMForecaster",
    "RefmLinearFilter",
]
