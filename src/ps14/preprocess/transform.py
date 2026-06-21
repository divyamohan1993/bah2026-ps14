"""Transforms: log10 with positive floor, and train-only feature scaling (R5 §4.6, §4.9).

``log10_floor`` is FULLY IMPLEMENTED (small, exact). The scaler wrapper encodes the
LEAKAGE-CRITICAL contract: fit statistics on the TRAIN window only, then transform
val/test (and never fit on GRASP/GSAT).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


def log10_floor(x: np.ndarray | pd.Series, floor: float = 0.01) -> np.ndarray | pd.Series:
    """``log10`` with a positive floor applied first; non-positives -> ``log10(floor)``.

    Parameters
    ----------
    x:
        Linear flux values (may contain NaN, zeros, or small negatives from despiking).
    floor:
        Positive floor (instrument noise floor, e.g. ``0.01`` pfu). Recorded so the
        inverse transform is well-defined. Never ``log10(0)`` (R5 §4.6).

    Returns
    -------
    Same type as input
        ``log10(max(x, floor))``; NaNs are preserved as NaN.
    """
    is_series = isinstance(x, pd.Series)
    arr = np.asarray(x, dtype="float64")
    nan_mask = np.isnan(arr)
    floored = np.where(arr < floor, floor, arr)
    out = np.log10(floored)
    out[nan_mask] = np.nan
    if is_series:
        return pd.Series(out, index=x.index, name=x.name)  # type: ignore[union-attr]
    return out


def inverse_log10(x: np.ndarray | pd.Series) -> np.ndarray | pd.Series:
    """Inverse of :func:`log10_floor` (``10**x``); NaNs preserved."""
    if isinstance(x, pd.Series):
        return pd.Series(np.power(10.0, np.asarray(x, dtype="float64")), index=x.index, name=x.name)
    return np.power(10.0, np.asarray(x, dtype="float64"))


def _new_sklearn_scaler(kind: str):
    """Instantiate a fresh, unfitted scikit-learn scaler for ``kind``."""
    if kind == "standard":
        return StandardScaler()
    if kind == "robust":
        return RobustScaler()
    raise ValueError(f"kind must be 'standard' or 'robust', got {kind!r}")


def fit_scaler(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    kind: str = "standard",
) -> dict[str, Any]:
    """Fit a feature scaler on the TRAIN slice only (leakage-critical, R5 §4.9).

    The fit statistics (center / scale per column) are computed **exclusively** from the
    rows in ``df`` — the caller is responsible for passing the chronological TRAIN slice so
    that no validation/test/GRASP statistics leak into the transform (R5 §4.9, §5.2).

    Parameters
    ----------
    df:
        TRAIN-only frame (the caller slices chronologically before calling this).
    feature_cols:
        Columns to scale (channel order is preserved in the returned state).
    kind:
        ``"standard"`` (z-score) or ``"robust"`` (median/IQR; better for heavy tails).

    Returns
    -------
    dict
        A JSON-serialisable ``scaler_state`` with keys ``kind``, ``columns``, ``center``,
        and ``scale`` (per-column floats). Consumed by :func:`apply_scaler` /
        :func:`inverse_scaler` and persisted by :func:`save_scaler`. NaNs in the train
        slice are ignored when computing the statistics (column-wise nan-aware moments).
    """
    cols = list(feature_cols)
    skl = _new_sklearn_scaler(kind)
    data = df.loc[:, cols].to_numpy(dtype="float64")

    if kind == "standard":
        center = np.nanmean(data, axis=0)
        scale = np.nanstd(data, axis=0)
    else:  # robust: median + IQR (matches sklearn RobustScaler defaults)
        center = np.nanmedian(data, axis=0)
        q75 = np.nanpercentile(data, 75, axis=0)
        q25 = np.nanpercentile(data, 25, axis=0)
        scale = q75 - q25

    center = np.where(np.isfinite(center), center, 0.0)
    # Guard against zero/degenerate scale (constant columns) -> divide-by-1 (no-op).
    scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)

    # Prime the sklearn scaler with the same fitted attributes so a downstream consumer can
    # use it directly if desired (kept as metadata, not the source of truth for transform).
    skl.fit(np.where(np.isnan(data), center, data))

    return {
        "kind": kind,
        "columns": cols,
        "center": center.astype("float64").tolist(),
        "scale": scale.astype("float64").tolist(),
    }


def apply_scaler(df: pd.DataFrame, scaler_state: dict[str, Any]) -> pd.DataFrame:
    """Transform a frame with an already-fit ``scaler_state`` (returns a copy).

    ``(x - center) / scale`` per column. Columns absent from ``df`` are skipped; NaNs are
    preserved as NaN (never imputed by scaling).
    """
    cols = scaler_state["columns"]
    center = np.asarray(scaler_state["center"], dtype="float64")
    scale = np.asarray(scaler_state["scale"], dtype="float64")
    out = df.copy()
    for col, c, s in zip(cols, center, scale, strict=True):
        if col in out.columns:
            out[col] = (out[col].to_numpy(dtype="float64") - c) / s
    return out


def inverse_scaler(df: pd.DataFrame, scaler_state: dict[str, Any]) -> pd.DataFrame:
    """Invert :func:`apply_scaler` (``x * scale + center`` per column); returns a copy."""
    cols = scaler_state["columns"]
    center = np.asarray(scaler_state["center"], dtype="float64")
    scale = np.asarray(scaler_state["scale"], dtype="float64")
    out = df.copy()
    for col, c, s in zip(cols, center, scale, strict=True):
        if col in out.columns:
            out[col] = out[col].to_numpy(dtype="float64") * s + c
    return out


def save_scaler(scaler_state: dict[str, Any], path: str | Path) -> None:
    """Persist a fitted ``scaler_state`` (joblib) to ``models/scaler_<split>.joblib``.

    The state is a plain dict (CONTRACTS.md §8); joblib is used for parity with the other
    persisted artifacts. The dict is also JSON-serialisable for human inspection.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_state, path)


def load_scaler(path: str | Path) -> dict[str, Any]:
    """Load a persisted ``scaler_state`` (joblib)."""
    return joblib.load(path)


__all__ = [
    "log10_floor",
    "inverse_log10",
    "fit_scaler",
    "apply_scaler",
    "inverse_scaler",
    "save_scaler",
    "load_scaler",
]
