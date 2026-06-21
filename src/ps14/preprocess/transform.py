"""Transforms: log10 with positive floor, and train-only feature scaling (R5 §4.6, §4.9).

``log10_floor`` is FULLY IMPLEMENTED (small, exact). The scaler wrapper encodes the
LEAKAGE-CRITICAL contract: fit statistics on the TRAIN window only, then transform
val/test (and never fit on GRASP/GSAT).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


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


def fit_scaler(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    kind: str = "standard",
):
    """Fit a feature scaler on the TRAIN slice only (leakage-critical, R5 §4.9).

    Parameters
    ----------
    df:
        TRAIN-only frame (the caller slices chronologically before calling this).
    feature_cols:
        Columns to scale.
    kind:
        ``"standard"`` (z-score) or ``"robust"`` (median/IQR; better for heavy tails).

    Returns
    -------
    Fitted scaler
        A scikit-learn scaler fit on ``df[feature_cols]`` (ignoring NaN per sklearn rules).
        Persist with :func:`save_scaler`.
    """
    raise NotImplementedError(
        "TODO: instantiate StandardScaler/RobustScaler; fit on df[feature_cols]; "
        "return it. MUST be called on the train slice only (R5 §4.9)."
    )


def apply_scaler(df: pd.DataFrame, feature_cols: list[str], scaler) -> pd.DataFrame:
    """Transform ``df[feature_cols]`` with an already-fit scaler (returns a copy)."""
    raise NotImplementedError(
        "TODO: out = df.copy(); out[feature_cols] = scaler.transform(...); return out."
    )


def save_scaler(scaler, path: str | Path) -> None:
    """Persist a fitted scaler (joblib) to ``models/scaler_<split>.joblib`` (CONTRACTS.md §8)."""
    raise NotImplementedError("TODO: joblib.dump(scaler, path).")


def load_scaler(path: str | Path):
    """Load a persisted scaler (joblib)."""
    raise NotImplementedError("TODO: return joblib.load(path).")


__all__ = [
    "log10_floor",
    "inverse_log10",
    "fit_scaler",
    "apply_scaler",
    "save_scaler",
    "load_scaler",
]
