"""Supervised windowing + chronological split with purge/embargo (CONTRACTS.md §4).

Converts the feature matrix into the model-ready tensors and produces leakage-free
chronological splits. The chronological split is FULLY IMPLEMENTED (small, exact, and
tested); the windowing function carries the no-look-ahead contract in its docstring.

Tensor shapes (CONTRACTS.md §4):
  X:        [N, L, F]      encoder features over [t-L+1 .. t]
  X_future: [N, H, F_kf]   known-future covariates over [t+1 .. t+H]
  y:        [N, n_h]       log10 flux at the named horizons
  y_exceed: [N, n_h]       1[flux >= HARSH_PFU] at each horizon
  t_index:  [N]            anchor time t per window
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ps14.constants import (
    DECODER_STEPS,
    DEFAULT_LOOKBACK_STEPS,
    HORIZON_NAMES,
    HORIZON_STEPS,
    LOG_HARSH,
)


@dataclass
class WindowTensors:
    """Container for the supervised window arrays (CONTRACTS.md §4)."""

    X: np.ndarray  # [N, L, F] float32
    X_future: np.ndarray  # [N, H, F_kf] float32
    y: np.ndarray  # [N, n_h] float32
    y_exceed: np.ndarray  # [N, n_h] float32 in {0,1}
    t_index: np.ndarray  # [N] datetime64[ns]
    feature_cols: list[str]
    known_future_cols: list[str]
    horizon_names: list[str]


def make_supervised(
    df: pd.DataFrame,
    feature_cols: list[str],
    known_future_cols: list[str],
    target_col: str,
    *,
    lookback: int = DEFAULT_LOOKBACK_STEPS,
    decoder_steps: int = DECODER_STEPS,
    horizon_steps: dict[str, int] | None = None,
    log_harsh: float = LOG_HARSH,
    drop_if_nan: bool = True,
) -> WindowTensors:
    """Build leakage-free supervised windows from the feature matrix.

    For each anchor time ``t``: ``X`` is the features over ``[t-lookback+1 .. t]``,
    ``X_future`` the known-future covariates over ``[t+1 .. t+decoder_steps]``, ``y`` the
    target at the named horizons, and ``y_exceed`` the exceedance label.

    Hard rules (R5 §5.2, CONTRACTS.md §4):
      * no value at ``> t`` appears in ``X``;
      * ``y``/``y_exceed`` are strictly future;
      * windows whose ``X`` or required ``y`` span a long-gap NaN are dropped when
        ``drop_if_nan`` is True;
      * returned arrays are UNSCALED (scaling is applied later, fit on train only).

    Parameters
    ----------
    df:
        Feature matrix (validated by ``schema.validate_features``), 5-min UTC index.
    feature_cols:
        Encoder feature columns (channel order preserved) — usually ``schema.FEATURE_COLUMNS``.
    known_future_cols:
        Decoder known-future columns — usually ``schema.KNOWN_FUTURE_COLUMNS``.
    target_col:
        The log-flux target column (``schema.TARGET``).
    lookback, decoder_steps:
        Encoder length ``L`` and decoder length ``H``.
    horizon_steps:
        Map of named horizon -> step offset (default ``constants.HORIZON_STEPS``).
    log_harsh:
        Exceedance threshold in log10 space (default ``constants.LOG_HARSH`` = 3.0).
    drop_if_nan:
        Drop windows containing NaN in ``X`` or required ``y``.

    Returns
    -------
    WindowTensors
        The stacked arrays + metadata.
    """
    raise NotImplementedError(
        "TODO: slide t over [lookback-1, n - decoder_steps); X = F[t-lookback+1:t+1]; "
        "Xf = KF[t+1:t+1+decoder_steps]; y[h] = target[t + horizon_steps[h]]; "
        "y_exceed = (y >= log_harsh); drop NaN windows; stack to float32 (CONTRACTS.md §4)."
    )


def chronological_split(
    t_index: np.ndarray | pd.DatetimeIndex,
    *,
    train: float = 0.70,
    val: float = 0.15,
    embargo_steps: int = 1296,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split window indices chronologically with an embargo gap (R5 §5.2, CONTRACTS.md §4.1).

    Returns integer index arrays into ``N`` for train/val/test with ``embargo_steps``
    windows removed between consecutive segments so no window straddles a boundary. Order
    is strictly ``train < val < test`` by time.

    Parameters
    ----------
    t_index:
        Per-window anchor times (only its length/order is used).
    train, val:
        Fractions of windows for train/val (test = remainder).
    embargo_steps:
        Number of windows purged at each seam (must be ``>= lookback + max horizon``).

    Returns
    -------
    (train_idx, val_idx, test_idx):
        Integer position arrays into the window axis.
    """
    n = len(t_index)
    if n == 0:
        empty = np.array([], dtype="int64")
        return empty, empty.copy(), empty.copy()
    i_tr = int(n * train)
    i_va = int(n * (train + val))
    train_idx = np.arange(0, max(i_tr - embargo_steps, 0), dtype="int64")
    val_idx = np.arange(min(i_tr + embargo_steps, n), max(i_va - embargo_steps, 0), dtype="int64")
    test_idx = np.arange(min(i_va + embargo_steps, n), n, dtype="int64")
    return train_idx, val_idx, test_idx


def save_windows(tensors: WindowTensors, path: str | Path) -> None:
    """Persist window tensors to ``windows.npz`` (CONTRACTS.md §8)."""
    raise NotImplementedError(
        "TODO: np.savez_compressed(path, X=..., X_future=..., y=..., y_exceed=..., "
        "t_index=..., feature_cols=..., known_future_cols=..., horizon_names=...)."
    )


def load_windows(path: str | Path) -> WindowTensors:
    """Load window tensors from ``windows.npz`` into a :class:`WindowTensors`."""
    raise NotImplementedError("TODO: np.load(path, allow_pickle=True) -> WindowTensors(...).")


# Re-export the horizon contract for convenience.
HORIZON_STEPS_DEFAULT = HORIZON_STEPS
HORIZON_NAMES_DEFAULT = HORIZON_NAMES

__all__ = [
    "WindowTensors",
    "make_supervised",
    "chronological_split",
    "save_windows",
    "load_windows",
    "HORIZON_STEPS_DEFAULT",
    "HORIZON_NAMES_DEFAULT",
]
