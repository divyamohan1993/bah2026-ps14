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
    if horizon_steps is None:
        horizon_steps = HORIZON_STEPS
    horizon_names = list(horizon_steps.keys())
    offsets = np.asarray([horizon_steps[h] for h in horizon_names], dtype="int64")

    feat = df.loc[:, feature_cols].to_numpy(dtype="float32")
    kf = df.loc[:, known_future_cols].to_numpy(dtype="float32")
    target = df[target_col].to_numpy(dtype="float32")
    index = df.index.to_numpy()

    n = len(df)
    # Need [t+1 .. t+decoder_steps] for X_future and t+max(offset) for y.
    max_future = int(max(decoder_steps, int(offsets.max(initial=0))))
    last_t = n - 1 - max_future  # inclusive

    x_list: list[np.ndarray] = []
    xf_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    yexc_list: list[np.ndarray] = []
    t_list: list[np.datetime64] = []

    for t in range(lookback - 1, last_t + 1):
        x = feat[t - lookback + 1 : t + 1]  # [L, F], only up to and incl. t
        xf = kf[t + 1 : t + 1 + decoder_steps]  # [H, F_kf], strictly future
        y_vals = target[t + offsets]  # [n_h], strictly future

        if drop_if_nan and (np.isnan(x).any() or np.isnan(y_vals).any() or np.isnan(xf).any()):
            continue

        x_list.append(x)
        xf_list.append(xf)
        y_list.append(y_vals)
        yexc_list.append((y_vals >= log_harsh).astype("float32"))
        t_list.append(index[t])

    n_feat = len(feature_cols)
    n_kf = len(known_future_cols)
    n_h = len(horizon_names)
    if x_list:
        x_arr = np.stack(x_list).astype("float32")
        xf_arr = np.stack(xf_list).astype("float32")
        y_arr = np.stack(y_list).astype("float32")
        yexc_arr = np.stack(yexc_list).astype("float32")
        t_arr = np.asarray(t_list, dtype="datetime64[ns]")
    else:  # no valid windows -> correctly-shaped empties
        x_arr = np.empty((0, lookback, n_feat), dtype="float32")
        xf_arr = np.empty((0, decoder_steps, n_kf), dtype="float32")
        y_arr = np.empty((0, n_h), dtype="float32")
        yexc_arr = np.empty((0, n_h), dtype="float32")
        t_arr = np.empty((0,), dtype="datetime64[ns]")

    return WindowTensors(
        X=x_arr,
        X_future=xf_arr,
        y=y_arr,
        y_exceed=yexc_arr,
        t_index=t_arr,
        feature_cols=list(feature_cols),
        known_future_cols=list(known_future_cols),
        horizon_names=horizon_names,
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
    """Persist window tensors to ``windows.npz`` via ``np.savez_compressed`` (CONTRACTS.md §8).

    Keys: ``X, X_future, y, y_exceed, t_index, feature_cols, known_future_cols,
    horizon_names``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=tensors.X,
        X_future=tensors.X_future,
        y=tensors.y,
        y_exceed=tensors.y_exceed,
        t_index=tensors.t_index,
        feature_cols=np.asarray(tensors.feature_cols, dtype=object),
        known_future_cols=np.asarray(tensors.known_future_cols, dtype=object),
        horizon_names=np.asarray(tensors.horizon_names, dtype=object),
    )


def load_windows(path: str | Path) -> WindowTensors:
    """Load window tensors from ``windows.npz`` into a :class:`WindowTensors`."""
    with np.load(path, allow_pickle=True) as data:
        return WindowTensors(
            X=data["X"],
            X_future=data["X_future"],
            y=data["y"],
            y_exceed=data["y_exceed"],
            t_index=data["t_index"],
            feature_cols=list(data["feature_cols"].tolist()),
            known_future_cols=list(data["known_future_cols"].tolist()),
            horizon_names=list(data["horizon_names"].tolist()),
        )


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
