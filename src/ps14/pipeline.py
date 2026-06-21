"""End-to-end pipeline orchestration (the glue the CLI + demo share).

This module composes the already-implemented stage modules into the four reusable
offline steps so both :mod:`ps14.cli` and ``scripts/run_demo.py`` call one code path:

1. :func:`read_raw_merged` — read the synthetic GOES + OMNI CDFs back through
   :mod:`ps14.io.cdf_reader` (exercising the real ISTP read path) and re-attach the
   static / derived columns from the parquet mirror, yielding a canonical MERGED frame.
2. :func:`preprocess_frame` — Hampel despike + short-gap interpolation (``*_imputed``
   masks) + uniform-grid enforcement + ``log10`` recompute, validated by
   :func:`ps14.datasets.schema.validate_merged`.
3. :func:`build_feature_windows` — :func:`ps14.features.offline.build_feature_matrix`
   then :func:`ps14.datasets.windowing.make_supervised` (with optional demo overrides for
   a tractable lookback / stride / subsample), validated against the schema.
4. (training / evaluation are handled by :mod:`ps14.train` and :mod:`ps14.evaluate`.)

Nothing here re-implements stage logic; it only orders the existing functions and persists
the contract artifacts (``grid_5min.parquet``, ``windows.npz``) under ``config.paths``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ps14.constants import DECODER_STEPS, HORIZON_STEPS
from ps14.datasets import schema, windowing
from ps14.features import offline
from ps14.io import cdf_reader
from ps14.preprocess import clean
from ps14.preprocess.transform import log10_floor
from ps14.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from ps14.config import Settings
    from ps14.datasets.windowing import WindowTensors

logger = get_logger("ps14.pipeline")

# Science channels that receive Hampel despiking + short-gap interpolation. (The log
# mirrors and derived columns are recomputed afterwards; static columns are never touched.)
_CLEAN_CHANNELS: tuple[str, ...] = (
    "flux_e2",
    "flux_seed",
    "vsw",
    "density",
    "bz_gsm",
    "bt",
    "ae",
    "al",
    "kp",
    "sym_h",
    "f107",
)


# ======================================================================================
# 1. Raw ingest (CDF round-trip)
# ======================================================================================
def read_raw_merged(
    raw_dir: str | Path = "data/raw/synthetic",
    *,
    prefer_cdf: bool = True,
) -> pd.DataFrame:
    """Reconstruct the canonical MERGED frame from the synthetic artifacts.

    When ``prefer_cdf`` is True and the GOES/OMNI CDFs exist, the science channels are read
    back through :func:`ps14.io.cdf_reader.read_cdf` (so the real fill/valid/TT2000 path is
    exercised) and merged with the static (``longitude``/``sat_id``) and derived (``pdyn``)
    columns taken from the parquet mirror. Otherwise the parquet mirror is returned directly.

    Returns
    -------
    pd.DataFrame
        A frame on the uniform 5-min UTC index with the canonical MERGED columns.
    """
    raw_dir = Path(raw_dir)
    parquet = raw_dir / "merged.parquet"
    goes_cdf = raw_dir / "synthetic_goes.cdf"
    omni_cdf = raw_dir / "synthetic_omni.cdf"

    if not parquet.exists():
        raise FileNotFoundError(
            f"{parquet} not found; generate it first (ps14 synth-data / synthetic.generate)."
        )
    mirror = pd.read_parquet(parquet)
    mirror.index = pd.DatetimeIndex(mirror.index, name="time").as_unit("ns")

    if not (prefer_cdf and goes_cdf.exists() and omni_cdf.exists()):
        logger.info("Reading merged frame from parquet mirror %s", parquet)
        return mirror

    logger.info("Reading science channels back through cdf_reader (%s, %s)", goes_cdf, omni_cdf)
    goes = cdf_reader.read_cdf(goes_cdf)
    omni = cdf_reader.read_cdf(omni_cdf)
    cdf_frame = goes.join(omni, how="outer").sort_index()
    cdf_frame.index = pd.DatetimeIndex(cdf_frame.index, name="time").as_unit("ns")

    # Start from the CDF-read science channels, then re-attach static / derived / mask
    # columns from the parquet mirror so the result is schema-complete.
    out = mirror.copy()
    for col in cdf_frame.columns:
        if col in out.columns:
            out[col] = cdf_frame[col].reindex(out.index)
    return out


def assert_cdf_roundtrip(
    raw_dir: str | Path = "data/raw/synthetic",
    *,
    rtol: float = 1e-6,
    atol: float = 1e-3,
) -> dict[str, float]:
    """Assert the GOES/OMNI CDFs read back to the parquet mirror within tolerance.

    Compares every science channel present in both the CDF read-back and the parquet mirror
    on their common non-NaN samples. Returns the per-channel max abs deviation for logging.
    Raises ``AssertionError`` if any channel deviates beyond ``rtol``/``atol``.
    """
    raw_dir = Path(raw_dir)
    mirror = pd.read_parquet(raw_dir / "merged.parquet")
    mirror.index = pd.DatetimeIndex(mirror.index, name="time").as_unit("ns")
    goes = cdf_reader.read_cdf(raw_dir / "synthetic_goes.cdf")
    omni = cdf_reader.read_cdf(raw_dir / "synthetic_omni.cdf")
    cdf_frame = goes.join(omni, how="outer").sort_index()
    cdf_frame.index = pd.DatetimeIndex(cdf_frame.index, name="time").as_unit("ns")

    deviations: dict[str, float] = {}
    for col in cdf_frame.columns:
        if col not in mirror.columns:
            continue
        a = mirror[col].to_numpy(dtype="float64")
        b = cdf_frame[col].reindex(mirror.index).to_numpy(dtype="float64")
        both = ~np.isnan(a) & ~np.isnan(b)
        if not both.any():
            continue
        dev = float(np.max(np.abs(a[both] - b[both])))
        deviations[col] = dev
        if not np.allclose(a[both], b[both], rtol=rtol, atol=atol):
            raise AssertionError(
                f"CDF round-trip mismatch for {col!r}: max abs dev {dev:.3g} exceeds "
                f"tol (rtol={rtol}, atol={atol})."
            )
    logger.info(
        "CDF round-trip OK (%d channels, max dev %.3g)",
        len(deviations),
        max(deviations.values(), default=0.0),
    )
    return deviations


# ======================================================================================
# 2. Preprocess (clean -> impute -> log -> validate)
# ======================================================================================
def preprocess_frame(df: pd.DataFrame, config: Settings) -> pd.DataFrame:
    """Clean the merged frame into a schema-valid 5-min grid.

    Steps (R5 §4): Hampel despike each science channel; interpolate short gaps (recording
    ``*_imputed`` masks) while leaving long gaps NaN; recompute the derived ``pdyn`` and the
    ``log_*`` mirrors; enforce dtypes; validate with :func:`schema.validate_merged`.

    Parameters
    ----------
    df:
        A canonical MERGED frame (e.g. from :func:`read_raw_merged`).
    config:
        Validated settings (uses ``preprocess.hampel`` / ``preprocess.gaps`` /
        ``preprocess.log_floor_pfu``).
    """
    pp = config.preprocess
    out = df.copy()
    out.index = pd.DatetimeIndex(out.index, name="time").as_unit("ns")
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    max_gap = pp.gaps.max_gap_steps
    for col in _CLEAN_CHANNELS:
        if col not in out.columns:
            continue
        despiked, _ = clean.hampel_filter(
            out[col], window=pp.hampel.window, n_sigma=pp.hampel.n_sigma, replace=pp.hampel.replace
        )
        filled, imputed = clean.interpolate_short_gaps(despiked, max_gap_steps=max_gap)
        out[col] = filled.astype("float64")
        mask_col = f"{col}_imputed"
        if mask_col in out.columns:
            # OR the new short-gap imputations into any existing mask.
            out[mask_col] = (out[mask_col].fillna(0).astype("int8") | imputed.to_numpy()).astype(
                "int8"
            )
        else:
            out[mask_col] = imputed.astype("int8")

    # Derived dynamic pressure (nPa) ~= 1.6726e-6 * N * Vsw^2 (offline.dynamic_pressure).
    out["pdyn"] = offline.dynamic_pressure(
        out["density"].to_numpy(dtype="float64"), out["vsw"].to_numpy(dtype="float64")
    )
    # Recompute log mirrors from the (now cleaned) linear flux channels.
    floor = pp.log_floor_pfu
    out["log_flux_e2"] = log10_floor(out["flux_e2"], floor=floor)
    out["log_flux_seed"] = log10_floor(out["flux_seed"], floor=floor)

    for col in schema.MERGED_REQUIRED:
        if col in out.columns:
            out[col] = out[col].astype("float64")
    out.index.name = "time"

    schema.validate_merged(out, expected_freq=pp.cadence)
    logger.info("Preprocessed merged frame: %d rows, %d columns", len(out), out.shape[1])
    return out


# ======================================================================================
# 3. Features + supervised windows
# ======================================================================================
def build_feature_windows(
    df: pd.DataFrame,
    config: Settings,
    *,
    lookback: int | None = None,
    decoder_steps: int | None = None,
    stride: int = 1,
    max_windows: int | None = None,
) -> WindowTensors:
    """Build the feature matrix then leakage-free supervised windows.

    Parameters
    ----------
    df:
        A schema-valid MERGED frame (output of :func:`preprocess_frame`).
    config:
        Validated settings (``features.lookback_steps`` is the default lookback).
    lookback:
        Encoder length override ``L`` (defaults to ``config.features.lookback_steps``). A
        SHORTER value (e.g. 288 = 1 day) keeps the dense window tensors tractable on CPU.
    decoder_steps:
        Decoder length ``H`` override (default ``constants.DECODER_STEPS`` = 144).
    stride:
        Keep every ``stride``-th window (subsampling in time) to cap memory; the windows
        remain chronological so the split contract still holds.
    max_windows:
        Optional hard cap on the number of windows (after striding), keeping the most recent
        ``max_windows`` so a demo run finishes in minutes.
    """
    lookback = int(config.features.lookback_steps if lookback is None else lookback)
    decoder_steps = int(DECODER_STEPS if decoder_steps is None else decoder_steps)

    feat = offline.build_feature_matrix(df, config.features)
    schema.validate_features(feat)

    wt = windowing.make_supervised(
        feat,
        feature_cols=schema.FEATURE_COLUMNS,
        known_future_cols=schema.KNOWN_FUTURE_COLUMNS,
        target_col=schema.TARGET,
        lookback=lookback,
        decoder_steps=decoder_steps,
        horizon_steps=HORIZON_STEPS,
    )
    if stride > 1 or max_windows is not None:
        wt = _subsample_windows(wt, stride=stride, max_windows=max_windows)
    logger.info(
        "Built %d windows (L=%d, H=%d, F=%d, stride=%d)",
        wt.X.shape[0],
        lookback,
        decoder_steps,
        wt.X.shape[2],
        stride,
    )
    return wt


def _subsample_windows(
    wt: WindowTensors, *, stride: int = 1, max_windows: int | None = None
) -> WindowTensors:
    """Keep every ``stride``-th window (and at most ``max_windows`` most-recent ones)."""
    n = wt.X.shape[0]
    idx = np.arange(0, n, max(1, stride))
    if max_windows is not None and idx.size > max_windows:
        idx = idx[-max_windows:]
    return windowing.WindowTensors(
        X=wt.X[idx],
        X_future=wt.X_future[idx],
        y=wt.y[idx],
        y_exceed=wt.y_exceed[idx],
        t_index=wt.t_index[idx],
        feature_cols=wt.feature_cols,
        known_future_cols=wt.known_future_cols,
        horizon_names=wt.horizon_names,
    )


# ======================================================================================
# Convenience: persist the contract artifacts
# ======================================================================================
def save_merged(df: pd.DataFrame, config: Settings) -> Path:
    """Write the merged 5-min frame to ``data/processed/grid_5min.parquet``."""
    out = Path(config.paths.data_processed) / "grid_5min.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    logger.info("Wrote merged grid -> %s", out)
    return out


def load_merged(config: Settings) -> pd.DataFrame:
    """Read back ``data/processed/grid_5min.parquet`` as a UTC-indexed frame."""
    path = Path(config.paths.data_processed) / "grid_5min.parquet"
    df = pd.read_parquet(path)
    df.index = pd.DatetimeIndex(df.index, name="time").as_unit("ns")
    return df


__all__ = [
    "read_raw_merged",
    "assert_cdf_roundtrip",
    "preprocess_frame",
    "build_feature_windows",
    "save_merged",
    "load_merged",
]
