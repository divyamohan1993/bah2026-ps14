"""Evaluation harness: per-horizon regression + event + probabilistic metrics (R3 §8).

Runs a fitted :class:`~ps14.models.base.Forecaster` against a held-out test set (or a
walk-forward CV schedule), computes the metric suite from :mod:`ps14.metrics` PER NAMED
HORIZON, compares against the persistence/climatology references, and writes a tidy
report to ``reports/metrics_<model>.{json,csv}`` (CONTRACTS.md §6/§8).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ps14.models.base import Forecaster


def evaluate_model(
    model: Forecaster,
    X: np.ndarray,
    X_future: np.ndarray,
    y: np.ndarray,
    y_exceed: np.ndarray,
    *,
    reference: Forecaster | None = None,
    threshold_log: float | None = None,
) -> dict[str, dict[str, float]]:
    """Score a model per named horizon (regression + event + probabilistic).

    Parameters
    ----------
    model:
        A fitted forecaster.
    X, X_future, y, y_exceed:
        Held-out test tensors (shapes per CONTRACTS.md §4).
    reference:
        Optional reference model (e.g. Persistence) for the skill score.
    threshold_log:
        Exceedance threshold in log10 space (default ``constants.LOG_HARSH``).

    Returns
    -------
    dict
        ``{horizon_name: {metric: value}}`` covering regression, event, and probabilistic
        metrics; suitable for serialization.
    """
    raise NotImplementedError(
        "TODO: y_pred = model.predict(...); q = model.predict_quantiles(...); "
        "p = model.predict_proba_exceed(...); for each horizon column call "
        "metrics.regression_report / event_report / pinball/crps/picp; include skill vs "
        "reference.predict(...) (R3 §8)."
    )


def walk_forward_cv(
    model_factory,
    tensors,
    *,
    n_folds: int = 5,
    embargo_steps: int = 1296,
) -> list[dict[str, dict[str, float]]]:
    """Expanding-window walk-forward cross-validation (R3 §8e).

    Trains on an expanding block and validates on the next, sliding forward, with an
    embargo gap — mirrors operational retraining.

    Parameters
    ----------
    model_factory:
        Zero-arg callable returning a fresh unfitted :class:`Forecaster`.
    tensors:
        The full :class:`~ps14.datasets.windowing.WindowTensors`.
    n_folds:
        Number of expanding folds.
    embargo_steps:
        Purge/embargo between train and validation each fold.

    Returns
    -------
    list[dict]
        Per-fold per-horizon metric dicts.
    """
    raise NotImplementedError(
        "TODO: build expanding (train, val) fold boundaries with embargo; for each fold "
        "fit a fresh model and call evaluate_model; collect results."
    )


def write_report(results: dict, path: str | Path) -> None:
    """Write the per-horizon metric report to JSON + CSV (CONTRACTS.md §8)."""
    raise NotImplementedError(
        "TODO: json.dump(results, ...); flatten to a tidy CSV keyed by horizon."
    )


__all__ = ["evaluate_model", "walk_forward_cv", "write_report"]
