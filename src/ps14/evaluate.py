"""Evaluation harness: per-horizon regression + event + probabilistic metrics (R3 §8).

Scores a fitted :class:`~ps14.models.base.Forecaster` against a held-out split, computing the
full metric suite from :mod:`ps14.metrics` PER NAMED HORIZON, including skill vs persistence
(and any supplied climatology/REFM reference). Also provides model comparison, a tidy metrics
table, and report plots (forecast-vs-actual with an uncertainty band, a reliability diagram,
and a skill-vs-horizon bar) saved under ``reports/`` (CONTRACTS.md §6/§8).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ps14 import metrics
from ps14.constants import HARSH_PFU, HORIZON_NAMES, LOG_HARSH
from ps14.datasets import schema
from ps14.models.base import Forecaster

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from ps14.datasets.windowing import WindowTensors

_TARGET_CHANNEL = schema.FEATURE_COLUMNS.index(schema.TARGET)


# ======================================================================================
# Split selection
# ======================================================================================
def _select_split(
    windows: WindowTensors, split: str, split_kwargs: dict[str, Any] | None = None
):
    """Return ``(X, X_future, y, y_exceed)`` for the requested chronological split.

    ``split_kwargs`` (``train``, ``val``, ``embargo_steps``) is forwarded to
    :func:`windowing.chronological_split` so the evaluation split can match the one used at
    training time. With the contract default embargo (1296) a small array can yield an empty
    split; pass a smaller ``embargo_steps`` for tiny/test data.
    """
    from ps14.datasets import windowing

    if split == "all":
        return windows.X, windows.X_future, windows.y, windows.y_exceed
    train_idx, val_idx, test_idx = windowing.chronological_split(
        windows.t_index, **(split_kwargs or {})
    )
    idx = {"train": train_idx, "val": val_idx, "test": test_idx}.get(split)
    if idx is None:
        raise ValueError(f"split must be one of train|val|test|all, got {split!r}")
    return windows.X[idx], windows.X_future[idx], windows.y[idx], windows.y_exceed[idx]


def _persistence_reference(X: np.ndarray, n_h: int) -> np.ndarray:
    """Persistence forecast (last observed log-flux repeated) as the skill reference."""
    last = np.asarray(X)[:, -1, _TARGET_CHANNEL]
    return np.repeat(last[:, None], n_h, axis=1).astype("float32")


# ======================================================================================
# Core scoring
# ======================================================================================
def evaluate_model(
    model: Forecaster,
    X: np.ndarray,
    X_future: np.ndarray,
    y: np.ndarray,
    y_exceed: np.ndarray,
    *,
    references: dict[str, Forecaster] | None = None,
    threshold_log: float | None = None,
) -> dict[str, dict[str, float]]:
    """Score a model per named horizon (regression + event + probabilistic).

    Parameters
    ----------
    model:
        A fitted forecaster.
    X, X_future, y, y_exceed:
        Held-out tensors (shapes per CONTRACTS.md §4).
    references:
        Optional ``{name: fitted Forecaster}`` whose ``predict`` provides a skill reference
        (e.g. ``{"climatology": clim}``). A persistence reference is always added.
    threshold_log:
        Exceedance threshold in log10 space (default ``constants.LOG_HARSH``).

    Returns
    -------
    dict
        ``{horizon_name: {metric: value}}`` with regression, event, and probabilistic
        metrics plus ``skill_vs_<ref>`` entries.
    """
    threshold_log = LOG_HARSH if threshold_log is None else threshold_log
    horizon_names = list(getattr(model, "horizon_names", HORIZON_NAMES))
    n_h = len(horizon_names)

    y = np.asarray(y, dtype="float64")
    y_exceed = np.asarray(y_exceed, dtype="float64")
    y_pred = np.asarray(model.predict(X, X_future), dtype="float64")
    quantiles = {float(k): np.asarray(v, dtype="float64") for k, v in
                 model.predict_quantiles(X, X_future).items()}
    proba = np.asarray(model.predict_proba_exceed(X, X_future), dtype="float64")

    # Reference predictions for the skill score.
    ref_preds: dict[str, np.ndarray] = {
        "persistence": _persistence_reference(X, n_h).astype("float64")
    }
    if references:
        for ref_name, ref_model in references.items():
            try:
                ref_preds[ref_name] = np.asarray(ref_model.predict(X, X_future), dtype="float64")
            except Exception as exc:  # pragma: no cover - defensive
                ref_preds[ref_name] = np.full_like(y_pred, np.nan)
                del exc

    taus = sorted(quantiles)
    lower = quantiles.get(taus[0]) if taus else None
    upper = quantiles.get(taus[-1]) if taus else None

    results: dict[str, dict[str, float]] = {}
    for j, h_name in enumerate(horizon_names):
        yt, yp = y[:, j], y_pred[:, j]
        reg = metrics.regression_report(yt, yp)
        for ref_name, ref in ref_preds.items():
            reg[f"skill_vs_{ref_name}"] = metrics.skill_score(yt, yp, ref[:, j])

        # Event metrics (binary truth at the threshold; binary pred from P50 crossing).
        true_event = y_exceed[:, j]
        pred_event = (yp >= threshold_log).astype("float64")
        event = metrics.event_report(true_event, pred_event, y_prob=proba[:, j])

        prob: dict[str, float] = {}
        for tau in taus:
            prob[f"pinball_{tau:g}"] = metrics.pinball_loss(yt, quantiles[tau][:, j], tau)
        prob["crps"] = metrics.crps_from_quantiles(
            yt, {tau: quantiles[tau][:, j] for tau in taus}
        )
        if lower is not None and upper is not None and len(taus) >= 2:
            prob["picp"] = metrics.picp(yt, lower[:, j], upper[:, j])
            prob["picp_nominal"] = float(taus[-1] - taus[0])

        results[h_name] = {**reg, **event, **prob}
    return results


def evaluate(
    model: Forecaster,
    windows: WindowTensors,
    split: str = "test",
    *,
    references: dict[str, Forecaster] | None = None,
    threshold_log: float | None = None,
    split_kwargs: dict[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate a fitted model on a chronological split of the window tensors.

    Thin wrapper over :func:`evaluate_model` that selects the ``train|val|test|all`` split
    from a :class:`~ps14.datasets.windowing.WindowTensors` and returns the per-horizon
    metric dict. ``split_kwargs`` (``train``/``val``/``embargo_steps``) is forwarded to the
    chronological split so it can match training.
    """
    X, X_future, y, y_exceed = _select_split(windows, split, split_kwargs)
    return evaluate_model(
        model, X, X_future, y, y_exceed, references=references, threshold_log=threshold_log
    )


def compare_models(
    models: dict[str, Forecaster],
    windows: WindowTensors,
    split: str = "test",
    *,
    references: dict[str, Forecaster] | None = None,
    split_kwargs: dict[str, Any] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate several models on the same split.

    Returns ``{model_name: {horizon: {metric: value}}}``.
    """
    return {
        name: evaluate(model, windows, split, references=references, split_kwargs=split_kwargs)
        for name, model in models.items()
    }


# ======================================================================================
# Reporting
# ======================================================================================
def metrics_table(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Render a per-horizon metric dict as a tidy :class:`pandas.DataFrame` (rows=horizon)."""
    import pandas as pd

    return pd.DataFrame(results).T.rename_axis("horizon")


def comparison_table(
    comparison: dict[str, dict[str, dict[str, float]]],
) -> pd.DataFrame:
    """Render :func:`compare_models` output as a long DataFrame keyed by (model, horizon)."""
    import pandas as pd

    rows = []
    for model_name, per_h in comparison.items():
        for horizon, vals in per_h.items():
            rows.append({"model": model_name, "horizon": horizon, **vals})
    return pd.DataFrame(rows).set_index(["model", "horizon"])


def write_report(results: dict, path: str | Path) -> None:
    """Write a per-horizon metric report to JSON + CSV (CONTRACTS.md §8).

    ``path`` is the JSON path; a sibling ``.csv`` is written from the tidy table.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    try:
        metrics_table(results).to_csv(path.with_suffix(".csv"))
    except Exception:  # pragma: no cover - CSV is best-effort
        pass


# ======================================================================================
# Plots (matplotlib; saved under reports/)
# ======================================================================================
def _ensure_reports_dir(reports_dir: str | Path) -> Path:
    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_forecast_vs_actual(
    model: Forecaster,
    windows: WindowTensors,
    split: str = "test",
    *,
    reports_dir: str | Path = "reports",
    max_points: int = 500,
    split_kwargs: dict[str, Any] | None = None,
) -> Path:
    """Plot forecast-vs-actual time series with a P10-P90 band, one panel per horizon."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, X_future, y, _ = _select_split(windows, split, split_kwargs)
    horizon_names = list(getattr(model, "horizon_names", HORIZON_NAMES))
    y_pred = np.asarray(model.predict(X, X_future))
    q = {float(k): np.asarray(v) for k, v in model.predict_quantiles(X, X_future).items()}
    taus = sorted(q)
    lo = q[taus[0]] if taus else y_pred
    hi = q[taus[-1]] if taus else y_pred

    n = min(max_points, y.shape[0])
    sl = slice(0, n)
    fig, axes = plt.subplots(len(horizon_names), 1, figsize=(11, 2.6 * len(horizon_names)),
                             sharex=True, squeeze=False)
    for j, h_name in enumerate(horizon_names):
        ax = axes[j, 0]
        ax.plot(np.asarray(y)[sl, j], color="black", lw=1.0, label="actual")
        ax.plot(y_pred[sl, j], color="tab:blue", lw=1.0, label="P50")
        if taus:
            ax.fill_between(
                np.arange(n), lo[sl, j], hi[sl, j], color="tab:blue", alpha=0.2,
                label=f"P{int(taus[0] * 100)}-P{int(taus[-1] * 100)}",
            )
        ax.axhline(LOG_HARSH, color="tab:red", ls="--", lw=0.8, label=f"{HARSH_PFU:g} pfu")
        ax.set_ylabel(f"log10 flux\n({h_name})")
        ax.legend(loc="upper right", fontsize=7, ncol=4)
    axes[-1, 0].set_xlabel("test window index")
    model_name = getattr(model, "name", "model")
    fig.suptitle(f"{model_name}: forecast vs actual ({split})")
    fig.tight_layout()
    out = _ensure_reports_dir(reports_dir) / f"forecast_vs_actual_{model_name}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_reliability(
    model: Forecaster,
    windows: WindowTensors,
    split: str = "test",
    *,
    reports_dir: str | Path = "reports",
    n_bins: int = 10,
    split_kwargs: dict[str, Any] | None = None,
) -> Path:
    """Plot the exceedance reliability (calibration) diagram per horizon."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, X_future, _, y_exceed = _select_split(windows, split, split_kwargs)
    horizon_names = list(getattr(model, "horizon_names", HORIZON_NAMES))
    proba = np.asarray(model.predict_proba_exceed(X, X_future))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1.0, label="perfect")
    for j, h_name in enumerate(horizon_names):
        centers, obs, counts = metrics.reliability_curve(
            np.asarray(y_exceed)[:, j], proba[:, j], n_bins=n_bins
        )
        valid = counts > 0
        ax.plot(centers[valid], obs[valid], marker="o", lw=1.2, label=h_name)
    ax.set_xlabel("forecast probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(f"{getattr(model, 'name', 'model')}: reliability ({split})")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out = _ensure_reports_dir(reports_dir) / f"reliability_{getattr(model, 'name', 'model')}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_skill_vs_horizon(
    results: dict[str, dict[str, float]],
    *,
    reports_dir: str | Path = "reports",
    skill_key: str = "skill_vs_persistence",
    model_name: str = "model",
) -> Path:
    """Bar chart of per-horizon skill score vs the persistence reference."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = list(results.keys())
    values = [float(results[h].get(skill_key, np.nan)) for h in horizons]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["tab:green" if v >= 0 else "tab:red" for v in values]
    ax.bar(horizons, values, color=colors)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_ylabel(skill_key)
    ax.set_title(f"{model_name}: skill vs persistence by horizon")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    fig.tight_layout()
    out = _ensure_reports_dir(reports_dir) / f"skill_vs_horizon_{model_name}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def render_report(
    model: Forecaster,
    windows: WindowTensors,
    split: str = "test",
    *,
    references: dict[str, Forecaster] | None = None,
    reports_dir: str | Path = "reports",
    split_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute metrics, write the JSON/CSV report, and save all three plots.

    Returns ``{"results": ..., "report_json": path, "figures": {name: path}}``.
    """
    model_name = getattr(model, "name", "model")
    results = evaluate(model, windows, split, references=references, split_kwargs=split_kwargs)
    report_json = Path(reports_dir) / f"metrics_{model_name}.json"
    write_report(results, report_json)
    figures = {
        "forecast_vs_actual": plot_forecast_vs_actual(
            model, windows, split, reports_dir=reports_dir, split_kwargs=split_kwargs
        ),
        "reliability": plot_reliability(
            model, windows, split, reports_dir=reports_dir, split_kwargs=split_kwargs
        ),
        "skill_vs_horizon": plot_skill_vs_horizon(
            results, reports_dir=reports_dir, model_name=model_name
        ),
    }
    return {"results": results, "report_json": report_json, "figures": figures}


__all__ = [
    "evaluate",
    "evaluate_model",
    "compare_models",
    "metrics_table",
    "comparison_table",
    "write_report",
    "plot_forecast_vs_actual",
    "plot_reliability",
    "plot_skill_vs_horizon",
    "render_report",
]
