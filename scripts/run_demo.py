#!/usr/bin/env python3
"""End-to-end PS-14 demo on synthetic data (the core, runnable demonstration).

Runs the FULL offline pipeline on a TRACTABLE synthetic dataset (modest span / lookback so
it finishes in minutes on CPU and fits in a few GB) and SAVES ARTIFACTS:

    data/raw/synthetic/      synthetic GOES + OMNI CDFs + merged parquet (round-trip checked)
    data/processed/          grid_5min.parquet + windows.npz
    models/                  trained baseline models (+ small TFT) + scalers
    reports/metrics.csv      models x horizons, full metric suite (machine-readable)
    reports/metrics.md       the same as a human-readable markdown table
    reports/*.png            forecast-vs-actual (P10-P90 band), reliability, skill-vs-horizon

Stages (each reuses the implemented modules; nothing is re-implemented here):
  1. synthetic.generate           -> CDF + parquet
  2. pipeline.assert_cdf_roundtrip -> CDF read-back matches parquet
  3. pipeline.read_raw_merged      -> MERGED frame (via cdf_reader)
  4. pipeline.preprocess_frame     -> despike / impute / log / validate
  5. pipeline.build_feature_windows-> features + leakage-free windows (demo lookback/stride)
  6. train all baselines + a small TFT (chronological split + train-only scaler)
  7. evaluate.compare_models       -> per-horizon metrics
  8. write reports/metrics.{csv,md} + the three plots

The full-scale 11-year / L=1152 production config is documented in config/default.yaml and
ARCHITECTURE.md but is NOT run here (it does not fit on CPU/RAM); this demo overrides the
span and lookback to stay tractable while keeping the 3 horizons (nowcast 8 / 6h 72 / 12h 144).

Usage::

    python3 scripts/run_demo.py [--years 1.5] [--lookback 288] [--stride 4]
                                [--max-windows 4000] [--no-tft] [--tft-epochs 3]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ps14 import evaluate as ev
from ps14 import pipeline
from ps14 import train as tr
from ps14.config import load_config
from ps14.datasets import windowing
from ps14.io import synthetic
from ps14.io.synthetic import SyntheticParams
from ps14.utils.logging import get_logger

logger = get_logger("ps14.demo")

# Models scored in the demo (baselines always; TFT optionally).
_BASELINES = ["persistence", "climatology", "lightgbm", "refm"]

# The headline metrics surfaced in the markdown table (full set still goes to the CSV).
_HEADLINE_COLUMNS = [
    "rmse",
    "mae",
    "pe",
    "skill_vs_persistence",
    "skill_vs_climatology",
    "r2",
    "lc",
    "bias",
    "uncertainty_factor",
    "pod",
    "far",
    "csi",
    "hss",
    "tss",
    "roc_auc",
    "brier",
    "bss",
    "crps",
    "picp",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--years", type=float, default=1.5, help="Synthetic span in years.")
    p.add_argument("--start", default="2014-01-01", help="Synthetic start date (UTC).")
    p.add_argument("--lookback", type=int, default=288, help="Demo encoder length L (1 day=288).")
    p.add_argument("--stride", type=int, default=4, help="Keep every Nth window (cap memory).")
    p.add_argument(
        "--max-windows",
        dest="max_windows",
        type=int,
        default=6000,
        help="Hard cap on windows (keeps most recent).",
    )
    p.add_argument("--seed", type=int, default=1993)
    p.add_argument("--no-tft", action="store_true", help="Skip the small TFT (baselines only).")
    p.add_argument("--tft-epochs", type=int, default=3, help="TFT training epochs (CPU).")
    return p.parse_args()


def _fmt(v: object) -> str:
    """Format a metric value for the markdown table."""
    if isinstance(v, float):
        if np.isnan(v):
            return "nan"
        return f"{v:.4g}"
    return str(v)


def write_markdown_table(comparison: dict, path: Path, *, horizon_names: list[str]) -> None:
    """Write a models x horizons markdown table of the headline metrics to ``path``."""
    long = ev.comparison_table(comparison).reset_index()
    cols = [c for c in _HEADLINE_COLUMNS if c in long.columns]
    lines: list[str] = []
    lines.append("# PS-14 demo metrics (synthetic data)\n")
    lines.append(
        "Per-horizon regression + event (>=1000 pfu) + probabilistic metrics on the "
        "chronological held-out TEST split. Regression metrics are in log10(flux) space. "
        "`pe` = Prediction Efficiency = 1 - MSE/Var(obs); `skill_vs_*` = 1 - MSE_model/MSE_ref "
        "(must be > 0 to beat the reference). Event metrics use the 1000 pfu threshold.\n"
    )
    header = "| model | horizon | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (2 + len(cols))
    lines.append(header)
    lines.append(sep)
    for h in horizon_names:  # group rows by horizon for readability
        for _, row in long[long["horizon"] == h].iterrows():
            vals = " | ".join(_fmt(row[c]) for c in cols)
            lines.append(f"| {row['model']} | {row['horizon']} | {vals} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote markdown metrics table -> %s", path)


def main() -> int:
    args = parse_args()
    t0 = time.time()
    cfg = load_config(args.config)
    reports_dir = Path(cfg.paths.reports)
    reports_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(cfg.paths.data_raw) / "synthetic"

    # ---- 1. synthetic data (CDF + parquet) -------------------------------------------
    logger.info("[1/8] Generating ~%.2f yr synthetic dataset (seed=%d)", args.years, args.seed)
    params = SyntheticParams(
        start=args.start,
        end=(pd.Timestamp(args.start) + pd.Timedelta(days=365.25 * args.years)).isoformat(),
        cadence="5min",
        seed=args.seed,
    )
    synthetic.generate(params, out_dir=raw_dir)

    # ---- 2. CDF round-trip ------------------------------------------------------------
    logger.info("[2/8] Verifying CDF round-trip (cdf_reader vs parquet)")
    dev = pipeline.assert_cdf_roundtrip(raw_dir)
    logger.info(
        "   round-trip OK on %d channels (max abs dev %.2g)",
        len(dev),
        max(dev.values(), default=0.0),
    )

    # ---- 3-4. ingest + preprocess -----------------------------------------------------
    logger.info("[3/8] Reading MERGED frame via cdf_reader")
    raw = pipeline.read_raw_merged(raw_dir)
    logger.info("[4/8] Preprocessing (despike / impute / log / validate)")
    merged = pipeline.preprocess_frame(raw, cfg)
    pipeline.save_merged(merged, cfg)
    exceed_frac = float((merged["flux_e2"].dropna() >= cfg.thresholds.harsh_pfu).mean())
    logger.info("   merged rows=%d, >=1000 pfu fraction=%.3f", len(merged), exceed_frac)

    # ---- 5. features + windows --------------------------------------------------------
    logger.info(
        "[5/8] Building features + supervised windows (L=%d, stride=%d)", args.lookback, args.stride
    )
    wt = pipeline.build_feature_windows(
        merged, cfg, lookback=args.lookback, stride=args.stride, max_windows=args.max_windows
    )
    windowing.save_windows(wt, Path(cfg.paths.data_processed) / "windows.npz")
    logger.info(
        "   windows: X=%s y=%s y_exceed.mean=%.3f",
        wt.X.shape,
        wt.y.shape,
        float(wt.y_exceed.mean()),
    )

    # Embargo is expressed in WINDOW-index units. The contract embargo is `lookback + max(H)`
    # *raw 5-min steps*; because the demo subsamples windows by `stride`, consecutive kept
    # windows are `stride` raw-steps apart, so the equivalent window-unit embargo is
    # ceil((lookback + 144) / stride). This still purges >= lookback+horizon of real time
    # between splits (no leakage) while leaving non-empty val/test on the tractable demo set.
    embargo_windows = int(np.ceil((args.lookback + 144) / max(1, args.stride)))
    cfg.split.embargo_steps = embargo_windows
    split_kwargs = dict(train=cfg.split.train, val=cfg.split.val, embargo_steps=embargo_windows)
    logger.info(
        "   chronological split embargo = %d windows (stride=%d)", embargo_windows, args.stride
    )

    # ---- 6. train models --------------------------------------------------------------
    models: dict[str, object] = {}
    logger.info("[6/8] Training baselines: %s", _BASELINES)
    for name in _BASELINES:
        ts = time.time()
        models[name] = tr.train(cfg, model_name=name, windows=wt, save=True)
        logger.info("   trained %-12s in %.1fs", name, time.time() - ts)

    if not args.no_tft:
        logger.info("   Training small TFT (%d epochs, CPU, long-dataframe path)", args.tft_epochs)
        try:
            cfg.model.name = "tft"
            cfg.model.decoder_steps = 144
            cfg.model.params = {
                "hidden_size": 24,
                "lstm_layers": 1,
                "attention_head_size": 2,
                "hidden_continuous_size": 12,
                "dropout": 0.1,
                "max_epochs": args.tft_epochs,
                "batch_size": 128,
                # A higher LR is needed for the small TFT to actually converge within a few
                # CPU epochs (lr=1e-3 barely moves it off init -> near-constant predictions).
                "learning_rate": 5e-3,
                "gradient_clip_val": 0.1,
                "accelerator": "cpu",
            }
            ts = time.time()
            models["tft"] = tr.train(cfg, model_name="tft", windows=wt, save=True)
            logger.info("   trained TFT in %.1fs", time.time() - ts)
        except Exception as exc:  # keep the demo running on baselines if TFT fails
            logger.warning("   TFT training failed (%s); continuing with baselines.", exc)

    # ---- 7. evaluate ------------------------------------------------------------------
    logger.info("[7/8] Evaluating all models on the TEST split (per horizon)")
    # Skill references: persistence (auto) + a fitted climatology for the long horizons.
    references = {"climatology": models["climatology"]}
    comparison = ev.compare_models(
        models, wt, split="test", references=references, split_kwargs=split_kwargs
    )

    # ---- 8. reports + plots -----------------------------------------------------------
    logger.info("[8/8] Writing reports + plots -> %s", reports_dir)
    long_table = ev.comparison_table(comparison)
    csv_path = reports_dir / "metrics.csv"
    long_table.to_csv(csv_path)
    logger.info("   wrote %s", csv_path)
    write_markdown_table(comparison, reports_dir / "metrics.md", horizon_names=wt.horizon_names)

    # Per-model JSON/CSV reports (CONTRACTS.md §8).
    for name, res in comparison.items():
        ev.write_report(res, reports_dir / f"metrics_{name}.json")

    # Plots: showcase the best-performing model (highest mean PE across horizons on TEST).
    def _mean_pe(per_h: dict) -> float:
        vals = [per_h[h].get("pe", float("nan")) for h in wt.horizon_names]
        finite = [v for v in vals if not np.isnan(v)]
        return float(np.mean(finite)) if finite else float("-inf")

    plot_model_name = max(comparison, key=lambda n: _mean_pe(comparison[n]))
    plot_model = models[plot_model_name]
    logger.info("   best model by mean PE = %s (plots use it)", plot_model_name)
    fig1 = ev.plot_forecast_vs_actual(
        plot_model, wt, split="test", reports_dir=reports_dir, split_kwargs=split_kwargs
    )
    fig2 = ev.plot_reliability(
        plot_model, wt, split="test", reports_dir=reports_dir, split_kwargs=split_kwargs
    )
    fig3 = ev.plot_skill_vs_horizon(
        comparison[plot_model_name], reports_dir=reports_dir, model_name=plot_model_name
    )
    logger.info("   plots: %s", [str(p) for p in (fig1, fig2, fig3)])

    # ---- console summary --------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"PS-14 DEMO COMPLETE in {time.time() - t0:.1f}s  (plots from '{plot_model_name}')")
    print(f"  synthetic span : {args.years} yr @ 5-min   windows: {wt.X.shape[0]}")
    print(f"  >=1000 pfu frac: {exceed_frac:.3f}")
    print("=" * 78)
    _print_console_table(comparison, wt.horizon_names)
    print(f"\nArtifacts: {csv_path}, {reports_dir / 'metrics.md'}, and reports/*.png")
    return 0


def _print_console_table(comparison: dict, horizon_names: list[str]) -> None:
    """Print a compact PE / RMSE / skill / HSS / ROC-AUC table to stdout."""
    keys = ["rmse", "pe", "skill_vs_persistence", "hss", "roc_auc"]
    header = f"{'model':<12} {'horizon':<8} " + " ".join(f"{k:>20}" for k in keys)
    print(header)
    print("-" * len(header))
    for model_name, per_h in comparison.items():
        for h in horizon_names:
            row = per_h.get(h, {})
            cells = " ".join(f"{_fmt(row.get(k, float('nan'))):>20}" for k in keys)
            print(f"{model_name:<12} {h:<8} {cells}")


if __name__ == "__main__":
    raise SystemExit(main())
