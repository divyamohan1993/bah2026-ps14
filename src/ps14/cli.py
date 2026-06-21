"""Command-line interface dispatching the pipeline stages (mirrors the Makefile).

Uses argparse (always available) so importing the package never requires optional deps.
Subcommands: ``synth-data``, ``fetch-data``, ``preprocess``, ``features``, ``baseline``,
``train``, ``evaluate``, ``serve``, ``dashboard``. Each loads the config and delegates to
the relevant module; bodies are TODO-marked until those modules are implemented.

Run via ``python -m ps14.cli <command>`` or the installed ``ps14`` console script.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ps14.config import DEFAULT_CONFIG_PATH, load_config
from ps14.utils.logging import get_logger

logger = get_logger("ps14.cli")


def _add_config_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the YAML config (default: config/default.yaml).",
    )


def cmd_synth_data(args: argparse.Namespace) -> int:
    """Generate physically-plausible synthetic CDFs + parquet (offline)."""
    cfg = load_config(args.config)
    logger.info("Generating synthetic data under %s", cfg.paths.data_raw)
    raise NotImplementedError(
        "TODO: from ps14.io import synthetic; "
        "synthetic.generate(SyntheticParams(**cfg.synthetic.model_dump()))."
    )


def cmd_fetch_data(args: argparse.Namespace) -> int:
    """Download real GOES/OMNI/Wind data (needs the [data] extra + network)."""
    cfg = load_config(args.config)
    logger.info("Fetching real datasets: %s", cfg.data.sources)
    raise NotImplementedError("TODO: from ps14.io import cdaweb; fetch each configured dataset_id.")


def cmd_preprocess(args: argparse.Namespace) -> int:
    """Clean -> resample -> align -> transform into the merged 5-min frame."""
    cfg = load_config(args.config)
    logger.info("Preprocessing into %s", cfg.paths.data_processed)
    raise NotImplementedError(
        "TODO: read raw (cdf_reader); clean (despike/gaps); resample; align (merge_geo_l1); "
        "transform (log10_floor); validate_merged; write grid_5min.parquet."
    )


def cmd_features(args: argparse.Namespace) -> int:
    """Build the feature matrix + supervised-window tensors."""
    cfg = load_config(args.config)
    logger.info("Building features + windows (lookback=%d)", cfg.features.lookback_steps)
    raise NotImplementedError(
        "TODO: offline.build_feature_matrix; windowing.make_supervised; windowing.save_windows."
    )


def cmd_baseline(args: argparse.Namespace) -> int:
    """Fit + score the baseline tier (persistence/climatology/lightgbm/refm)."""
    cfg = load_config(args.config)
    logger.info("Running baseline tier (windows under %s)", cfg.paths.data_processed)
    raise NotImplementedError(
        "TODO: load windows; for each baseline: fit on train, "
        "evaluate.evaluate_model, write report."
    )


def cmd_train(args: argparse.Namespace) -> int:
    """Train the configured primary model (TFT dual-head by default)."""
    cfg = load_config(args.config)
    logger.info("Training model: %s", cfg.model.name)
    raise NotImplementedError("TODO: from ps14.train import train; train(cfg).")


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Walk-forward evaluation + per-horizon metric report."""
    cfg = load_config(args.config)
    logger.info("Evaluating model: %s", cfg.model.name)
    raise NotImplementedError(
        "TODO: load model + test windows; evaluate.evaluate_model; write_report."
    )


def cmd_serve(args: argparse.Namespace) -> int:
    """Launch the FastAPI + APScheduler real-time service."""
    cfg = load_config(args.config)
    logger.info(
        "Serving on %s:%d (source=%s)", cfg.serving.host, cfg.serving.port, cfg.serving.source
    )
    raise NotImplementedError("TODO: from ps14.serve.api import run; run(cfg).")


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Hint to launch the Streamlit dashboard (Streamlit runs the script directly)."""
    logger.info("Launch the dashboard with: streamlit run src/ps14/dashboard/app.py")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(prog="ps14", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    commands = {
        "synth-data": (cmd_synth_data, "Generate synthetic CDFs + parquet (offline)."),
        "fetch-data": (cmd_fetch_data, "Download real GOES/OMNI/Wind data."),
        "preprocess": (cmd_preprocess, "Clean/resample/align/transform -> merged frame."),
        "features": (cmd_features, "Build feature matrix + window tensors."),
        "baseline": (cmd_baseline, "Fit + score the baseline tier."),
        "train": (cmd_train, "Train the primary model."),
        "evaluate": (cmd_evaluate, "Walk-forward evaluation + report."),
        "serve": (cmd_serve, "Launch the real-time service."),
        "dashboard": (cmd_dashboard, "Launch the Streamlit dashboard."),
    }
    for name, (func, help_text) in commands.items():
        p = sub.add_parser(name, help=help_text)
        _add_config_arg(p)
        p.set_defaults(func=func)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``ps14`` console script and ``python -m ps14.cli``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
