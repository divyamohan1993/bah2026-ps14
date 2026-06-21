"""Command-line interface dispatching the pipeline stages (mirrors the Makefile).

Uses argparse (always available) so importing the package never requires optional deps.
Subcommands: ``synth-data``, ``fetch-data``, ``preprocess``, ``features``, ``baseline``,
``train``, ``evaluate``, ``serve``, ``dashboard``. Each loads the config and delegates to
the relevant module.

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
    from ps14.io import synthetic
    from ps14.io.synthetic import SyntheticParams

    out_dir = cfg.paths.data_raw / "synthetic"
    logger.info("Generating synthetic data under %s", out_dir)
    params = SyntheticParams(**cfg.synthetic.model_dump())
    artifacts = synthetic.generate(params, out_dir=out_dir)
    for name, path in artifacts.items():
        logger.info("  %s -> %s", name, path)
    return 0


def cmd_fetch_data(args: argparse.Namespace) -> int:
    """Download real OMNI solar-wind drivers (needs the [data] extra + network)."""
    cfg = load_config(args.config)
    from ps14.io import cdaweb

    start = cfg.data.time_range.get("start", "2017-09-01T00:00:00Z")
    end = cfg.data.time_range.get("end", "2017-09-15T00:00:00Z")
    dataset_id = cfg.data.sources.get("omni_drivers", cdaweb.OMNI_DRIVERS_DATASET)
    logger.info("Fetching OMNI drivers %s for %s..%s", dataset_id, start, end)
    try:
        df = cdaweb.fetch_cdaweb(
            dataset_id, cdaweb.OMNI_DRIVER_VARIABLES, start, end, cache_dir=cfg.paths.data_raw
        )
        logger.info("  %s: %d rows", dataset_id, len(df))
    except Exception as exc:  # pragma: no cover - network/dep dependent
        logger.warning("  %s fetch failed: %s", dataset_id, exc)
        return 1
    return 0


def cmd_preprocess(args: argparse.Namespace) -> int:
    """Clean -> resample -> align -> transform into the merged 5-min frame."""
    cfg = load_config(args.config)
    from ps14 import pipeline

    logger.info("Preprocessing into %s", cfg.paths.data_processed)
    raw = pipeline.read_raw_merged(cfg.paths.data_raw / "synthetic")
    merged = pipeline.preprocess_frame(raw, cfg)
    pipeline.save_merged(merged, cfg)
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    """Build the feature matrix + supervised-window tensors."""
    cfg = load_config(args.config)
    from ps14 import pipeline
    from ps14.datasets import windowing

    logger.info("Building features + windows (lookback=%d)", cfg.features.lookback_steps)
    merged = pipeline.load_merged(cfg)
    wt = pipeline.build_feature_windows(
        merged,
        cfg,
        lookback=args.lookback,
        stride=args.stride,
        max_windows=args.max_windows,
    )
    out = cfg.paths.data_processed / "windows.npz"
    windowing.save_windows(wt, out)
    logger.info("Saved %d windows -> %s", wt.X.shape[0], out)
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    """Fit + score the baseline tier (persistence/climatology/lightgbm/refm)."""
    cfg = load_config(args.config)
    from ps14 import evaluate as ev
    from ps14 import train as tr
    from ps14.datasets import windowing

    logger.info("Running baseline tier (windows under %s)", cfg.paths.data_processed)
    wt = windowing.load_windows(cfg.paths.data_processed / "windows.npz")
    split_kwargs = dict(
        train=cfg.split.train, val=cfg.split.val, embargo_steps=cfg.split.embargo_steps
    )
    for name in ("persistence", "climatology", "lightgbm", "refm"):
        model = tr.train(cfg, model_name=name, windows=wt, save=True)
        results = ev.evaluate(model, wt, split="test", split_kwargs=split_kwargs)
        ev.write_report(results, cfg.paths.reports / f"metrics_{name}.json")
        logger.info("  %s scored; report -> %s", name, cfg.paths.reports / f"metrics_{name}.json")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train the configured primary model (TFT dual-head by default)."""
    cfg = load_config(args.config)
    from ps14 import train as tr

    logger.info("Training model: %s", cfg.model.name)
    tr.train(cfg, model_name=args.model)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Walk-forward evaluation + per-horizon metric report."""
    cfg = load_config(args.config)
    from ps14 import evaluate as ev
    from ps14 import train as tr
    from ps14.datasets import windowing

    name = (args.model or cfg.model.name).lower()
    logger.info("Evaluating model: %s", name)
    wt = windowing.load_windows(cfg.paths.data_processed / "windows.npz")
    # Re-fit on the in-memory windows (a load path per model is also available via .load()).
    model = tr.train(cfg, model_name=name, windows=wt, save=False)
    split_kwargs = dict(
        train=cfg.split.train, val=cfg.split.val, embargo_steps=cfg.split.embargo_steps
    )
    out = ev.render_report(
        model, wt, split="test", reports_dir=cfg.paths.reports, split_kwargs=split_kwargs
    )
    logger.info("Report -> %s; figures -> %s", out["report_json"], list(out["figures"].values()))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Launch the FastAPI + APScheduler real-time service."""
    cfg = load_config(args.config)
    import uvicorn

    from ps14.serve.api import create_app

    logger.info(
        "Serving on %s:%d (source=%s)", cfg.serving.host, cfg.serving.port, cfg.serving.source
    )
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.serving.host, port=cfg.serving.port)
    return 0


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
        if name == "features":
            p.add_argument(
                "--lookback",
                type=int,
                default=None,
                help="Encoder length override L (default from config.features).",
            )
            p.add_argument(
                "--stride",
                type=int,
                default=1,
                help="Keep every Nth window (subsample in time to cap memory).",
            )
            p.add_argument(
                "--max-windows",
                dest="max_windows",
                type=int,
                default=None,
                help="Hard cap on number of windows (keeps most recent).",
            )
        if name in {"train", "evaluate"}:
            p.add_argument(
                "--model", default=None, help="Model name override (default config.model.name)."
            )
        p.set_defaults(func=func)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``ps14`` console script and ``python -m ps14.cli``."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
