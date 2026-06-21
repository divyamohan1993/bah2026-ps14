#!/usr/bin/env python3
"""Real-data (CDAWeb) validation demo — Milestone 3 (best-effort, network-dependent).

Attempts a SMALL real fetch via :mod:`ps14.io.cdaweb` (cdasws):

* OMNI_HRO_1MIN solar-wind drivers for a ~3-month window, and
* a real GOES >2 MeV integral electron flux dataset (tries the GOES-R SEISS MPS-HI IDs from
  docs/research/02, falling back across probes; if all 404 we proceed with OMNI alone).

On a successful electron fetch it assembles a small MERGED-shaped frame, builds windows, fits
a Persistence baseline, evaluates it per horizon, and writes ``reports/real_data_demo.md`` with
the dataset IDs used, the date range, row counts, and baseline metrics.

This is intentionally fault-tolerant: ANY fetch/parse failure is caught and documented; the
synthetic pipeline remains the guaranteed path. Exit code is 0 whether or not the network
fetch succeeds (the report records what happened).

Usage::  python3 scripts/real_data_demo.py [--start 2017-09-01] [--end 2017-12-01]
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from ps14.io import cdaweb

# Candidate GOES >2 MeV integral electron datasets to try, in order (R2 §18-19).
_GOES_CANDIDATES: list[tuple[str, list[str]]] = [
    # GOES-R SEISS MPS-HI integral >2 MeV electron flux. Variable names vary by processing;
    # we try a few common candidates and keep whichever the server returns.
    ("DN_SEIS-L2-MPSH_G18", ["AvgIntElectronFlux", "IntElectronFlux", "flux"]),
    ("DN_SEIS-L2-MPSH_G16", ["AvgIntElectronFlux", "IntElectronFlux", "flux"]),
    ("DN_SEIS-L2-MPSH_G17", ["AvgIntElectronFlux", "IntElectronFlux", "flux"]),
    # Legacy GOES-13/15 EPEAD >2 MeV electron flux (E2).
    ("GOES13_EPEAD-SCIENCE-ELECTRONS-E13EW_1MIN", ["E2W_UNCOR_FLUX", "E2W_COR_FLUX"]),
    ("GOES15_EPS-EPEAD-ELECTRONS-E13EW_1MIN", ["E2W_UNCOR_FLUX", "E2W_COR_FLUX"]),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2017-09-01")
    p.add_argument("--end", default="2017-12-01")
    p.add_argument("--reports-dir", default="reports")
    return p.parse_args()


def _iso(s: str) -> str:
    return pd.Timestamp(s).strftime("%Y-%m-%dT%H:%M:%SZ")


def try_omni(start: str, end: str, notes: list[str]) -> pd.DataFrame | None:
    """Fetch OMNI_HRO_1MIN drivers; return a frame or None (logging to ``notes``)."""
    try:
        df = cdaweb.fetch_cdaweb(
            cdaweb.OMNI_DRIVERS_DATASET,
            cdaweb.OMNI_DRIVER_VARIABLES,
            _iso(start),
            _iso(end),
            cache_dir="data/raw/real",
        )
        notes.append(
            f"OMNI fetch OK: {cdaweb.OMNI_DRIVERS_DATASET} -> {len(df)} rows, "
            f"{df.shape[1]} variables ({list(df.columns)})."
        )
        return df
    except Exception as exc:  # noqa: BLE001 - document and continue
        notes.append(
            f"OMNI fetch FAILED ({cdaweb.OMNI_DRIVERS_DATASET}): {type(exc).__name__}: {exc}"
        )
        return None


def try_goes(start: str, end: str, notes: list[str]) -> tuple[str, pd.Series] | None:
    """Try each GOES electron-flux candidate; return (dataset_id, series) or None."""
    for dataset_id, var_candidates in _GOES_CANDIDATES:
        for var in var_candidates:
            try:
                df = cdaweb.fetch_cdaweb(
                    dataset_id, [var], _iso(start), _iso(end), cache_dir="data/raw/real"
                )
            except Exception as exc:  # noqa: BLE001 - try the next candidate
                notes.append(f"GOES try {dataset_id}:{var} FAILED: {type(exc).__name__}: {exc}")
                continue
            if df is not None and not df.empty and df.shape[1] >= 1:
                col = df.columns[0]
                series = df[col].dropna()
                if series.size > 0:
                    notes.append(
                        f"GOES fetch OK: {dataset_id}:{var} -> {series.size} non-NaN rows."
                    )
                    return dataset_id, series.rename("flux_e2")
            notes.append(f"GOES try {dataset_id}:{var}: empty/no data.")
    return None


def build_real_merged(omni: pd.DataFrame, goes: pd.Series, cfg) -> pd.DataFrame:
    """Assemble a small MERGED-schema frame from real OMNI + GOES on a 5-min grid."""
    from ps14.datasets import schema
    from ps14.features import offline
    from ps14.preprocess.resample import resample_uniform
    from ps14.preprocess.transform import log10_floor

    # Map OMNI variable names -> canonical columns.
    rename = {
        "flow_speed": "vsw",
        "proton_density": "density",
        "Pressure": "pdyn",
        "BZ_GSM": "bz_gsm",
        "F": "bt",
        "AE_INDEX": "ae",
        "AL_INDEX": "al",
        "SYM_H": "sym_h",
        "KP": "kp",
        "F10_INDEX": "f107",
    }
    o = omni.rename(columns={k: v for k, v in rename.items() if k in omni.columns})
    merged = o.join(goes.to_frame(), how="outer").sort_index()
    merged = resample_uniform(merged, cadence="5min", agg="mean")
    if "row_missing" in merged.columns:
        merged = merged.drop(columns=["row_missing"])

    # OMNI KP is stored x10; rescale to 0-9 if it looks scaled.
    if "kp" in merged and merged["kp"].dropna().max() > 9:
        merged["kp"] = merged["kp"] / 10.0
    # Fill any missing canonical columns so feature-building has the full schema. Channels
    # absent from OMNI_HRO_1MIN (kp, f107) are filled with quiet-time constants so they don't
    # poison every window with NaN (they are weak features; this is a best-effort real demo).
    _constants = {"kp": 2.0, "f107": 100.0}
    for col in schema.MERGED_REQUIRED:
        if col not in merged.columns or merged[col].isna().all():
            merged[col] = _constants.get(col, np.nan)
    # Seed flux unknown from this product -> mirror the >2 MeV channel as a stand-in.
    if "flux_seed" not in merged or merged["flux_seed"].isna().all():
        merged["flux_seed"] = merged["flux_e2"]
    merged["pdyn"] = offline.dynamic_pressure(
        merged["density"].to_numpy("float64"), merged["vsw"].to_numpy("float64")
    )
    floor = cfg.preprocess.log_floor_pfu
    merged["log_flux_e2"] = log10_floor(merged["flux_e2"], floor=floor)
    merged["log_flux_seed"] = log10_floor(merged["flux_seed"], floor=floor)
    merged["mlt"] = (merged.index.hour + merged.index.minute / 60.0).to_numpy("float64")
    merged["longitude"] = -75.0
    merged["sat_id"] = pd.Categorical(["GOES-REAL"] * len(merged))
    for col in schema.MERGED_REQUIRED:
        merged[col] = merged[col].astype("float64")
    merged.index.name = "time"
    return merged


def main() -> int:
    args = parse_args()
    from ps14.config import load_config

    cfg = load_config()
    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    notes.append(f"Run at {dt.datetime.now(dt.timezone.utc).isoformat()} UTC")
    notes.append(f"Requested window: {args.start} .. {args.end}")

    omni = try_omni(args.start, args.end, notes)
    goes = try_goes(args.start, args.end, notes)

    metrics_section = ""
    status = "PARTIAL"
    if omni is not None and goes is not None:
        try:
            from ps14 import evaluate as ev
            from ps14 import pipeline
            from ps14.models.baselines import Persistence

            dataset_id_goes, goes_series = goes
            # Real Sept-2017 data has ~20-26% gaps (storm-time instrument outages). Raise the
            # short-gap interpolation tolerance and shorten the lookback so enough complete
            # windows survive the drop-if-NaN rule (this is a best-effort real-data validation).
            cfg.preprocess.gaps.max_gap_steps = 288  # interpolate gaps up to ~1 day for the demo
            merged_raw = build_real_merged(omni, goes_series, cfg)
            merged = pipeline.preprocess_frame(merged_raw, cfg)
            wt = pipeline.build_feature_windows(
                merged, cfg, lookback=72, stride=2, max_windows=4000
            )
            if wt.X.shape[0] >= 30:
                emb = 60
                m = Persistence().fit(wt.X, wt.X_future, wt.y, wt.y_exceed)
                res = ev.evaluate(
                    m,
                    wt,
                    split="test",
                    split_kwargs=dict(train=0.7, val=0.15, embargo_steps=emb),
                )
                rows = [
                    "| horizon | rmse | mae | pe | skill_vs_persistence | roc_auc | hss |",
                    "|---|---|---|---|---|---|---|",
                ]
                for h in wt.horizon_names:
                    r = res[h]
                    rows.append(
                        f"| {h} | {r['rmse']:.4g} | {r['mae']:.4g} | {r['pe']:.4g} | "
                        f"{r['skill_vs_persistence']:.4g} | {r['roc_auc']:.4g} | {r['hss']:.4g} |"
                    )
                metrics_section = "\n".join(rows)
                notes.append(
                    f"Real-data windows: X={wt.X.shape}, merged rows={len(merged)}, "
                    f"GOES dataset={dataset_id_goes}."
                )
                status = "SUCCESS"
            else:
                notes.append(f"Too few windows ({wt.X.shape[0]}) for evaluation after gaps.")
        except Exception as exc:  # noqa: BLE001 - document and continue
            notes.append(f"Real-data pipeline FAILED: {type(exc).__name__}: {exc}")
    else:
        notes.append("Skipping pipeline: need BOTH OMNI and a GOES electron dataset.")

    # Write the report.
    lines = [
        "# PS-14 real-data (CDAWeb) validation demo",
        "",
        f"**Status:** {status}",
        "",
        "## Fetch log",
        "",
        *[f"- {n}" for n in notes],
        "",
    ]
    if metrics_section:
        lines += [
            "## Persistence baseline metrics on REAL data (per horizon, log10 space)",
            "",
            metrics_section,
            "",
            "Regression metrics in log10(flux) space; event metrics at the 1000 pfu threshold.",
            "",
        ]
    else:
        lines += [
            "## Metrics",
            "",
            "No baseline metrics were produced (see fetch log). Synthetic remains the "
            "guaranteed end-to-end path (`scripts/run_demo.py`).",
            "",
        ]
    out = reports / "real_data_demo.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[{status}] wrote {out}")
    for n in notes:
        print("  -", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
