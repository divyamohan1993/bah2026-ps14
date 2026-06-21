# PS-14 · Forecasting the >2 MeV Electron Radiation Environment at GEO

**BAH-2026 Problem Statement 14 — "Forecasting Energetic Particle Radiation Environment for
ISRO's Geostationary Satellites."**

An end-to-end AI/ML system that **reads, processes, and visualizes CDF-format space-weather data**
and **forecasts the >2 MeV integral electron flux at geostationary orbit (GEO)** at three horizons
— **30–45 min (nowcast), 6 h, and 12 h** — plus a **calibrated probability of crossing the NOAA
"harsh" alert threshold of 1000 pfu** at each horizon.

> The full design is in [`ARCHITECTURE.md`](ARCHITECTURE.md); the binding inter-module data
> contracts are in [`CONTRACTS.md`](CONTRACTS.md); the cited research is under `docs/research/`.

---

## Why this matters

>2 MeV "killer" electrons penetrate spacecraft shielding and bury charge in dielectrics; a
resulting electrostatic discharge can corrupt or destroy satellite electronics (Galaxy 15,
Telstar 401). The flux varies by **>3 orders of magnitude**, but **solar-wind speed drives it with
a ~1–2 day lag**, and **L1 monitors give a ~20–90 min lead** — so a multi-horizon forecast is both
feasible and operationally valuable for scheduling load-shedding and deferring manoeuvres.

## Architecture at a glance

```text
 DATA (CDF: GOES >2 MeV target + OMNI/Wind drivers; SYNTHETIC generator for offline)
   │  cdflib read · ISTP fill/valid mask · TT2000 leap-aware
   ▼
 CLEAN (Hampel/MAD despike · gap detect: short→interp, long→NaN)
   ▼
 RESAMPLE 5-min → ALIGN L1→GEO → TRANSFORM log10 (scalers fit on TRAIN only)
   ▼
 FEATURES (lags · rolling · coupling fns · sin/cos MLT/season known-future)
   ▼
 WINDOWS  X:[N,L,F]  y:[N,3]  y_exceed:[N,3]   (chrono split + purge/embargo)
   ▼
 MODEL  dual-head TFT (P10/P50/P90 + focal-BCE exceedance)  ·  baselines  ·  backups
   ▼
 EVALUATE per horizon (PE/RMSE · POD/FAR/HSS/TSS/ROC-AUC · pinball/CRPS/PICP)
   ▼
 SERVE  FastAPI + APScheduler 60 s  ·  O(1) ring-buffer/Welford/deque  ·  ONNX + cache
   ▼
 DASHBOARD  Streamlit: live flux + solar wind, forecast ± bands, alert status
```

## Quickstart (fully offline — no network, no real data)

```bash
# 1. Install (core only is enough for synth-data → baseline; add extras as needed)
python -m pip install -e ".[dl,serve,viz,dev]"

# 2. Generate physically-plausible synthetic data, written to CDF (exercises the real reader)
make synth-data

# 3. Run the offline pipeline: clean → resample → align → transform → features → windows
make preprocess
make features

# 4. Fit + score the baseline tier (persistence / climatology / LightGBM / REFM)
make baseline

# 5. Train the primary model (TFT dual-head) and evaluate per horizon
make train
make evaluate

# 6. Launch the real-time service and the dashboard (synthetic replay by default)
make serve        # FastAPI on :8000  → GET /forecast /latest /health, WS /ws
make dashboard    # Streamlit live UI
```

Everything above runs **without internet**. To use real data instead, install the `[data]` extra
and run `make fetch-data` (GOES SEISS/EPEAD, `OMNI_HRO_1MIN`, Wind) — see `config/default.yaml`.

## Repository layout

```
ARCHITECTURE.md   Master design (problem, diagram, data, features, model, serving, eval, roadmap)
CONTRACTS.md      Binding schemas + tensor shapes + model interface for parallel builders
config/           default.yaml + per-model YAMLs (tft / nhits / baseline)
src/ps14/         io · preprocess · features · datasets · models · serve · dashboard · cli
tests/            O(1) primitives, metrics, schema (real tests) + stubs for the rest
docs/research/    R1 domain · R2 catalog · R3 ML SOTA · R4 platform · R5 CDF engineering
data/ models/ notebooks/   (git-ignored content; .gitkeep retained)
```

## How it maps to PS-14

| PS-14 objective / evaluation parameter | Where it lives |
|---|---|
| Read/process/visualize CDF flux + solar-wind data | `io/cdf_reader.py`, `preprocess/*`, `dashboard/app.py` |
| Identify important solar-wind variables | feature spec (ARCHITECTURE (d)); TFT variable-selection + LightGBM importances |
| Preprocess: despike / interpolate / handle missing | `preprocess/clean.py` (Hampel/MAD, gap-aware) |
| Multi-step forecast using time history of inputs+outputs | dual-head TFT, direct multi-horizon (`models/tft.py`) |
| 30–45 min + 6 h + 12 h forecasts | `constants.HORIZON_STEPS = {nowcast:8, 6h:72, 12h:144}` |
| Accuracy of predicted fluxes | `metrics.py` + `evaluate.py` (PE/RMSE/skill + event + probabilistic) |
| Harsh-radiation alerting | exceedance head + 1000 pfu threshold + dashboard alert status |

## Models

- **Primary:** dual-head **Temporal Fusion Transformer** — quantile regression (P10/P50/P90) +
  focal-BCE exceedance head, direct multi-horizon, native known-future covariates and
  interpretable variable selection (pytorch-forecasting + Lightning).
- **Baselines (always reported):** persistence, diurnal climatology, LightGBM (direct), REFM-style
  linear filter.
- **Backups:** N-HiTS / TiDE; fine-tuned foundation model (Chronos-Bolt / TimesFM).

## Status

Architecture + scaffold. The O(1) online primitives (`features/online.py`), metrics
(`metrics.py`), and schema (`datasets/schema.py`) are implemented and tested; the remaining
modules carry precise signatures + contract-encoding docstrings and are being filled in per the
roadmap in `ARCHITECTURE.md` (k).

## License

MIT.
