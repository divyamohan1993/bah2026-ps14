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

The single command below runs the **complete end-to-end pipeline** on a tractable synthetic
dataset (finishes in a few minutes on CPU) and writes all metrics, tables, plots, and trained
models into `reports/` and `models/`.

```bash
# 1. Install the package (editable) with the optional extras the demo uses.
python3 -m pip install -e ".[dl,serve,viz,dev]"

# 2. Run the FULL end-to-end demo: synthetic CDF generation → CDF round-trip check →
#    preprocess → features/windows → fit all baselines + a small TFT → per-horizon
#    evaluation → write reports/metrics.{csv,md} + plots. (≈ a few minutes on CPU.)
python3 scripts/run_demo.py
#    Equivalent Make target (same defaults):  make demo
#    Faster baseline-only run:                python3 scripts/run_demo.py --no-tft
```

Outputs after the demo:

- `reports/metrics.md` / `reports/metrics.csv` — per-horizon metrics for all 5 models.
- `reports/forecast_vs_actual_lightgbm.png`, `reports/reliability_lightgbm.png`,
  `reports/skill_vs_horizon_lightgbm.png` — diagnostic plots for the best model.
- `models/{persistence.npz, climatology.json, refm.joblib, lightgbm.joblib, tft.ckpt, tft.joblib}`.
- `data/raw/synthetic/{synthetic_goes.cdf, synthetic_omni.cdf, merged.parquet}` and
  `data/processed/{grid_5min.parquet, windows.npz}`.

### Run the individual stages (instead of the one-shot demo)

Each stage is also a CLI subcommand / Make target (they share the same code paths):

```bash
make synth-data     # python3 -m ps14.cli synth-data   → synthetic GOES + OMNI CDFs + parquet
make preprocess     # python3 -m ps14.cli preprocess    → cleaned/aligned 5-min merged frame
make features       # python3 -m ps14.cli features       → feature matrix + supervised windows
make baseline       # python3 -m ps14.cli baseline       → fit + score persistence/clim/lgbm/refm
make train          # python3 -m ps14.cli train          → train the TFT dual-head model
make evaluate       # python3 -m ps14.cli evaluate       → per-horizon metric report + figures
python3 -m ps14.cli --help                               # all subcommands
```

### Launch the real-time service and dashboard

```bash
make serve          # FastAPI on :8000 (synthetic replay by default)
                    #   GET /health · GET /latest · GET /forecast · GET /forecast/{horizon}
                    #   POST /forecast (feature vector → multi-horizon P10/P50/P90 + P(exceed))
                    #   WS  /ws  (streaming updates)
make dashboard      # streamlit run src/ps14/dashboard/app.py  → live flux, bands, alert status
```

### Smoke-test the serving / ONNX / dashboard path (no long-running server)

```bash
python3 scripts/smoke_serving.py
#   [PASS] FastAPI /health + POST /forecast
#   [PASS] ONNX export + onnxruntime Predictor (+ O(1) forecast cache hit)
#   [PASS] ClimatologyLUT O(1) fallback
#   [PASS] Streamlit dashboard module imports
```

### Optional: validate on real CDAWeb data

```bash
python3 -m pip install -e ".[data]"   # adds cdasws/cdflib data extras
python3 scripts/real_data_demo.py     # fetches OMNI + GOES electron flux from NASA CDAWeb,
                                      # builds windows, scores the persistence baseline →
                                      # reports/real_data_demo.md  (needs network)
```

## Results (synthetic demo, chronological held-out TEST split)

Generated by `python3 scripts/run_demo.py`; full table in
[`reports/metrics.md`](reports/metrics.md). **Regression metrics are in `log10(flux)` space.**
`PE` (Prediction Efficiency) `= 1 − MSE/Var(obs)`; `skill_vs_persistence = 1 − MSE_model/MSE_persistence`
(must be `> 0` to beat persistence). Event metrics use the **1000 pfu** threshold.

**Best model — LightGBM (direct multi-horizon):**

| horizon | RMSE (log) | PE | skill_vs_persistence | corr (r²) | HSS (≥1000 pfu) | ROC-AUC |
|---|---|---|---|---|---|---|
| nowcast | 0.134 | 0.969 | +0.436 | 0.985 | 0.919 | 0.994 |
| 6 h     | 0.183 | 0.941 | +0.879 | 0.972 | 0.871 | 0.983 |
| 12 h    | 0.203 | 0.928 | +0.920 | 0.967 | 0.840 | 0.984 |

**Baselines / model comparison (PE per horizon, higher is better):**

| model | nowcast PE | 6 h PE | 12 h PE | nowcast ROC-AUC | 12 h ROC-AUC |
|---|---|---|---|---|---|
| **LightGBM** | **0.969** | **0.941** | **0.928** | **0.994** | **0.984** |
| TFT (demo, 8 epochs) | 0.945 | 0.496 | 0.143 | 0.992 | 0.697 |
| REFM (linear filter) | 0.946 | 0.610 | 0.542 | 0.990 | 0.815 |
| Persistence | 0.945 | 0.516 | 0.099 | 0.942 | 0.572 |
| Climatology | −0.212 | −0.139 | −0.248 | 0.614 | 0.461 |

**Takeaways (honest):**

- **LightGBM is the strongest model at every horizon** — it is the only model that beats
  persistence at the nowcast (skill +0.44) and stays high-skill out to 12 h (PE 0.93), with
  excellent event discrimination (ROC-AUC 0.98–0.99, HSS 0.84–0.92).
- Persistence is a very strong nowcast baseline (PE 0.945) but **collapses with lead time**
  (PE → 0.10 at 12 h), which is exactly why a driver-aware model is needed.
- **The TFT achieves positive skill at every horizon** even in this CPU demo (8 epochs on a
  ~1.5 yr subset): it ties persistence at the nowcast (PE 0.945) while substantially improving
  event discrimination there (HSS 0.90 vs 0.88, ROC-AUC 0.99 vs 0.94), edges out persistence at
  12 h (PE 0.14 vs 0.10, ROC-AUC 0.70 vs 0.57), and trails it only marginally at 6 h (PE 0.50 vs
  0.52) while still posting a much higher 6 h ROC-AUC (0.85 vs 0.72). It remains behind LightGBM
  at every horizon — honest for a small CPU run; closing that gap needs the full-scale config
  (longer history, more epochs, GPU) documented in `config/default.yaml` and `ARCHITECTURE.md`.

## How it maps to PS-14

| PS-14 objective / evaluation parameter | Where it lives | Status |
|---|---|---|
| Read/process/visualize CDF flux + solar-wind data | `io/cdf_reader.py`, `preprocess/*`, `dashboard/app.py` | ✅ CDF round-trip checked in demo |
| Identify important solar-wind variables | feature spec (ARCHITECTURE d); LightGBM importances / TFT variable selection | ✅ |
| Preprocess: despike / interpolate / handle missing | `preprocess/clean.py` (Hampel/MAD, gap-aware) | ✅ |
| Multi-step forecast from time history of inputs+outputs | direct multi-horizon LightGBM + dual-head TFT (`models/`) | ✅ |
| 30–45 min + 6 h + 12 h forecasts | `constants.HORIZON_STEPS = {nowcast:8, 6h:72, 12h:144}` | ✅ all 3 horizons scored |
| Accuracy of predicted fluxes | `metrics.py` + `evaluate.py` (PE/RMSE/skill + event + probabilistic) | ✅ see Results |
| Harsh-radiation alerting | exceedance head + 1000 pfu threshold + dashboard alert status | ✅ HSS/ROC-AUC reported |

## Repository layout

```
ARCHITECTURE.md   Master design (problem, diagram, data, features, model, serving, eval, roadmap)
CONTRACTS.md      Binding schemas + tensor shapes + model interface for parallel builders
config/           default.yaml + per-model YAMLs (tft / nhits / baseline)
src/ps14/         io · preprocess · features · datasets · models · serve · dashboard · cli
scripts/          run_demo.py (end-to-end) · real_data_demo.py (CDAWeb) · smoke_serving.py
tests/            O(1) primitives, metrics, schema, preprocess, baselines, serve, windowing
reports/          metrics.{csv,md} + per-model JSON/CSV + diagnostic plots (after `run_demo.py`)
docs/research/    R1 domain · R2 catalog · R3 ML SOTA · R4 platform · R5 CDF engineering
data/ models/ notebooks/   (large content git-ignored; .gitkeep retained)
```

## Models

- **Primary:** dual-head **Temporal Fusion Transformer** — quantile regression (P10/P50/P90) +
  focal-BCE exceedance head, direct multi-horizon, native known-future covariates and
  interpretable variable selection (pytorch-forecasting + Lightning).
- **Baselines (always reported):** persistence, diurnal climatology, **LightGBM (direct)** — the
  best model in the demo — and a REFM-style linear filter.
- **Backups:** N-HiTS / TiDE; fine-tuned foundation model (Chronos-Bolt / TimesFM).

## Real-data validation (NASA CDAWeb)

`scripts/real_data_demo.py` fetches live data and writes
[`reports/real_data_demo.md`](reports/real_data_demo.md). In the recorded run it pulled
**`OMNI_HRO_1MIN`** (8 solar-wind/IMF/index variables, 131 041 rows) and GOES electron flux from
**`GOES13_EPEAD-SCIENCE-ELECTRONS-E13EW_1MIN`** (`E2W_COR_FLUX`, 105 024 valid rows) over
**2017-09-01 … 2017-12-01**, merged to 26 209 rows → `X = (4000, 72, 33)` windows, and scored the
persistence baseline (nowcast PE 0.935 on real flux). Several modern GOES-16/17/18 SEISS product
IDs returned HTTP 400 from CDAWeb for that window; the loader falls back across candidate dataset
IDs until one returns data (logged in the report).

## License

MIT.
