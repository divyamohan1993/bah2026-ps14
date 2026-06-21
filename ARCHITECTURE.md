# PS-14 Master Architecture — Forecasting the >2 MeV Energetic Electron Radiation Environment at Geostationary Orbit

**BAH-2026 Problem Statement 14 · "Forecasting Energetic Particle Radiation Environment for ISRO's Geostationary Satellites"**

**Package:** `ps14` · **Status:** architecture + scaffold · **Date:** 2026-06-20

This is the authoritative design for an end-to-end AI/ML system that reads, processes, and
visualizes CDF-format space-weather data and forecasts the **>2 MeV integral electron flux at
geostationary orbit (GEO, L ≈ 6.6)** at three horizons — **30–45 min (nowcast), 6 h, 12 h** —
together with a **calibrated probability of exceeding the NOAA "harsh" alert threshold of
1000 pfu** at each horizon.

It is self-contained but cross-references the five DEEP research notes that justify every
decision:

- `docs/research/01_domain_science.md` — physics of the killer-electron belt, drivers, lags, thresholds. *(cited as **R1**)*
- `docs/research/02_satellite_data_catalog.md` — 45-mission data catalog, gap-filling & cross-validation matrix, access. *(**R2**)*
- `docs/research/03_ml_forecasting_sota.md` — model survey, baselines, UQ, imbalance, metrics, CV. *(**R3**)*
- `docs/research/04_fast_platform_engineering.md` — O(1) online features, ONNX, caching, serving, dashboard. *(**R4**)*
- `docs/research/05_cdf_data_engineering.md` — CDF/ISTP internals, products, preprocessing, leakage-free windows. *(**R5**)*

---

## Table of contents

- [(a) Problem framing & success criteria](#a-problem-framing--success-criteria)
- [(b) System overview & block diagram](#b-system-overview--block-diagram)
- [(c) Data layer](#c-data-layer)
- [(d) Feature specification](#d-feature-specification)
- [(e) Modeling](#e-modeling)
- [(f) Real-time O(1) serving](#f-real-time-o1-serving)
- [(g) Evaluation protocol](#g-evaluation-protocol--metric-definitions)
- [(h) Directory tree](#h-directory-tree)
- [(i) Data contracts](#i-data-contracts)
- [(j) Configuration schema](#j-configuration-schema)
- [(k) Phased build roadmap](#k-phased-build-roadmap)

---

## (a) Problem framing & success criteria

### a.1 The physical problem (from R1)

GEO satellites sit in the heart-to-outer-edge of the outer Van Allen belt. **>2 MeV
("relativistic"/"killer") electrons** there vary by **more than three orders of magnitude** on
timescales from minutes (dropouts) to the solar cycle. They penetrate spacecraft shielding and
bury charge in dielectrics; when the internal field exceeds breakdown an **electrostatic
discharge** can corrupt or destroy electronics (Galaxy 15, Telstar 401). Operators need advance
warning to schedule load-shedding, defer manoeuvres, and increase monitoring.

The physics gives the forecast its leverage (R1 §0–3):

- **Solar-wind speed `Vsw` is the dominant external driver**; the >2 MeV flux response **peaks
  ~1–2 days after** a `Vsw` increase (Paulikas & Blake 1979; Wing 2016 TE peak at +2 d; Wang 2024
  SHAP: Vsw lagged 1 d strongest). This lag is *why* 6 h / 12 h forecasting is feasible.
- **Acceleration** = inward radial diffusion (ULF/Pc5 waves, `Vsw`-driven) + local chorus-wave
  acceleration of a **substorm-injected seed population (~100–300 keV)**. Both take hours→~2 days.
  Therefore **lower-energy GOES channels (tens–hundreds keV) lead the >2 MeV channel.**
- **Loss** = magnetopause shadowing (dynamic-pressure / southward-Bz driven, minutes–hours) and
  EMIC/chorus precipitation. "Dropouts" can drop flux 1–3 orders of magnitude in hours.
- **L1 (Wind) gives a ~20–90 min lead** (mean ≈ 47 min) for the solar wind to reach the
  magnetosphere — the physical basis of the **30–45 min nowcast**: at that horizon skill comes
  mostly from belt persistence + immediate loss signals (Pdyn), while 6 h / 12 h increasingly tap
  the `Vsw`-driven acceleration lag.
- **Strong diurnal cycle (~1 order of magnitude):** max near local noon, min near local midnight,
  because a GOES at a fixed longitude samples a different MLT every hour. Plus a **semiannual
  (equinox) cycle (~×2)** from the Russell–McPherron effect.

### a.2 The learning problem (anchor decisions)

| Aspect | Decision | Why |
|---|---|---|
| **Target** | `log10(>2 MeV integral electron flux / pfu)` at GEO | Flux spans >3 decades; log makes errors ~homoscedastic and matches how damage/thresholds scale (R1 §7, R3 §10). |
| **Horizons** | nowcast (30–45 min), 6 h, 12 h — emitted **directly** (MIMO), read off a single 12 h decoder | Direct/joint multi-horizon avoids recursive error blow-up that sinks REFM past 1 d (R3 §5). |
| **Exceedance head** | per-horizon **P(flux ≥ 1000 pfu)**, calibrated | Operators act on *alerts*; the class is rare (~4–15 % of days); a calibrated probability beats a point estimate near a hard threshold (R1 §6–7, R3 §7). |
| **Cadence** | uniform **5-min** grid | Matches SWPC's 5-min averaged product and the directly-comparable Wei/Zhang 2024 study (R1 §5.8, R4 §1.1). |
| **Input history** | **~4 days** lookback | Empirically optimal for this target (Wei/Zhang 2024; R1 §8, R3 §10). |

### a.3 Success criteria

**Functional (maps directly to PS-14 "Expected Outcomes / Evaluation Parameters"):**

1. **Read + visualize** >2 MeV electron flux and solar-wind data from **archived CDF files**
   (ISTP-correct: `DEPEND_0`, `FILLVAL`, `VALIDMIN/MAX`, TT2000 leap-aware). *PS-14 obj. 1.*
2. **Identify important solar-wind variables** from domain knowledge **and** confirm them via the
   model's TFT variable-selection / LightGBM importances. *PS-14 obj. 2 + eval. param. 1.*
3. **Preprocess**: despike (Hampel/MAD), gap-aware interpolation, resample, log. *PS-14 obj. 3.*
4. **Multi-step forecast** accounting for input/output time history: nowcast + 6 h + 12 h. *obj. 4–5.*
5. **Demonstrate + visualize** outputs and their accuracy (live dashboard + metric reports). *obj. 6.*

**Quantitative skill bar (per horizon, on a chronological held-out test set; R1 §5, R3 §0/§8):**

| Horizon | Primary skill target (Prediction Efficiency `PE = 1 − MSE/Var`) | Must beat |
|---|---|---|
| Nowcast 30–45 min | `PE ≳ 0.90` | persistence |
| 6 h | `PE ≳ 0.80` (literature TFT-class ≈ 0.83) | persistence + diurnal climatology |
| 12 h | `PE ≳ 0.70` (literature ≈ 0.75) | climatology + REFM-style linear filter |

> `PE = Prediction Efficiency = 1 − MSE/Var(obs)` (R²-like skill; 1 = perfect, 0 = climatology).
> Report the intuitive **"uncertainty factor" = 10^RMSE(log10 flux)** alongside (R1 §5 note).

**Event (≥1000 pfu) target:** positive **TSS** and **HSS**, **ROC-AUC ≳ 0.9** at nowcast/6 h, and
a well-calibrated reliability diagram (**Brier skill score > 0**). *PS-14 "accuracy of predicted
fluxes" + R3 §8b.*

**Engineering:** fully runnable **offline** (synthetic generator → CDF → full pipeline → trained
baseline → dashboard) with **no network**; real downloaders optional. Per-minute nowcast hot path
is **amortized O(1)** per sample (R4 §0/§2).

---

## (b) System overview & block diagram

Two coupled paths share one feature definition and one model artifact:

- **Offline / training path** — bulk CDF (real or synthetic) → clean → align/merge → features →
  supervised windows → train/evaluate → export ONNX + climatology LUT.
- **Online / serving path** — SWPC real-time JSON (or synthetic replay) → ring-buffer online
  features (O(1)) → cached ONNX inference → API + dashboard.

```text
                          ┌────────────────────────────────────────────────────────────┐
                          │                       DATA SOURCES                          │
   OFFLINE (train)  ◀─────┤  Bulk CDF: GOES SEISS/EPEAD (>2 MeV target) · OMNI_HRO_1MIN  │
                          │            · Wind SWE/MFI · GRASP/GSAT (cross-val)           │
                          │  Real-time: SWPC JSON (electrons, solar wind, Kp, alerts)    │──▶ ONLINE (serve)
                          │  SYNTHETIC GENERATOR (offline, no network) → writes CDF      │
                          └───────────────┬───────────────────────────────┬────────────┘
                                          │ cdflib (read)                 │ requests (poll 60 s)
                                          ▼                               │
        ┌────────────────────────────────────────────────┐               │
        │ (1) INGEST / CDF PARSE  src/ps14/io             │               │
        │   cdf_reader: zVars → DEPEND_0 epoch (TT2000)   │               │
        │   FILLVAL / VALIDMIN/MAX mask → datetime64[ns]  │               │
        │   synthetic.py: Vsw→flux(1–2 d lag)+diurnal+CME │               │
        └───────────────┬────────────────────────────────┘               │
                        ▼                                                 │
        ┌────────────────────────────────────────────────┐               │
        │ (2) CLEAN  src/ps14/preprocess/clean.py         │               │
        │   fill/valid mask · Hampel/MAD despike          │               │
        │   gap detect (short→interp, long→NaN, flag)     │               │
        └───────────────┬────────────────────────────────┘               │
                        ▼                                                 │
        ┌────────────────────────────────────────────────┐               │
        │ (3) RESAMPLE → uniform 5-min  resample.py       │               │
        │ (4) ALIGN / MERGE (L1→GEO lag) align.py         │               │
        │ (5) TRANSFORM log10 + floor; scalers TRAIN-only │               │
        └───────────────┬────────────────────────────────┘               │
                        ▼                  ┌──────────────────────────────┐
        ┌────────────────────────────────┐│  HISTORICAL LAKE             │
        │ (6) FEATURES (offline)         ││  partitioned Parquet         │
        │   lags · rolling · coupling fns ││  + DuckDB query engine       │
        │   sin/cos MLT/season (known-fut)││  → climatology LUT (O(1))    │
        └───────────────┬────────────────┘└──────────────┬───────────────┘
                        ▼                                 │
        ┌────────────────────────────────┐               │
        │ (7) DATASETS / WINDOWS          │               │
        │   X:[N,L,F] y:[N,H] y_excd:[N,H]│               │
        │   chrono split + purge/embargo  │               │
        └───────────────┬────────────────┘               │
                        ▼                                 │
        ┌────────────────────────────────────────────────┐│
        │ (8) MODEL  src/ps14/models                      ││  ┌──────────────────────────────┐
        │   PRIMARY: dual-head TFT (P10/P50/P90 +         ││  │ (online) O(1) FEATURES        │
        │            focal-BCE exceedance)                ││  │ RingBuffer · Welford ·        │
        │   BACKUPS: N-HiTS/TiDE · Chronos/TimesFM        ││  │ MonotonicDeque (amortized O1) │
        │   BASELINES: persistence · climatology ·        ││  └──────────────┬───────────────┘
        │              LightGBM · REFM linear filter      ││                 ▼
        └───────────────┬────────────────────────────────┘│  ┌──────────────────────────────┐
                        ▼                                  │  │ (9b) CACHED O(1) INFERENCE    │
        ┌────────────────────────────────┐                 │  │ key=hash(round(features))     │
        │ (9a) EVALUATE  evaluate.py      │                 │  │ HIT→cache · MISS→ONNX(CPU)    │
        │   regression · event · prob.    │                 │  │ climatology LUT fallback      │
        │   walk-forward CV; per horizon  │                 │  └──────────────┬───────────────┘
        │   export → ONNX + LUT ──────────┼─────────────────┘                 ▼
        └────────────────────────────────┘                    ┌──────────────────────────────┐
                                                               │ (10) SERVE  FastAPI+Uvicorn  │
                                                               │ /forecast /latest /health /ws │
                                                               │ APScheduler 60 s refresh      │
                                                               │ Redis/dict hot cache          │
                                                               └──────────────┬───────────────┘
                                                                              ▼
                                                               ┌──────────────────────────────┐
                                                               │ (11) DASHBOARD  Streamlit     │
                                                               │ live flux + solar wind,       │
                                                               │ multi-horizon ± uncertainty,  │
                                                               │ alert status (1000 pfu)       │
                                                               └──────────────────────────────┘
```

**Why this is "O(1)" on the hot path (R4 §9):** the per-minute serving loop touches only the
*new* samples (O(new)), constant-time online-feature updates (ring buffer / Welford / monotonic
deque), a constant-time cache lookup (hit) or one small ONNX call (miss), and constant-time cache
writes/reads. Nothing in the serving path scales with the 11-year history or the window length.

---

## (c) Data layer

### c.1 Source roles (from R2 + R5)

| Role | Primary product | CDAWeb / endpoint | Cadence | Notes |
|---|---|---|---|---|
| **TARGET** (>2 MeV flux, GEO) | GOES-R SEISS **MPS-HI** | `DN_SEIS-L2-MPSH_G16/17/18/19` | 1/5 min | Integral >2 MeV electron channel = NOAA alert quantity (R5 §3.1). |
| **TARGET (legacy)** | GOES-13/14/15 **EPEAD** (`E2 >2 MeV`) + MAGED 40–475 keV | `GOES1x_EPS-*` | 1/5 min | Extends record pre-2018; cross-calibrate to SEISS; flag proton contamination. |
| **DRIVERS (merged)** | **`OMNI_HRO_1MIN`** | CDAWeb / OMNIWeb | 1 min | **Primary driver matrix.** Bow-shock-nose time-shifted; bundles AE/AL/SYM-H/Kp/Dst/F10.7 + IMF + plasma (R2 §F, R5 §3.3). |
| **DRIVERS (raw L1)** | **Wind** SWE (`Vp`,`Np`) + MFI (`BGSM`→Bz) | `WI_H1_SWE`, `WI_H0_MFI` | ~92 s | Sensitivity studies / explicit-lag control. |
| **CROSS-VAL** | ISRO **GRASP/GSAT** | ISSDC | — | Independent Indian-longitude validation; **held out**, never fits scalers (R5 §3.4). |
| **REAL-TIME** | SWPC JSON | `services.swpc.noaa.gov` | 1–5 min | `integral-electrons-*.json`, `solar-wind/plasma-*.json`, `mag-*.json`, `noaa-planetary-k-index.json`, `alerts.json` (R4 §1.1). |

The **45-mission catalog** (R2) is the cross-validation/gap-filling reservoir, summarized as a
matrix in R2's "Gap-Filling & Cross-Validation Matrix". Key levers we implement hooks for:

- **GEO target sensor outage** → other GEO longitudes (GOES-West, LANL-GEO, FY-4, Electro-L).
- **Absolute >2 MeV calibration** → Van Allen Probes REPT/MagEIS (community ground truth).
- **L1 driver gap** → OMNI already auto-interleaves ACE/Wind/IMP-8/Geotail at the bow-shock nose.
- **Loss/precipitation term** → POES/MetOp MEPED.

### c.2 Gap-filling & cross-validation strategy

1. **Prefer OMNI for drivers** so L1 gaps and propagation are handled upstream (R5 §3.3, §5.1).
2. **Per-variable coverage manifest** (`data/manifest.csv`): `(dataset_id, file, start, end,
   n_records, sha256, fill_fraction, download_utc)` documents exactly which spans exist and their
   gap fraction (R5 §7) — drives split boundaries and reproducibility.
3. **Short gaps interpolated + flagged; long gaps left NaN**; windows containing long gaps are
   dropped from training (R5 §4, §5.2). Never interpolate across a long outage.
4. **Cross-instrument validation** on GRASP/GSAT and (optionally) RBSP overlap as a final,
   held-out check (R3 §8e).

### c.3 Synthetic data generator (the offline backbone — `src/ps14/io/synthetic.py`)

**Mandate:** the system must be fully runnable with **no real data and no network**. The
synthetic generator produces a *physically plausible* multi-year dataset and **writes it to CDF**
so the real `cdf_reader.py` path is exercised end-to-end, plus Parquet for speed.

Design (encodes the physics from R1 so downstream models learn the right structure):

1. **Solar-wind speed `Vsw(t)`** — quiet baseline (~400 km/s) + recurrent **high-speed streams
   (HSS)** on a ~27-day recurrence + stochastic **CME shocks** (sharp rise, elevated speed/density,
   southward Bz excursion). Ornstein–Uhlenbeck noise for realism.
2. **IMF `Bz`, density `N`, dynamic pressure `Pdyn = N·Vsw²`** — coupled to events: shocks raise N
   and drive Bz southward; HSS give sustained moderate Bz fluctuation.
3. **Geomagnetic indices `AE/AL`, `Kp`, `SYM-H`** — derived from a coupling function of (Vsw, Bz)
   with appropriate response/recovery so substorm (AE) and storm (SYM-H) behavior is consistent.
4. **Seed population (sub-MeV)** — responds to AE (injection) on an hours timescale; feeds the
   >2 MeV channel.
5. **>2 MeV flux `J(t)`** — log-flux driven by a **leaky-integrator / impulse-response** of `Vsw`
   with a **1–2 day lag** and a 2-day running-mean term (R1 §2), **plus** seed-population
   contribution, **plus** a **diurnal cycle** (~1 order of magnitude, max at local noon) keyed to
   MLT, **plus** a **semiannual (equinox)** term, **minus** prompt **dropouts** when Pdyn spikes /
   Bz strongly south (magnetopause shadowing). Output is **log-normal** with realistic variance.
6. **Realism artifacts** — inject **NaN gaps** (short and long), **spikes** (so the Hampel filter
   has work to do), and per-variable **FILLVAL/VALIDMIN/MAX** so CDF masking is exercised.
7. **Write** GOES-like and OMNI-like CDFs (`cdflib.CDF` writer, TT2000 epoch, ISTP attributes)
   under `data/raw/synthetic/`, plus a mirror Parquet.

This generator is also the **online replay source**: the serving path can consume a synthetic
SWPC-shaped stream when the network is unavailable.

---

## (d) Feature specification

Ranking synthesizes R1 §8 (Wing 2016, Wang 2024 SHAP, Chu 2021, Boynton 2016, Sakaguchi 2013,
Ganushkina 2024, Wei/Zhang 2024). **Known-future** = available for all horizons → fed to the TFT
*decoder* (calendar/geometry covariates); everything else is **observed-past** → TFT *encoder*.

| # | Feature (column) | Physical role | Lag / window | Source | Known-future? |
|---|---|---|---|---|---|
| 1 | `log_flux_e2` (autoregressive >2 MeV) | Belt has multi-day memory; strongest short-horizon predictor | lags 0 → ~4 d; rolling mean/std/min/max over 1 h/6 h/24 h | GOES SEISS/EPEAD | no (observed) |
| 2 | `vsw` + `vsw_mean_2d`, `vsw_lag_24h`, `vsw_lag_48h` | #1 external driver (ULF diffusion + chorus) | **+1–2 d lag** + 2-d running mean; instantaneous (propagated) for nowcast loss | OMNI / Wind | no |
| 3 | `ae`, `al` (+ `ae_mean_1d`) | Substorm injection of seed/source pop; chorus generation | lags 0 → +2 d | OMNI | no |
| 4 | `log_flux_seed` (GOES sub-MeV 40 keV–0.8 MeV) | Direct precursor accelerated to >2 MeV; faster than Vsw | lags hours → ~1 d | GOES MPS-HI/MAGED | no |
| 5 | `bz_gsm` + coupling fns `newell`, `vbs`, `epsilon` | Southward → injection/ULF (build-up); extreme south+shock → loss | accel ~h–d; loss min–h (propagated) | OMNI / Wind | no |
| 6 | `density` | Low N favors enhancement; high-N pulses → shadowing loss | instantaneous–hours (propagated) | OMNI / Wind | no |
| 7 | `pdyn` (∝ N·Vsw²) | Magnetopause compression → prompt shadowing loss | minutes–hours (propagated) | derived | no |
| 8 | `kp` | Coarse global activity proxy | 0 → +1–2 d | OMNI | no |
| 9 | `sym_h` (Dst) | Storm phase (main-phase depletion vs recovery) | history over storm (days) | OMNI | no |
| 10 | `mlt_sin`, `mlt_cos` | ~1-order diurnal noon/midnight variation at GEO | instantaneous (cyclic) | ephemeris / clock | **yes** |
| 11 | `tod_sin`, `tod_cos` | Time-of-day diurnal cycle | cyclic | calendar | **yes** |
| 12 | `doy_sin`, `doy_cos` | Semiannual / equinox (Russell–McPherron, ~×2) | slow cyclic | calendar | **yes** |
| 13 | `r0_standoff` (Shue magnetopause standoff) | Dayside compression / shadowing geometry | instantaneous | derived (Pdyn, Bz) | no |
| 14 | `*_imputed` masks | Missingness indicator per imputed channel | per sample | preprocessing | no |
| 15 | `sat_id` / `longitude` (static) | Calibration/longitude offset between platforms | static | metadata | static covariate |

**Coupling functions (physics-informed; R3 §10, Newell 2007):**

- `vbs` = half-wave-rectified dawn–dusk E-field (`v·Bs`, 0 when `Bz>0`).
- `newell` = `v^(4/3) · B_T^(2/3) · sin^(8/3)(θc/2)` (universal SW→magnetosphere coupling).
- `epsilon` ∝ `v·B² · sin⁴(θc/2)` (Akasofu energy input).
- `θc` clock angle = `atan2(By, Bz)`.

**Target/diurnal note:** the diurnal + 27-day cycles are physical and predictable, so the cyclic
encodings are *known-future* decoder inputs (R3 §0/§10). Optionally model `Δlog_flux`
(differenced) to remove diurnal/27-day structure (R3 §10) — provided as a config flag.

Final feature ranking is **confirmed empirically** by the TFT variable-selection network and the
LightGBM feature importances — satisfying PS-14's "identify important solar-wind variables".

---

## (e) Modeling

### e.1 Primary model — dual-head Temporal Fusion Transformer (TFT)

**Why TFT (R3 §3c, §11):** purpose-built for multi-horizon with **mixed inputs** (static /
known-future / observed-past channels), native **quantile (probabilistic)** outputs, and an
**interpretable variable-selection network** that directly answers the "important drivers"
objective. Attention-class models already beat LSTM on this exact target (Wei/Zhang 2024).

```text
        static: sat_id/longitude ──► Static Covariate Encoders ──► context vectors
                                                                       │
  observed-past (encoder)            known-future (decoder)            │
  log_flux_e2, vsw, ae, seed,        mlt_sin/cos, tod_sin/cos,         │
  bz, density, pdyn, kp, sym_h,      doy_sin/cos                       ▼
  coupling fns, *_imputed                 │                  Variable Selection Networks
        │                                 │                   (per-timestep, interpretable)
        ▼                                 ▼                            │
   ┌────────────── LSTM encoder ──┐  ┌── LSTM decoder ──┐              │
   │  local processing            │  │  local processing │◀────────────┘
   └──────────────┬───────────────┘  └────────┬─────────┘
                  └────────► Interpretable Multi-Head Attention ◄──────┘
                                         │
                          Gated Residual Networks + gating
                                         │
                 ┌───────────────────────┴────────────────────────┐
                 ▼                                                 ▼
   HEAD 1: Quantile regression                   HEAD 2: Exceedance classification
   P10 / P50 / P90 of log10 flux                 P(flux ≥ 1000 pfu) per horizon
   at {nowcast, 6 h, 12 h}                        (sigmoid)
   loss = pinball (τ=0.1,0.5,0.9)                 loss = focal-BCE (γ for rare events)
```

**Combined loss (R3 §6–7):**

```
L = Σ_h Σ_τ pinball_τ( y_h, q_{τ,h} )  +  λ · Σ_h focal_BCE( 1[y_h ≥ log10(1000)], p_h )
```

with optional **inverse-frequency / output-weighting** on the regression term to fight
storm-peak underestimation, and `λ` balancing the heads. Library: **pytorch-forecasting +
PyTorch-Lightning** (`TimeSeriesDataSet` handles the static/known-future/observed-past split;
`QuantileLoss` is native; the exceedance head is added as a custom multi-task wrapper).

**Multi-horizon strategy:** **direct / joint (MIMO)** — one decoder spanning 12 h at 5-min cadence
(144 steps); the nowcast (≈ steps 6–9), 6 h (step 72), and 12 h (step 144) are read off the single
multi-step output (R3 §5). Avoids recursive error accumulation.

**Uncertainty:** train with pinball → report **P10/P50/P90**; optionally a **deep ensemble (3–5
seeds)** and a **Conformalized Quantile Regression** wrapper on a held-out calibration block for
distribution-free coverage (R3 §6).

### e.2 Strong baselines (must implement — R3 §1)

| Baseline | Definition | Role |
|---|---|---|
| **Persistence** | `ŷ(t+h) = y(t)` | The bar to beat at nowcast (autocorrelation makes it strong). |
| **Diurnal climatology** | mean log-flux per (local-time bin, Kp bin, day-of-year) from train | Captures diurnal + seasonal; long-horizon reference + serving fallback LUT. |
| **LightGBM (direct multi-horizon)** | GBDT on lag + rolling + coupling features, one model per horizon | Strongest non-DL baseline + feature-importance oracle; deliverable-grade. |
| **REFM-style linear filter** | linear prediction filter on `Vsw` (NOAA REFM lineage) | The literal operational benchmark; PE collapses past ~1 d (the opportunity). |

### e.3 Backups (R3 §11)

- **N-HiTS / TiDE** (neuralforecast/darts): fast, simple, strong direct multi-horizon; robust
  fallback if TFT overfits the limited data.
- **Fine-tuned foundation model** — **Chronos-Bolt** (T5, native quantiles + covariates,
  Apache-2.0) or **TimesFM+Cov** (validated on this exact target, R²≈0.90 @6 h with few-shot).
  Pair with the exceedance head to fix known storm-peak underestimation / onset lag.

All models implement the same `Forecaster` ABC (see [(i) Data contracts](#i-data-contracts)) so
they are interchangeable in train/evaluate/serve.

---

## (f) Real-time O(1) serving

The hot path keeps every per-sample update **amortized O(1)** regardless of window length (R4
§2). Three primitives are **fully implemented** in `src/ps14/features/online.py`.

### f.1 Ring buffer — O(1) append / O(1) evict

Fixed-size NumPy-backed circular buffer with wrap-around `head`/`count`. `list.pop(0)` is O(N);
the ring buffer is O(1) with fixed memory and a contiguous view for vectorized reads.

```
push(x): buf[head] = x; head = (head+1) % capacity; count = min(count+1, capacity)   # O(1)
```

### f.2 Welford — O(1) running mean & variance (numerically stable)

```
n += 1; δ = x − μ; μ += δ/n; δ2 = x − μ; M2 += δ·δ2
var_pop = M2/n ;  var_samp = M2/(n−1)
```

O(1) time/memory per tracked statistic, no stored history; avoids catastrophic cancellation of
the naive "sum-of-squares" form. A **windowed variant** supports the symmetric remove-then-add
update for a fixed-N sliding window. Used for solar-wind running stats, flux z-scores, online
normalization (R4 §2.2).

### f.3 Monotonic deque — amortized O(1) rolling min / max

Deque of indices whose values are monotonic (non-increasing for max ⇒ front = window maximum).
Each index is pushed once and popped at most once ⇒ amortized O(1)/sample; query is O(1) — strictly
better than a heap (O(log n)) or rescan (O(window)).

```
push_max(j, x, window):
  while dq and dq[0]   <= j − window: dq.popleft()   # drop expired
  while dq and val[dq[-1]] <= x:      dq.pop()        # drop dominated
  dq.append(j); return val[dq[0]]                      # window max, O(1)
```

### f.4 Cached inference + climatology LUT (the real O(1) win — R4 §3.5)

- **Forecast memo**: `key = hash(round(features, k))`; identical/near-identical inputs (e.g. no new
  sample) return in O(1) with zero model calls. Back with Redis (shared) or a process `dict` /
  `lru_cache` (sub-µs).
- **Latest-value cache**: most recent flux/solar-wind/Kp + current forecast under fixed keys → O(1)
  API/dashboard reads.
- **Climatology LUT**: baseline flux quantiles by `(day-of-year, Kp-bin, Vsw-bin, longitude)`
  precomputed from history → O(1) baseline retrieval, anomaly scoring, and model fallback.
- **Inference engine**: **ONNX Runtime on CPU**, single warm `InferenceSession`,
  `graph_optimization_level=ORT_ENABLE_ALL`, `inter_op_num_threads=1` for low tail latency;
  optional int8/fp16 (benchmark first — int8 can be slower on tiny models). TorchScript fallback;
  TensorRT only on GPU (R4 §3).

### f.5 Serving stack & latency budget (R4 §6)

**FastAPI + Uvicorn + APScheduler (60 s) + Redis/dict**, optional Docker.

| Stage | Budget |
|---|---|
| Poll SWPC JSON (network) | ~50–300 ms |
| Parse + dedup + ring-buffer/feature update (O(new)) | < 5 ms |
| Inference (ORT CPU, cache miss) | < 5 ms |
| Cache write + WebSocket push | < 5 ms (+~1 ms Redis) |
| **Total compute** | **≈ 10–20 ms** (vs 60 s refresh) |

Wall-clock freshness is bounded by SWPC's 1–5 min cadence, not our code. **Storage:** partitioned
Parquet (durable lake) + DuckDB (query engine) + Arrow (zero-copy) + Redis/dict (hot cache);
optional Zarr only for N-D spectra (R4 §4).

---

## (g) Evaluation protocol & metric definitions

Report **every metric per horizon separately** (nowcast / 6 h / 12 h) — PE degrades strongly with
horizon, so an aggregate hides failure modes (R3 §8d). Implemented in `src/ps14/metrics.py`.

### g.1 Regression (per horizon, in log10 space)

| Metric | Definition |
|---|---|
| RMSE, MAE | on `log10(flux)`. |
| **Prediction Efficiency PE** | `1 − MSE/Var(obs)` (R²-like skill; = R² vs the mean). |
| **Skill score vs reference** | `1 − MSE_model/MSE_ref`, ref ∈ {persistence (short h), climatology / REFM (long h)}. Must be **> 0** to be useful. |
| R², linear correlation (LC), bias | standard; bias = mean error. |
| **Uncertainty factor** | `10^RMSE(log10 flux)` (intuitive "off by ×N"). |

### g.2 Event / threshold (≥1000 pfu) — from the 2×2 contingency table

`POD` (recall), `FAR` (false-alarm ratio), `POFD`, `CSI`, **`HSS`** (Heidke), **`TSS = POD − POFD`**
(True Skill), `F1`, **`ROC-AUC`**. Space-weather verification emphasizes F1/POD/TSS/HSS (R3 §8b;
SWPC verification glossary). Probabilistic event: **reliability diagram**, **Brier score / Brier
skill score**.

### g.3 Probabilistic continuous

**Pinball loss** (per quantile), **CRPS**, **PICP / interval coverage** (does P10–P90 contain the
truth ~80 % of the time?), reliability diagrams (R3 §8c).

### g.4 CV protocol (leakage-critical — R3 §8e, R5 §5.2)

- **Strict chronological split** (~7 yr train / 2 yr val / 2 yr test), **never random** — random
  splits leak future info through autocorrelation.
- **Walk-forward (expanding) CV** for hyperparameters — mirrors operational retraining.
- **Purge + embargo ≥ lookback + horizon** between splits so no window straddles a boundary.
- **Scalers / log floor / clip bounds fit on TRAIN only.**
- Ensure each split spans quiet **and** storm periods.
- Separate **calibration block** for conformal prediction.
- Final independent cross-instrument check on **GRASP/GSAT**.

---

## (h) Directory tree

```text
bah2026-ps14/
├── ARCHITECTURE.md            # this document (headline deliverable)
├── CONTRACTS.md               # authoritative inter-module data contracts
├── README.md                  # overview + quickstart, maps to PS-14 objectives
├── pyproject.toml             # package `ps14`, src layout, core + optional extras
├── Makefile                   # setup synth-data fetch-data preprocess features train ...
├── .gitignore                 # python + data/* (keep .gitkeep) + models + *.cdf (keep synthetic)
├── config/
│   ├── default.yaml           # paths, sources, preprocessing, features, horizons, thresholds, splits, serving
│   └── model/
│       ├── tft.yaml           # TFT dual-head hyperparameters
│       ├── nhits.yaml         # N-HiTS backup hyperparameters
│       └── baseline.yaml      # persistence / climatology / LightGBM / REFM params
├── data/
│   ├── raw/.gitkeep           # downloaded + synthetic CDFs (immutable)
│   ├── interim/.gitkeep       # per-variable cleaned series (parquet)
│   └── processed/.gitkeep     # merged grid + window tensors
├── models/.gitkeep            # trained checkpoints, ONNX, scalers, LUTs
├── notebooks/.gitkeep         # EDA only (import from src/)
├── docs/
│   └── research/              # R1–R5 (DO NOT MODIFY)
│       ├── 01_domain_science.md
│       ├── 02_satellite_data_catalog.md
│       ├── 03_ml_forecasting_sota.md
│       ├── 04_fast_platform_engineering.md
│       └── 05_cdf_data_engineering.md
├── src/
│   └── ps14/
│       ├── __init__.py
│       ├── config.py          # pydantic Settings loading YAML
│       ├── constants.py       # channels, thresholds (1000 pfu), horizon steps, units
│       ├── cli.py             # typer/argparse CLI dispatching the Makefile targets
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── logging.py     # structured logging setup
│       │   └── timeops.py     # cyclic encodings, L1→GEO lag, TT2000 helpers
│       ├── io/
│       │   ├── __init__.py
│       │   ├── cdf_reader.py  # cdflib: discover vars, DEPEND_0 epoch, FILLVAL/VALIDMIN/MAX, TT2000
│       │   ├── cdaweb.py      # cdasws/pyspedas fetch + cache
│       │   ├── hapi.py        # hapiclient fetch
│       │   ├── swpc_realtime.py  # SWPC JSON client (both list-of-lists & list-of-objects)
│       │   └── synthetic.py   # physically-plausible generator → writes CDF + parquet
│       ├── preprocess/
│       │   ├── __init__.py
│       │   ├── clean.py       # Hampel/MAD despike, fill/valid mask, gap detect
│       │   ├── resample.py    # uniform 5-min resample
│       │   ├── align.py       # L1→GEO propagation + merge
│       │   └── transform.py   # log10 + floor, scalers (train-only)
│       ├── features/
│       │   ├── __init__.py
│       │   ├── offline.py     # lags, rolling, coupling fns, time encodings
│       │   └── online.py      # RingBuffer, Welford, MonotonicDeque (FULLY IMPLEMENTED)
│       ├── datasets/
│       │   ├── __init__.py
│       │   ├── windowing.py   # supervised windows, chrono split, purge/embargo
│       │   └── schema.py      # canonical column/dtype contract + validation (IMPLEMENTED)
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py        # Forecaster ABC
│       │   ├── baselines.py   # Persistence, Climatology, LightGBM, REFM linear filter
│       │   ├── tft.py         # dual-head TFT
│       │   ├── nhits.py       # N-HiTS / TiDE backup
│       │   └── foundation.py  # Chronos-Bolt / TimesFM backup
│       ├── metrics.py         # regression + event + probabilistic metrics (IMPLEMENTED)
│       ├── evaluate.py        # walk-forward evaluation harness
│       ├── train.py           # training orchestration
│       ├── serve/
│       │   ├── __init__.py
│       │   ├── inference.py   # ONNX load + hash cache + climatology LUT
│       │   ├── api.py         # FastAPI app (/forecast /latest /health /ws)
│       │   └── scheduler.py   # APScheduler 60 s refresh
│       └── dashboard/
│           ├── __init__.py
│           └── app.py         # Streamlit skeleton
└── tests/
    ├── __init__.py
    ├── test_synthetic.py
    ├── test_cdf_reader.py
    ├── test_preprocess.py
    ├── test_features_online.py   # real tests for the O(1) primitives
    ├── test_windowing.py
    ├── test_metrics.py
    ├── test_baselines.py
    └── test_schema.py
```

---

## (i) Data contracts

> The authoritative, exhaustive version lives in **`CONTRACTS.md`**. Summary here so the design is
> self-contained. All times are **UTC, `datetime64[ns]`**, on a uniform **5-min** grid. Flux units
> are **pfu = particles cm⁻² s⁻¹ sr⁻¹**; `log_*` columns are `log10(value)`.

### i.1 Canonical "merged dataframe" (after clean → align → transform)

`pandas.DataFrame` indexed by `time: DatetimeIndex[ns, UTC]` (name `"time"`), 5-min freq, sorted,
unique. Core columns (extra mission columns allowed; these are required/canonical):

| Column | Dtype | Units | Description |
|---|---|---|---|
| `flux_e2` | float64 | pfu | >2 MeV integral electron flux at GEO (target, linear). |
| `log_flux_e2` | float64 | log10 pfu | `log10(flux_e2)` with positive floor (the model target). |
| `flux_seed` | float64 | pfu | Sub-MeV seed-channel integral flux. |
| `log_flux_seed` | float64 | log10 pfu | `log10(flux_seed)`. |
| `vsw` | float64 | km/s | Solar-wind bulk speed (propagated/merged). |
| `density` | float64 | cm⁻³ | Solar-wind proton density. |
| `pdyn` | float64 | nPa | Dynamic pressure (∝ N·Vsw²). |
| `bz_gsm` | float64 | nT | IMF Bz (GSM). |
| `bt` | float64 | nT | IMF magnitude \|B\|. |
| `ae` | float64 | nT | Auroral electrojet index. |
| `al` | float64 | nT | AL index. |
| `kp` | float64 | — | Planetary K-index (0–9, may be fractional). |
| `sym_h` | float64 | nT | SYM-H (high-res Dst). |
| `f107` | float64 | sfu | F10.7 solar radio flux. |
| `mlt` | float64 | hours | Magnetic local time of the GEO sensor (0–24). |
| `longitude` | float64 | deg | GEO sensor longitude (static per satellite). |
| `sat_id` | category | — | Source satellite identifier. |
| `<col>_imputed` | int8 | {0,1} | 1 where `<col>` was interpolated over a short gap. |

**Invariants:** monotonic-increasing unique 5-min index; no `inf`; `flux_* ≥ floor > 0` before
log; long gaps remain `NaN` (not imputed); `mlt ∈ [0,24)`, `kp ∈ [0,9]`. Validated by
`ps14.datasets.schema.validate_merged(df)`.

### i.2 Feature matrix (after `features/offline.py`)

Same `time` index. Adds engineered columns (all float64 unless noted):

- **Lags:** `log_flux_e2_lag_{1,6,72,288,576}` (5-min steps), `vsw_lag_{288,576}` (24 h/48 h).
- **Rolling:** `log_flux_e2_roll{mean,std,min,max}_{12,72,288}`, `vsw_rollmean_576` (2-d), `ae_rollmean_288`.
- **Coupling:** `vbs`, `newell`, `epsilon`, `clock_angle`, `r0_standoff`.
- **Known-future cyclic:** `tod_sin`,`tod_cos`,`doy_sin`,`doy_cos`,`mlt_sin`,`mlt_cos`.
- **Masks:** the `*_imputed` columns propagate.

`schema.FEATURE_COLUMNS`, `schema.KNOWN_FUTURE_COLUMNS`, `schema.STATIC_COLUMNS`,
`schema.TARGET_COLUMN = "log_flux_e2"` are the canonical name lists.

### i.3 Supervised-window tensors (after `datasets/windowing.py`)

For lookback `L` (default 1152 = 4 d at 5-min), horizon `H` (default 144 = 12 h), `F` features,
`n_h` named horizons (default 3: nowcast/6 h/12 h):

| Array | Shape | Dtype | Meaning |
|---|---|---|---|
| `X` | `[N, L, F]` | float32 | Encoder features over `[t−L+1 … t]` (knowable at `t`). |
| `X_future` | `[N, H, F_kf]` | float32 | Known-future covariates over `[t+1 … t+H]`. |
| `y` | `[N, n_h]` | float32 | `log10` flux at the named horizons. |
| `y_exceed` | `[N, n_h]` | float32 ∈ {0,1} | `1[flux ≥ 1000 pfu]` at each horizon. |
| `t_index` | `[N]` | datetime64[ns] | Anchor time `t` of each window. |

**Hard rule:** no value at `> t` appears in `X`. Windows containing long-gap `NaN` are dropped.
`HORIZON_STEPS = {"nowcast": 8, "6h": 72, "12h": 144}` (5-min steps; nowcast ≈ 40 min). Persisted
as `data/processed/windows.npz`.

### i.4 Model interface (`models/base.py`)

```python
class Forecaster(ABC):
    def fit(self, X, X_future, y, y_exceed) -> "Forecaster": ...
    def predict(self, X, X_future) -> np.ndarray:            # [N, n_h] median log-flux
    def predict_quantiles(self, X, X_future) -> dict[float, np.ndarray]:  # τ -> [N, n_h]
    def predict_proba_exceed(self, X, X_future) -> np.ndarray:  # [N, n_h] in [0,1]
    def save(self, path) -> None: ...
    @classmethod
    def load(cls, path) -> "Forecaster": ...
```

### i.5 Forecast payload (serving JSON — `/forecast`)

```json
{
  "issued_utc": "2026-06-20T12:00:00Z",
  "satellite": "GOES-18",
  "horizons": {
    "nowcast": {"lead_min": 40, "p10": 2.1, "p50": 2.6, "p90": 3.0,
                 "flux_p50_pfu": 398.1, "p_exceed_1000pfu": 0.12, "alert": false},
    "6h":      {"lead_min": 360, "...": "..."},
    "12h":     {"lead_min": 720, "...": "..."}
  },
  "threshold_pfu": 1000.0,
  "model": "tft-dualhead-v1",
  "source": "synthetic|swpc"
}
```

---

## (j) Configuration schema

`config/default.yaml` is loaded and validated by `ps14.config` (pydantic). Top-level sections:

```yaml
paths:        {data_raw, data_interim, data_processed, models, reports}
data:
  sources:    {goes_target, omni_drivers, wind_backup, grasp_val, swpc_realtime}  # dataset IDs / URLs
  time_range: {start, end}
  satellites: [GOES-16, GOES-17, GOES-18]
preprocess:
  cadence: "5min"
  hampel:   {window: 7, n_sigma: 3.0, replace: "nan"}
  gaps:     {max_gap_steps: 6}            # 30 min; longer stays NaN
  log_floor_pfu: 0.01
  l1_to_geo:  {method: "omni_preshifted"} # or "ballistic"
features:
  lookback_steps: 1152                    # 4 days at 5-min
  lags_steps:    [1, 6, 72, 288, 576]
  roll_windows:  [12, 72, 288, 576]
  coupling:      [vbs, newell, epsilon]
  difference_target: false
model:
  name: "tft"                             # tft | nhits | foundation | persistence | climatology | lightgbm | refm
  horizons_steps: {nowcast: 8, "6h": 72, "12h": 144}
  decoder_steps: 144                      # full 12 h MIMO decoder
  quantiles: [0.1, 0.5, 0.9]
thresholds:
  harsh_pfu: 1000.0
  exceedance_loss_weight: 1.0             # lambda for focal-BCE head
split:
  mode: "chronological"                   # train < val < test
  train: 0.7
  val: 0.15
  embargo_steps: 1296                     # >= lookback + horizon
serving:
  refresh_seconds: 60
  poll: {electrons: 60, solar_wind: 60, kp: 600}
  cache: "dict"                           # dict | redis
  onnx: {threads_intra: 2, threads_inter: 1, opt_level: "all"}
  source: "synthetic"                     # synthetic | swpc
seed: 1993
```

`config/model/{tft,nhits,baseline}.yaml` carry per-model hyperparameters (hidden sizes, attention
heads, learning rate, GBDT params, etc.) and are merged over `model:`.

---

## (k) Phased build roadmap

| Phase | Deliverable | Modules | Exit criterion |
|---|---|---|---|
| **0. Scaffold** *(this PR)* | Importable package, configs, contracts, tests for O(1) primitives + metrics + schema | all stubs + `online.py`, `metrics.py`, `schema.py` | `python -c "import ps14"` works; primitive/metric/schema tests pass. |
| **1. Synthetic + IO** | Offline data exists end-to-end | `io/synthetic.py`, `io/cdf_reader.py` | `make synth-data` writes CDFs; `cdf_reader` round-trips them; `test_synthetic`/`test_cdf_reader` pass. |
| **2. Preprocess** | Clean 5-min merged frame | `preprocess/*` | `validate_merged` passes on synthetic output; despike/gap/resample tests pass. |
| **3. Features + windows** | Model-ready tensors | `features/offline.py`, `datasets/windowing.py` | `windows.npz` with correct shapes; no-look-ahead test passes. |
| **4. Baselines + metrics** | Skill reference numbers | `models/baselines.py`, `evaluate.py` | Persistence/climatology/LightGBM/REFM scored per horizon; report generated. |
| **5. TFT dual-head** | Primary model | `models/tft.py`, `train.py` | TFT beats persistence@nowcast and climatology@6 h on val; quantiles calibrated. |
| **6. Serving + ONNX** | Real-time O(1) loop | `serve/*` | `/forecast` returns cached multi-horizon JSON; export → ONNX; latency < budget. |
| **7. Dashboard** | Live demo | `dashboard/app.py` | Streamlit shows live flux/solar-wind + forecast ± bands + alert status. |
| **8. Backups + CV** | Robustness | `models/{nhits,foundation}.py` | Walk-forward CV; backups benchmarked; GRASP/GSAT cross-validation. |

---

*End of ARCHITECTURE.md. Decisions trace to the five research notes (R1–R5) under
`docs/research/`. The companion `CONTRACTS.md` is the binding inter-module spec.*
