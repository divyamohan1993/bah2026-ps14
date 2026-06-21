# PS-14 Data Contracts

**Authoritative, binding spec of the inter-module data contracts** so parallel builders can work
independently. If code and this document disagree, this document wins until amended. Companion to
`ARCHITECTURE.md`. Canonical constants live in `src/ps14/datasets/schema.py` and
`src/ps14/constants.py`; this file is the human-readable source of truth they encode.

**Global conventions**

- **Time:** UTC, `numpy.datetime64[ns]`, naive (no tz object), index name `"time"`. Uniform
  **5-minute** grid, monotonically increasing, unique. CDF epochs are converted leap-aware
  (TT2000 → UTC) *before* resampling (R5 §1.2).
- **Flux units:** `pfu = particles cm⁻² s⁻¹ sr⁻¹`. Columns prefixed `log_` are `log10(value)`
  with a positive floor applied first (`config.preprocess.log_floor_pfu`, default `0.01`).
- **Missing data:** `NaN` only (never sentinel fill values — those are masked at read time).
  `*_imputed` int8 companion columns mark short-gap interpolation; long gaps stay `NaN`.
- **Dtypes:** science columns `float64` in dataframes; window tensors `float32`; masks/flags
  `int8`; `sat_id` `category`.
- **Harsh threshold:** `HARSH_PFU = 1000.0` → in log space `LOG_HARSH = log10(1000) = 3.0`.

---

## 1. CDF read contract (`io/cdf_reader.py`)

A reader function returns, per science variable, a **time-indexed `pandas.Series`** (or a
`DataFrame` for multi-channel variables) with:

- index = `DatetimeIndex[ns]` named `"time"` (leap-aware from `DEPEND_0`).
- values = `float64`, with `FILLVAL` and out-of-`[VALIDMIN, VALIDMAX]` set to `NaN`.
- `series.attrs["units"]`, `series.attrs["catdesc"]`, `series.attrs["source_var"]` populated.

**Discovery rule (R5 §1.3):** open → list zVariables → keep `VAR_TYPE == "data"` → for each, read
its `DEPEND_0` epoch → mask fill/valid → apply any quality-flag companion → convert epoch to
`datetime64[ns]`. Readers must support record/time subsetting so 11 years need not load at once.

---

## 2. Canonical MERGED dataframe (`schema.MERGED_COLUMNS`)

Produced by `preprocess` (clean → resample → align → transform). Indexed by `time` (see global
conventions). **Required canonical columns** (mission-specific extras permitted but ignored by the
model unless added to `FEATURE_COLUMNS`):

| Column | Dtype | Units | Role | Notes |
|---|---|---|---|---|
| `flux_e2` | float64 | pfu | target (linear) | >2 MeV integral electron flux at GEO. |
| `log_flux_e2` | float64 | log10 pfu | **TARGET** | `log10(flux_e2)`, floored. `schema.TARGET_COLUMN`. |
| `flux_seed` | float64 | pfu | feature | sub-MeV (40 keV–0.8 MeV) seed flux. |
| `log_flux_seed` | float64 | log10 pfu | feature | `log10(flux_seed)`, floored. |
| `vsw` | float64 | km/s | feature | solar-wind bulk speed (merged/propagated). |
| `density` | float64 | cm⁻³ | feature | solar-wind proton density. |
| `pdyn` | float64 | nPa | feature | dynamic pressure ∝ N·Vsw². |
| `bz_gsm` | float64 | nT | feature | IMF Bz (GSM). |
| `bt` | float64 | nT | feature | IMF magnitude \|B\|. |
| `ae` | float64 | nT | feature | auroral electrojet index. |
| `al` | float64 | nT | feature | AL index. |
| `kp` | float64 | — | feature | planetary K-index (0–9). |
| `sym_h` | float64 | nT | feature | SYM-H (high-res Dst). |
| `f107` | float64 | sfu | feature | F10.7 solar radio flux. |
| `mlt` | float64 | hours | feature/geom | magnetic local time of sensor (0–24). |
| `longitude` | float64 | deg | static | GEO sensor longitude. |
| `sat_id` | category | — | static | source satellite. |
| `{col}_imputed` | int8 | {0,1} | mask | 1 where `{col}` short-gap interpolated. |

**Invariants** (enforced by `schema.validate_merged`):

1. Index is `DatetimeIndex[ns]`, name `"time"`, sorted, unique, uniform 5-min step.
2. All `MERGED_REQUIRED` columns present with the dtype above.
3. No `±inf`; `flux_*` are either `NaN` or `≥ log_floor`-implied positive value.
4. `mlt ∈ [0, 24)`; `kp ∈ [0, 9]`; `*_imputed ∈ {0, 1}`.
5. Long gaps (> `max_gap_steps`) remain `NaN` (not imputed); rows are never dropped to fill them.

---

## 3. FEATURE matrix (`schema.FEATURE_COLUMNS`)

Produced by `features/offline.py` from the merged frame; same `time` index. All `float64` unless
noted. Builders must keep these names exact.

**Observed-past base** (subset of merged): `log_flux_e2`, `log_flux_seed`, `vsw`, `density`,
`pdyn`, `bz_gsm`, `bt`, `ae`, `al`, `kp`, `sym_h`, `f107`.

**Lag features** (steps are 5-min units): `log_flux_e2_lag_1`, `log_flux_e2_lag_6`,
`log_flux_e2_lag_72`, `log_flux_e2_lag_288`, `log_flux_e2_lag_576`, `vsw_lag_288`, `vsw_lag_576`.

**Rolling features:** `log_flux_e2_rollmean_{12,72,288}`, `log_flux_e2_rollstd_{72,288}`,
`log_flux_e2_rollmin_72`, `log_flux_e2_rollmax_72`, `vsw_rollmean_576`, `ae_rollmean_288`.

**Coupling functions:** `vbs`, `newell`, `epsilon`, `clock_angle`, `r0_standoff`.

**Known-future cyclic** (`schema.KNOWN_FUTURE_COLUMNS`): `tod_sin`, `tod_cos`, `doy_sin`,
`doy_cos`, `mlt_sin`, `mlt_cos`.

**Static** (`schema.STATIC_COLUMNS`): `sat_id` (category), `longitude` (float64).

**Masks:** all `{col}_imputed` int8 columns propagate.

Rolling/lag windows reference `config.features`. `schema.validate_features(df)` checks presence
and dtype of `FEATURE_COLUMNS + KNOWN_FUTURE_COLUMNS` and that there are no `±inf`.

---

## 4. SUPERVISED-WINDOW tensors (`datasets/windowing.py`)

Parameters (defaults from `config`): lookback `L = 1152` (4 d), full MIMO decoder `H = 144`
(12 h), named horizons `n_h = 3`, `F = len(FEATURE_COLUMNS)`, `F_kf = len(KNOWN_FUTURE_COLUMNS)`.

| Array | Shape | Dtype | Definition |
|---|---|---|---|
| `X` | `[N, L, F]` | float32 | features over `[t−L+1 … t]` — every value knowable at `t`. |
| `X_future` | `[N, H, F_kf]` | float32 | known-future covariates over `[t+1 … t+H]`. |
| `y` | `[N, n_h]` | float32 | `log10` flux at named horizons (columns ordered by `HORIZON_NAMES`). |
| `y_exceed` | `[N, n_h]` | float32 | `1.0` if `flux ≥ HARSH_PFU` at that horizon else `0.0`. |
| `t_index` | `[N]` | datetime64[ns] | anchor time `t` per window. |

**Horizon map** (`constants.HORIZON_STEPS`, 5-min steps): `{"nowcast": 8, "6h": 72, "12h": 144}`
(`HORIZON_NAMES = ["nowcast", "6h", "12h"]`; nowcast 8 steps ≈ 40 min, inside the 30–45 min band).

**Rules (R5 §5.2):** (1) no value at `> t` in `X`; (2) `y`/`y_exceed` strictly future; (3) drop
any window whose `X` or required `y` spans a long-gap `NaN`; (4) returned `X`/`y` for DL models are
unscaled — scaling is applied by a scaler **fit on train only** (R5 §4.9). Persisted via
`np.savez_compressed` to `data/processed/windows.npz` with keys
`X, X_future, y, y_exceed, t_index, feature_cols, known_future_cols, horizon_names`.

### 4.1 Chronological split contract

`chronological_split(t_index, train, val, embargo_steps) -> (train_idx, val_idx, test_idx)` returns
integer index arrays into `N`, with an **embargo gap ≥ L + max(H)** removed between consecutive
segments so no window straddles a boundary. Order is strictly `train < val < test` by time.

---

## 5. MODEL interface (`models/base.py::Forecaster`)

Every model (baseline, TFT, N-HiTS, foundation) implements this ABC so train/evaluate/serve treat
them interchangeably. All array I/O matches §4.

```python
class Forecaster(ABC):
    name: str
    horizon_names: list[str]          # == constants.HORIZON_NAMES

    def fit(self, X, X_future, y, y_exceed, *, val=None) -> "Forecaster": ...
    def predict(self, X, X_future) -> np.ndarray:                     # [N, n_h] median log-flux
    def predict_quantiles(self, X, X_future) -> dict[float, np.ndarray]:  # {τ: [N, n_h]}
    def predict_proba_exceed(self, X, X_future) -> np.ndarray:        # [N, n_h] ∈ [0,1]
    def save(self, path: str | Path) -> None: ...
    @classmethod
    def load(cls, path: str | Path) -> "Forecaster": ...
```

- Baselines that have no probabilistic output still implement `predict_quantiles` (degenerate:
  all τ equal `predict`) and `predict_proba_exceed` (0/1 step from the point forecast) so the
  evaluation harness is uniform.
- `predict` returns the **median (P50)** in log10 space. Convert to linear pfu with `10**y`.

---

## 6. METRICS contract (`metrics.py`)

All functions take 1-D or 2-D `np.ndarray` of equal shape (`y_true`, `y_pred` in **log10** space
for regression; probabilities for event/probabilistic) and return `float` (or per-horizon dict
when given 2-D `[N, n_h]`). Key signatures:

- Regression: `rmse`, `mae`, `prediction_efficiency`, `skill_score(y_true, y_pred, y_ref)`,
  `r2`, `linear_correlation`, `bias`, `uncertainty_factor`.
- Event (threshold in **linear pfu** via `constants.HARSH_PFU`): `contingency_table` →
  `pod`, `far`, `pofd`, `csi`, `hss`, `tss`, `f1`, `roc_auc`, `brier_score`, `brier_skill_score`.
- Probabilistic: `pinball_loss(y_true, q_pred, tau)`, `crps`, `picp(y_true, lower, upper)`,
  `reliability_curve`.

`evaluate.py` calls these **per named horizon** and writes a tidy report
(`reports/metrics_<model>.json` / `.csv`) keyed by `horizon`.

---

## 7. SERVING payloads (`serve/`)

- `GET /health` → `{"status":"ok","model":<name>,"source":<synthetic|swpc>,"uptime_s":<float>}`.
- `GET /latest` → most recent observed `flux_e2`, `vsw`, `bz_gsm`, `kp`, `mlt`, `time`, and current
  alert flag (O(1) cache read).
- `GET /forecast` → the multi-horizon payload defined in `ARCHITECTURE.md` §i.5 (P10/P50/P90,
  `flux_p50_pfu`, `p_exceed_1000pfu`, `alert`, per horizon).
- `WS /ws` → pushes the same `/forecast` object on each 60 s refresh.

`inference.py` cache key = `hash(np.round(feature_vector, CACHE_DECIMALS))`; climatology LUT keyed
by `(doy_bin, kp_bin, vsw_bin, longitude)` returns a baseline P50/P10/P90 used as fallback when the
model artifact or a fresh feature vector is unavailable.

---

## 8. File formats & locations

| Artifact | Path | Format |
|---|---|---|
| Raw/synthetic CDFs | `data/raw/synthetic/*.cdf`, `data/raw/<dataset_id>/...` | CDF (TT2000, ISTP) |
| Per-variable cleaned series | `data/interim/*.parquet` | Parquet |
| Merged 5-min frame | `data/processed/grid_5min.parquet` | Parquet (DuckDB-queryable) |
| Window tensors | `data/processed/windows.npz` | NPZ (`np.savez_compressed`) |
| Coverage manifest | `data/manifest.csv` | CSV |
| Scalers / log floor | `models/scaler_<split>.joblib` | joblib |
| Trained model | `models/<name>.pt` / `.joblib` | torch / joblib |
| Exported inference graph | `models/<name>.onnx` | ONNX |
| Climatology LUT | `models/climatology_lut.npz` | NPZ |
| Metric reports | `reports/metrics_<model>.{json,csv}` | JSON/CSV |

---

*End of CONTRACTS.md. Builders: import names from `ps14.datasets.schema` and `ps14.constants`
rather than hard-coding strings, so a contract change is a one-line edit.*
