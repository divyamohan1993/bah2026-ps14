# 05 — CDF Data Engineering & Preprocessing (BAH-2026 PS-14)

**Scope:** Data-engineering and preprocessing layer for forecasting >2 MeV electron flux at
geostationary orbit (GEO). Covers the NASA Common Data Format (CDF), Python CDF libraries,
the exact GOES / Wind / OMNI products and variable names, the preprocessing pipeline,
L1→magnetosphere time alignment, supervised-window construction, reproducible repo
structure, and data versioning/caching.

> **Reproducibility note on dataset IDs.** Dataset IDs and variable names below are taken from
> SPDF/CDAWeb documentation pages and product guides (URLs cited). CDAWeb evolves; the live
> authority is the CDAWeb dataset index (`https://cdaweb.gsfc.nasa.gov/`), the per-letter "Notes"
> pages, and each dataset's CDF skeleton/master. **Always confirm the exact ID + variable names
> against the live portal / a downloaded CDF master before wiring them into code.** GOES-R SEISS
> in particular is distributed both via CDAWeb and via NOAA NCEI/AWS as netCDF — verify which
> form you ingest.

---

## 1. The NASA Common Data Format (CDF)

CDF is a **self-describing, machine-independent binary container** for scalar and
multidimensional scientific data, developed and maintained by NASA's Space Physics Data
Facility (SPDF) at GSFC. Almost all heliophysics / space-weather archives (CDAWeb) distribute
data as CDF (`.cdf`).

### 1.1 Structure: variables, attributes, records

- **Variables.** Two kinds:
  - **rVariables** ("r" = regular): all share the same dimensionality. Legacy; rarely used in
    modern ISTP files.
  - **zVariables** ("z"): each can have its own dimensionality. **Modern CDFs are almost all
    zVariables** — this is what you iterate over.
- **Records.** A variable is an array of *records*; for time series the record index is the time
  step. Variables can be **record-varying (RV)** (one value per time step, e.g. flux) or
  **non-record-varying (NRV)** (a single constant array, e.g. an energy-channel table).
- **Attributes.** Metadata, in two scopes:
  - **Global attributes** — file-level (mission, instrument, generation date, dataset title,
    `Logical_source`, `TEXT`, etc.).
  - **Variable attributes** — per-variable (`UNITS`, `FILLVAL`, `VALIDMIN`/`VALIDMAX`,
    `DEPEND_0`, `CATDESC`, `VAR_TYPE`, …). See §1.3.

### 1.2 Epoch (time) types — and why TT2000 matters

CDF stores time in dedicated epoch variables. There are three types, and **the leap-second
behaviour is the reason TT2000 exists**:

| Type | Storage | Time base | Resolution | Leap seconds |
|------|---------|-----------|------------|--------------|
| `CDF_EPOCH` | 8-byte float (double) | 0 AD (Gregorian) | **milliseconds** | **Undefined / broken** — a leap second is "overloaded" onto the first second of the next day, so the leap second and the following second map to the *same* value |
| `CDF_EPOCH16` | two 8-byte floats (16 bytes, complex) | 0 AD | **picoseconds** | Same limitation as `CDF_EPOCH` (no leap-second model) |
| `CDF_TIME_TT2000` | **8-byte signed integer (int64)** | **J2000** (2000-01-01 12:00 TT) | **nanoseconds** | **Correctly modelled.** Counts SI nanoseconds in Terrestrial Time; uses a leap-second table to convert TT↔UTC: `TT = TAI + 32.184 s`, `TT = UTC + ΔAT + 32.184 s`, where ΔAT is the cumulative leap seconds since 1960 |

**Why TT2000 matters for this project.** Because `CDF_EPOCH`/`EPOCH16` have no defined time
system and mishandle leap seconds, *cross-mission comparison is unreliable at the second level*
— exactly the situation here (GOES at GEO vs Wind/OMNI at L1, 11-year span crossing several
leap seconds: 2005, 2008, 2012, 2015, 2016). `CDF_TIME_TT2000` is monotonic across leap
seconds and is the SPDF-recommended standard, so:

- Prefer datasets whose `DEPEND_0`/`Epoch` is TT2000.
- Convert TT2000 → UTC with a leap-second–aware routine (cdflib does this; it ships and
  auto-updates a `CDFLeapSeconds.txt` table).
- When you finally store a uniform numpy `datetime64[ns]` grid, you are in **UTC without leap
  seconds** (numpy/pandas do not represent leap seconds). That is fine for a 5-minute-cadence
  forecasting grid — the ≤1 s leap offset is negligible relative to the cadence — **but do the
  leap-aware conversion first**, then resample, rather than treating raw epoch integers as
  seconds.

Refs: CDF leap-second page, TT2000 AGU-2011 write-up, leap-second requirements (URLs in §8).

### 1.3 ISTP/IACG metadata conventions — the attributes you must honour

CDAWeb CDFs follow the **ISTP/IACG** guidelines. The variable attributes that drive correct
ingestion:

- **`VAR_TYPE`** — classifies the variable: `data` (science values you model), `support_data`
  (e.g. the time/energy axes), `metadata`, `ignore_data`. **Filter to `VAR_TYPE == "data"`**
  to find the real measurements, and follow their `DEPEND_*` to the axes.
- **`DEPEND_0`** — names the **time variable** (the record/epoch axis) for a data variable.
  This is how you discover *which* epoch variable a flux variable is sampled on. `DEPEND_1`,
  `DEPEND_2`, … name higher dimensions (e.g. energy-channel or look-direction axes).
- **`FILLVAL`** — the value used for missing/invalid data (commonly `-1.0e31` for floats).
  **Required for time-varying variables.** Replace with `NaN` on read (§4.1).
- **`VALIDMIN` / `VALIDMAX`** — physically valid range; values outside are bad/uncalibrated.
  Their CDF datatype matches the variable's datatype.
- **`SCALEMIN` / `SCALEMAX`** — suggested display range (not for QC).
- **`UNITS`**, **`CATDESC`** (human description), **`FIELDNAM`**, **`LABLAXIS`/`LABL_PTR_1`**,
  **`DISPLAY_TYPE`** (`time_series`, `spectrogram`, …), **`SI_CONVERSION`**.
- Quality is often a separate companion variable (e.g. a `*_QUAL_FLAG` / `DQF`), linked via
  `DEPEND_0` to the same epoch — read and apply it.

**Ingestion rule of thumb:** open file → list zVariables → keep `VAR_TYPE=="data"` →
for each, read its `DEPEND_0` epoch → mask `FILLVAL` and out-of-`[VALIDMIN,VALIDMAX]` →
apply any quality-flag variable → convert epoch to `datetime64[ns]`.

Refs: ISTP guidelines, "Concise Guide to CDF", IMAP CDF requirements (URLs in §8).

---

## 2. Python CDF libraries compared

| Library | Backend / deps | Output | Notes for this project |
|---------|----------------|--------|------------------------|
| **`cdflib`** ✅ **recommended** | **Pure Python + NumPy** (no NASA C library). Optional: `astropy`, `xarray` | numpy arrays; `cdf_to_xarray` helper | Easiest to install (pip, works in CI/containers with zero system deps). Explicit, leap-aware epoch conversion (`cdfepoch.to_datetime`). Reads CDF v3, writes v3. **Use this as the core reader.** |
| `spacepy.pycdf` | **Wraps the NASA CDF C library** (must install the C lib separately) | auto-converts EPOCH/EPOCH16/TT2000 → Python `datetime` on read | Mature, full-featured, but the C-lib dependency is a deployment headache. It also **drops leap-second info** because Python `datetime` has no leap seconds. |
| `pyspedas` / `pytplot` | Higher-level; uses cdflib under the hood | "tplot" variables; `get_data()` → (times, data) | Great for *fetching* CDAWeb data and quick plots; has per-mission loaders (`pyspedas.projects.wind.mfi(...)`, GOES loaders, a generic CDAWeb loader). Heavier dependency. Good for exploration / download; you can still hand the downloaded `.cdf` to cdflib. |
| `cdasws` (CDAS web services client) | `requests` + (cdflib or spacepy) | **`xarray.Dataset` with full ISTP/SPDF metadata**, or SpacePy datamodel, or pandas | Best **programmatic download** path: server-side subsetting by dataset ID + variable + time range; returns xarray with attributes intact. Pairs naturally with cdflib. |
| `hapiclient` | HAPI REST protocol | numpy / pandas | Standardised cross-archive time-series API; good alternative to cdasws for streaming subsets. |
| `astropy`-based | `astropy.time`, `sunpy` | Tables / `TimeSeries` | Not a native CDF reader; useful downstream for time scales (UTC/TT/TAI) and for leap-aware `Time` objects if you need them. |

**Recommendation:** **`cdflib` for reading local CDFs**, **`cdasws` (or `pyspedas`) for
fetching/caching** from CDAWeb, **`xarray`/`pandas` + `pyarrow`** for the processed layer.
Pin versions in `pyproject.toml`.

### 2.1 Reading a CDF with cdflib (core pattern)

```python
import cdflib
import numpy as np
import pandas as pd

def read_cdf_variable(path: str, var: str) -> pd.Series:
    """Read one science variable from a CDF into a time-indexed pandas Series,
    masking fill/invalid values and converting the epoch to UTC datetime64."""
    cdf = cdflib.CDF(path)

    # --- discover variables -------------------------------------------------
    info = cdf.cdf_info()
    zvars = info.zVariables          # list[str] of zVariable names
    # rvars = info.rVariables        # usually empty in modern ISTP files

    # --- find the time axis via ISTP DEPEND_0 ------------------------------
    atts = cdf.varattsget(var)                  # dict of variable attributes
    epoch_var = atts["DEPEND_0"]                # name of the epoch variable
    fillval = atts.get("FILLVAL", None)
    vmin = atts.get("VALIDMIN", None)
    vmax = atts.get("VALIDMAX", None)
    units = atts.get("UNITS", "")

    # --- read data + epoch --------------------------------------------------
    data = np.asarray(cdf.varget(var), dtype="float64")
    epoch = cdf.varget(epoch_var)               # TT2000 int64 / EPOCH float

    # --- leap-aware epoch -> numpy datetime64[ns] (UTC) --------------------
    times = cdflib.cdfepoch.to_datetime(epoch)  # ndarray[datetime64[ns]]

    # --- mask fill + out-of-valid-range to NaN -----------------------------
    if fillval is not None:
        data = np.where(np.isclose(data, float(fillval)), np.nan, data)
    if vmin is not None:
        data = np.where(data < float(vmin), np.nan, data)
    if vmax is not None:
        data = np.where(data > float(vmax), np.nan, data)

    s = pd.Series(data, index=pd.DatetimeIndex(times, name="time"), name=var)
    s.attrs["units"] = units
    return s
```

Helpful cdflib calls:
- `cdf.cdf_info()` → object with `.zVariables`, `.rVariables`, `.Attributes`, `.Majority`, …
- `cdf.varinq(var)` → datatype, dimensions, record count, RV/NRV.
- `cdf.varattsget(var)` / `cdf.globalattsget()` → metadata dicts.
- `cdf.varget(var, startrec=, endrec=, starttime=, endtime=)` → array (supports record/time
  subsetting so you don't have to load 11 years at once).
- `cdflib.cdf_to_xarray(path, to_datetime=True)` → an `xarray.Dataset` with metadata, if you
  prefer xarray end-to-end.

### 2.2 Converting CDF epochs to pandas/numpy

```python
import cdflib

# Works for TT2000 (int64), CDF_EPOCH (float ms), CDF_EPOCH16 (complex ps):
dt64 = cdflib.cdfepoch.to_datetime(epoch_array)     # -> np.ndarray[datetime64[ns]]
idx  = pd.DatetimeIndex(dt64, tz=None)              # naive UTC index

# Other useful conversions:
iso   = cdflib.cdfepoch.encode(epoch_array)         # ISO-8601 UTC strings
unix  = cdflib.cdfepoch.unixtime(epoch_array)       # seconds since 1970-01-01
parts = cdflib.cdfepoch.breakdown(epoch_array)      # [Y,M,D,h,m,s,ms,us,ns]
# inverse: cdflib.cdfepoch.compute([...]) -> epoch
```

> CDF_EPOCH16's picosecond precision is truncated to nanoseconds when mapped to `datetime64[ns]`
> (numpy limit) — irrelevant at 5-min cadence.

### 2.3 Programmatic fetch with cdasws (download + cache)

```python
from cdasws import CdasWs
cdas = CdasWs()

# returns an xarray.Dataset with ISTP/SPDF metadata preserved:
status, ds = cdas.get_data(
    "OMNI_HRO_5MIN",
    ["flow_speed", "proton_density", "BZ_GSM", "F"],
    "2010-01-01T00:00:00Z", "2010-02-01T00:00:00Z",
)
```

`pyspedas` equivalent for exploration: `pyspedas.projects.wind.mfi(trange=[...])` then
`pytplot.get_data("BGSM")`.

---

## 3. The exact GOES, Wind & OMNI products

### 3.1 GOES — >2 MeV electron flux at GEO

GOES carries energetic-particle instruments at GEO. **Two eras:**

**(a) GOES-R series (16/17/18/19): SEISS / MPS-HI.** EPS was replaced by the **Space
Environment In-Situ Suite (SEISS)**. The relevant sensor is the **Magnetospheric Particle
Sensor – High energy (MPS-HI)**, which measures **50 keV–4 MeV electrons plus an integral
`>2 MeV` channel** (and 80 keV–12 MeV protons). The **`>2 MeV` integral electron channel is the
NOAA SWPC operational radiation-belt alert quantity** — this is the forecast target. SEISS
Level-1b/Level-2 products (incl. 1- and 5-min averages) are distributed by **NOAA NCEI/AWS as
netCDF** and via SPDF/CDAWeb. *Confirm the exact CDAWeb dataset ID + electron-flux variable name
(differential channels + the `>2 MeV` integral) from the live CDAWeb index / CDF master before
coding.*

**(b) Legacy GOES (8–15): EPS / EPEAD / MAGED.** Confirmed CDAWeb dataset IDs:

| CDAWeb dataset ID | Content | Notes |
|-------------------|---------|-------|
| `G0_K0_EP8`, `G8_K0_EP8`, `G9_K0_EP8`, `GOES11_K0_EP8`, … | EPS "key parameters" incl. **"GOES Electron Flux (>2 MeV) [E2]"** and **(>4 MeV) [E3]** | **Integral `>2 MeV` electron flux** — directly the target quantity for older satellites |
| `GOES13_EPS-MAGED_1MIN` / `_5MIN`, `GOES14_…`, `GOES15_…` | **MAGED** (Magnetospheric Electron Detector) differential electron flux **40–475 keV** | Differential, *lower* energy than `>2 MeV`; useful extra features, not the integral target |
| `*_EPS-EPEAD_*` (GOES-13/14/15 EPEAD) | EPEAD integral electron channels (`E1 >0.6`, `E2 >2`, `E3 >4 MeV`) | EPEAD is the renamed EPS; **`E2 >2 MeV` is the legacy target channel** |

- **Units:** integral electron flux is reported in **`pfu`** = **particles cm⁻² s⁻¹ sr⁻¹**
  (sometimes written `cm-2 s-1 sr-1`; "pfu" is also colloquially used for the >10 MeV proton
  unit, so always read `UNITS` rather than assuming).
- **Quality:** apply the dataset's quality/flag variable; legacy EPS/EPEAD electron channels are
  known to suffer proton contamination during SEP events and require the science-reprocessed
  corrections (NCEI EPEAD ATBD). Recent calibration assessments: Rodriguez et al. 2025.
- **Cadence:** native 1-min (and 5-min averaged products).

### 3.2 Wind — solar wind plasma (SWE) and IMF (MFI)

Wind sits near **L1**. Confirmed instruments / datasets:

| CDAWeb dataset ID | Instrument | Key variables | Cadence |
|-------------------|-----------|---------------|---------|
| `WI_K0_SWE` | **SWE** (Solar Wind Experiment) key params | **`Vp`** (proton bulk speed, km/s), **`Np`** (proton density, cm⁻³), `THERMAL_SPD` | ~92 s |
| `WI_H1_SWE` | SWE (definitive proton moments) | `Proton_VX/VY/VZ_GSE`, `Proton_V_nonlin` (speed), `Proton_Np_nonlin` (density), `Proton_W_nonlin` (thermal speed) | ~92 s |
| `WI_H0_SWE` | SWE VEIS electron moments | electron moments | ~6–12 s |
| `WI_H0_MFI` | **MFI** (Magnetic Field Investigation) composite | **`BGSM`** (B vector in GSM, nT; `BGSM[...,2]` = **Bz**), `BGSE`, `BF1` (\|B\|) | 1-min / 3-s / 1-hr packed |
| `WI_H1_MFI` | MFI high-res | `BGSM`, `BGSE` | 3 s / 1 min |

For an electron-flux forecaster the physically important Wind inputs are **Vsw (`Vp`)**,
**Np**, dynamic pressure (∝ Np·Vp²), and **IMF Bz (GSM)** — Bz sign/southward duration controls
substorm/convection injection and radiation-belt enhancement.

### 3.3 OMNI — the cleaned, time-shifted merged alternative (recommended for L1 inputs)

**OMNI** merges Wind + ACE (+ IMP-8, …), removes bad points, and **time-shifts every record to
the nose of Earth's bow shock** (§5). For a GEO forecaster this is the **preferred solar-wind
source** because the propagation lag is already applied and the series is gap-screened.

| Dataset ID | Cadence | Time base | Notes |
|------------|---------|-----------|-------|
| **`OMNI_HRO_5MIN`** ✅ | 5 min | **shifted to bow-shock nose** | Best match for a 5-min electron-flux grid |
| `OMNI_HRO_1MIN` | 1 min | shifted to bow-shock nose | Higher res; downsample to 5 min |
| `OMNI2_H0_MRG1HR` | 1 hr | shifted to bow-shock nose | Long-baseline / climatology |

Key OMNI variable names: **`flow_speed`** (Vsw, km/s), **`proton_density`** (cm⁻³),
**`BZ_GSM`** (nT), **`F`** (\|B\|, nT), `BX_GSE`, `BY_GSM`, `T` (proton temperature), `Pressure`
(dynamic pressure, nPa), `E` (motional E-field), plus geomagnetic indices **`SYM_H`/`AE`/`AL`**
and the GOES-derived proton fluxes. Fill value is typically `9999.99`/`999999.9` per variable —
**read `FILLVAL` per variable; do not assume.**

> Practical choice: use **OMNI_HRO_5MIN** as the solar-wind/IMF/geomagnetic driver block (lag
> already applied), and GOES SEISS/EPEAD for the **`>2 MeV` electron flux** target. Reserve raw
> Wind SWE/MFI for sensitivity studies or if you want to control the propagation lag yourself.

### 3.4 ISRO GRASP / GSAT (validation, 1–2 yr)

Used for **independent validation** of the model over the Indian GEO longitude. Likely delivered
as CDF/netCDF or CSV; ingest through the same fill/valid/epoch path. *Confirm format and
variable names from the ISRO data product documentation; treat as a held-out validation set, do
not fit scalers on it.*

---

## 4. Preprocessing pipeline (in order)

A space-weather time-series pipeline for ML. Each step is a small, testable function; persist
intermediate outputs (§7).

**0. Ingest & harmonise.** Read each CDF variable (§2.1) into a UTC-indexed Series; standardise
column names and units across satellites/eras (e.g. unify GOES-13/14/15 + GOES-16/17/18 `>2 MeV`
electron flux into one canonical column, documenting any inter-satellite calibration offset).

**1. Fill / invalid handling.** Replace `FILLVAL` and out-of-`[VALIDMIN,VALIDMAX]` with `NaN`
(per §1.3). Apply quality/flag variables. Drop obviously non-physical values (e.g. negative
flux, zero density) → `NaN`.

**2. Despiking (robust outlier removal).** Use a **Hampel filter** (rolling-median + MAD): for a
window, compute the median and the MAD; flag a point as an outlier if it deviates from the
window median by more than `n_sigma · 1.4826 · MAD` (the **1.4826** factor makes MAD a
consistent estimator of σ for Gaussian data). Replace outliers with the window median **or**
with `NaN` (then let gap logic decide). Typical: window 5–11 samples, `n_sigma = 3`. Prefer
Hampel/MAD over mean±k·std because flux is heavy-tailed and bursty. (See §4.1 snippet.)

**3. Gap detection.** Reindex onto the intended uniform grid and **measure run-lengths of
consecutive `NaN`s**. Classify: *short gaps* (≤ threshold, e.g. ≤6 samples = 30 min at 5-min
cadence) vs *long gaps*. Keep an **explicit boolean "imputed/gap" mask** as a feature/diagnostic.

**4. Interpolation (gap-aware).** **Interpolate only short gaps**; **leave long gaps as `NaN`**
(or mark windows containing them invalid). Use **linear/time** interpolation for monotone-ish
plasma params; avoid spline for spiky flux (overshoot). **Never interpolate across a long data
outage** — that fabricates dynamics the model would learn as real. Always **flag interpolated
points** so they can be excluded from loss/metrics if desired.

**5. Resample to a uniform cadence (e.g. 5 min).** Downsample higher-rate inputs (Wind ~92 s,
GOES 1 min) with an aggregation that respects the variable (mean for plasma, but consider
**aggregating flux in linear space then taking log**, and tracking max within the bin for
burst-sensitive targets). Upsample lower-rate inputs only within the short-gap rule. Use
`resample(...).asfreq()` to create the grid, then controlled `interpolate(limit=...)`.

**6. Log10 transform of flux.** Electron flux spans many decades → model `log10(flux)`. Handle
non-positives **before** the log: clip to a small floor (e.g. the instrument noise floor or
`1e-2 pfu`) or set ≤0 → `NaN`. Record the floor; never `log10(0)`.

**7. Time-alignment / merge.** Join GOES (GEO) with OMNI/Wind (L1) onto the common 5-min grid,
applying the propagation lag (§5). Inner/outer-join policy must be explicit and the gap mask
carried through.

**8. Feature engineering (optional but standard).** Lagged values, rolling means/maxima,
dynamic pressure (Np·Vsw²), southward-Bz integrals/duration, time-of-day/season,
solar-cycle/F10.7 context, recurrence (27-day) features.

**9. Chronological split → scale on TRAIN only.** Split **by time** (train < val < test) and
**fit the scaler/standardiser (and the log floor, and any clipping bounds) on the training
window only**, then `transform` val/test. Fitting on the full series leaks future statistics —
a classic, silent leakage source. Hold ISRO GRASP/GSAT entirely out.

### 4.1 Hampel despike (rolling median + MAD)

```python
import numpy as np
import pandas as pd

def hampel_filter(s: pd.Series, window: int = 7, n_sigma: float = 3.0,
                  replace: str = "nan") -> tuple[pd.Series, pd.Series]:
    """Robust despiking. Returns (filtered_series, outlier_mask).
    window: number of samples (centered). n_sigma: MAD threshold.
    replace: 'nan' -> set outliers to NaN (recommended, let gap logic handle);
             'median' -> replace with rolling median.
    """
    k = 1.4826  # MAD -> sigma consistency constant for Gaussian data
    med = s.rolling(window, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=1).median()
    threshold = n_sigma * k * mad
    diff = (s - med).abs()
    outliers = diff > threshold
    out = s.copy()
    if replace == "median":
        out[outliers] = med[outliers]
    else:
        out[outliers] = np.nan
    return out, outliers.fillna(False)
```

### 4.2 Gap-aware resample + merge onto a common grid

```python
import numpy as np
import pandas as pd

def to_uniform_grid(s: pd.Series, freq: str = "5min",
                    max_gap: int = 6, agg: str = "mean") -> pd.DataFrame:
    """Resample to `freq`, interpolate only short gaps (<= max_gap samples),
    leave long gaps as NaN, and return a companion 'imputed' flag column."""
    # 1) aggregate to the grid (downsample); asfreq() for the canonical index
    grid = getattr(s.resample(freq), agg)()        # e.g. mean within each bin
    grid = grid.asfreq(freq)                        # ensure regular, NaN where empty

    # 2) measure consecutive-NaN run lengths
    isna = grid.isna()
    grp = (~isna).cumsum()
    run_len = isna.groupby(grp).transform("sum")    # length of each NaN run

    # 3) interpolate only short gaps; long gaps stay NaN
    interp = grid.interpolate(method="time", limit=max_gap,
                              limit_area="inside")
    fill_ok = isna & (run_len <= max_gap)
    out = grid.copy()
    out[fill_ok] = interp[fill_ok]

    return pd.DataFrame({s.name: out, f"{s.name}__imputed": fill_ok.astype("int8")})


def merge_geo_l1(goes: pd.DataFrame, omni: pd.DataFrame,
                 lag: pd.Timedelta | None = None) -> pd.DataFrame:
    """Align GEO (GOES) and L1 (OMNI/Wind) on one grid.
    If using OMNI_HRO (already bow-shock-shifted) pass lag=None.
    If using raw Wind at L1, pass the propagation lag (see section 5)."""
    if lag is not None:                  # shift L1 forward in time toward Earth
        omni = omni.copy()
        omni.index = omni.index + lag
    return goes.join(omni, how="inner")   # inner-join on the common 5-min grid
```

---

## 5. Time alignment: L1 → magnetosphere, and leakage-free windows

### 5.1 Propagating solar wind from L1 to Earth

Wind/ACE sit ~1.5×10⁶ km sunward (L1); their measurements reach the magnetosphere later.

- **Flat (ballistic) propagation — simple baseline.** `Δt = Δx / Vsw`, where `Δx` is the
  Sun–Earth-line distance from the monitor to the target (bow shock / GEO) and `Vsw` is the
  measured radial speed. Recompute per sample (lag varies ~30–70 min with speed). Easy, but
  ignores front tilt.
- **OMNI phase-front technique — recommended.** OMNI HRO time-shifts each record using a
  **minimum-variance / phase-front-normal (PFN)** method (King & Papitashvili 2006; Weimer et
  al. 2002): it assumes solar-wind discontinuities are planar phase fronts and convects them to
  the **bow-shock nose** using the front orientation, not just `Δx/Vsw`. **If you use
  `OMNI_HRO_5MIN`, the lag is already applied** — set `lag=None` in the merge and add only the
  small residual bow-shock-nose → GEO transit if you want it.
- **Practical recommendation:** use **OMNI HRO** (PFN shift done for you). If you must use raw
  Wind, apply ballistic `Δx/Vsw` as a documented approximation, and treat the lag as an explicit,
  reproducible parameter.

### 5.2 Supervised windows without look-ahead leakage

Convert the merged, uniform-cadence frame into `(X, y)` with a **sliding window**:

- **Inputs:** lookback of `L` steps `[t-L+1 … t]` of the driver features (OMNI/Wind + recent
  flux).
- **Targets:** flux at one or more horizons `[t+1 … t+H]` (multi-horizon).
- **Hard rule:** every feature in a sample must be **knowable at time `t`** — never include any
  value at `> t` in `X`. Lag features must be older than the forecast horizon.
- **Split chronologically** (train earlier than val earlier than test); **fit scalers on train
  only** (§4.9). With overlapping windows, leakage can still occur at the train/val/test seams —
  use a **purge/embargo gap** of at least `L + H` steps between splits so no window straddles a
  boundary.
- **Drop windows that contain long-gap `NaN`s** (use the imputed mask from §4.2).

```python
import numpy as np
import pandas as pd

def make_supervised(df: pd.DataFrame, feature_cols: list[str],
                    target_col: str, lookback: int = 24, horizon: int = 6,
                    drop_if_nan: bool = True):
    """Build (X, y) sliding windows with no look-ahead.
    X[i] = features over [t-lookback+1 .. t]; y[i] = target over [t+1 .. t+horizon].
    X shape: (n, lookback, n_features); y shape: (n, horizon).
    """
    F = df[feature_cols].to_numpy(dtype="float32")
    tcol = df[target_col].to_numpy(dtype="float32")
    n = len(df)
    X, Y, idx = [], [], []
    for t in range(lookback - 1, n - horizon):
        x = F[t - lookback + 1 : t + 1]            # only up to and incl. t
        y = tcol[t + 1 : t + 1 + horizon]          # strictly future
        if drop_if_nan and (np.isnan(x).any() or np.isnan(y).any()):
            continue
        X.append(x); Y.append(y); idx.append(df.index[t])
    return np.asarray(X), np.asarray(Y), pd.DatetimeIndex(idx, name="t")


def chronological_split(index: pd.DatetimeIndex, train=0.7, val=0.15,
                        embargo: int = 30):
    """Index ranges for train/val/test with an embargo gap (>= lookback+horizon)
    between segments to prevent window straddling / leakage."""
    n = len(index)
    i_tr = int(n * train)
    i_va = int(n * (train + val))
    train_idx = np.arange(0, i_tr - embargo)
    val_idx   = np.arange(i_tr + embargo, i_va - embargo)
    test_idx  = np.arange(i_va + embargo, n)
    return train_idx, val_idx, test_idx
```

---

## 6. Project structure / packaging (reproducible scientific-ML repo)

Modern (2026) stack: **`src/` layout**, single **`pyproject.toml`**, **`uv`** for env/lock,
**`ruff`** (lint+format), **`pytest`**, **`mypy`** type checks, **`pre-commit`**, a thin
**Makefile**, structured **logging**, and **config in YAML** (not hard-coded). The `src/` layout
prevents accidental imports of the working tree instead of the installed package.

### 6.1 Recommended directory tree

```text
bah2026-ps14/
├── pyproject.toml              # deps, build, ruff/pytest/mypy config (single source)
├── uv.lock                     # locked, reproducible environment (uv)
├── README.md
├── Makefile                    # make data | features | train | test | lint
├── .pre-commit-config.yaml     # ruff, ruff-format, trailing-ws, end-of-file
├── .gitignore                  # ignores data/raw, data/processed, models, .venv
├── .python-version
├── dvc.yaml                    # (optional) data/feature/train pipeline stages
├── params.yaml                 # (optional, DVC) hyperparams / pipeline params
│
├── config/                     # YAML run configs (no magic numbers in code)
│   ├── data.yaml               # dataset IDs, variable names, time ranges, cadence
│   ├── preprocess.yaml         # hampel window, gap thresholds, log floor, split
│   └── model.yaml              # lookback, horizon, architecture, training
│
├── data/                       # DATA IS NOT COMMITTED (DVC/cache only)
│   ├── raw/                    # downloaded CDFs, immutable  (.cdf)
│   ├── interim/                # per-variable cleaned series (parquet)
│   ├── processed/              # merged uniform-grid tables / window arrays
│   │   ├── grid_5min.parquet
│   │   └── windows.npz         # X, y, index for the model
│   └── manifest.csv            # dataset coverage manifest (see section 7)
│
├── src/
│   └── ps14/                   # importable package
│       ├── __init__.py
│       ├── config.py           # load/validate YAML (pydantic)
│       ├── logging.py          # structured logging setup
│       ├── io/
│       │   ├── cdf_reader.py    # cdflib read + ISTP fill/valid/epoch (sec 2.1)
│       │   ├── cdaweb.py        # cdasws/pyspedas fetch + local cache
│       │   └── store.py         # parquet/zarr/npz read+write helpers
│       ├── preprocess/
│       │   ├── clean.py         # fill/valid masking, quality flags
│       │   ├── despike.py       # Hampel/MAD (sec 4.1)
│       │   ├── gaps.py          # gap detect + gap-aware interpolate (sec 4.2)
│       │   ├── resample.py      # uniform grid resampling
│       │   ├── transform.py     # log10 + floor, scalers (fit on train)
│       │   └── align.py         # L1->GEO lag + merge (sec 4.2 / 5.1)
│       ├── features/windows.py  # supervised windows + chrono split (sec 5.2)
│       ├── models/              # model definitions (kept separate from data)
│       └── cli.py               # CLI entry points (download/preprocess/...)
│
├── scripts/                    # thin runnable wrappers (call src/ps14)
│   ├── download.py
│   └── build_dataset.py
│
├── notebooks/                  # EDA only; import from src/, no business logic
│   └── 01_explore_goes_flux.ipynb
│
├── tests/                      # pytest; small fixture CDFs
│   ├── conftest.py
│   ├── test_cdf_reader.py
│   ├── test_despike.py
│   ├── test_gaps.py
│   └── test_windows.py         # asserts NO look-ahead leakage
│
├── models/                     # trained weights/checkpoints (DVC-tracked)
├── reports/                    # metrics, figures
└── docs/
    └── research/               # this document + other research notes
```

### 6.2 `pyproject.toml` skeleton (CLI + tool config)

```toml
[project]
name = "ps14"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "cdflib>=1.3",          # core CDF reader (pure python)
  "cdasws>=1.8",          # CDAWeb programmatic fetch -> xarray
  "numpy>=1.26",
  "pandas>=2.2",
  "xarray>=2024.1",
  "pyarrow>=15",          # parquet
  "scipy>=1.12",
  "scikit-learn>=1.4",    # scalers, metrics, splits
  "pyyaml>=6",
  "pydantic>=2",          # config validation
]

[project.optional-dependencies]
fetch  = ["pyspedas>=1.5", "hapiclient>=0.2"]
deep   = ["torch>=2.2"]                  # or tensorflow / lightning
store  = ["zarr>=2.17"]
dev    = ["pytest>=8", "pytest-cov", "ruff>=0.5", "mypy>=1.10", "pre-commit>=3.7"]

[project.scripts]
ps14-download   = "ps14.cli:download"
ps14-build      = "ps14.cli:build_dataset"
ps14-train      = "ps14.cli:train"

[tool.ruff]
line-length = 100
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "NPY"]

[tool.pytest.ini_options]
addopts = "-q --cov=ps14"
testpaths = ["tests"]

[tool.mypy]
python_version = "3.11"
warn_unused_ignores = true

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

Env management: `uv venv && uv sync` (locked) — `uv` replaces pip/venv/pip-tools/pyenv in one
binary. A `conda environment.yml` is an acceptable alternative if a team prefers conda (and is
sometimes easier if you ever fall back to `spacepy.pycdf`'s C library).

### 6.3 Makefile (entry points)

```make
.PHONY: setup data features train test lint
setup:    ; uv sync
data:     ; uv run ps14-download --config config/data.yaml
features: ; uv run ps14-build    --config config/preprocess.yaml
train:    ; uv run ps14-train    --config config/model.yaml
test:     ; uv run pytest
lint:     ; uv run ruff check . && uv run ruff format --check . && uv run mypy src
```

---

## 7. Data versioning & caching

- **Cache downloaded CDFs locally** under `data/raw/` keyed by `{dataset_id}/{YYYY}/...cdf`;
  make the downloader **idempotent** (skip if present + checksum OK). cdasws/pyspedas both cache;
  you can also drive direct file pulls from the SPDF tree.
- **Coverage manifest** (`data/manifest.csv`): one row per `(dataset_id, file)` with
  `start, end, n_records, sha256, fill_fraction, download_utc`. This documents exactly which
  11-year spans you actually have and their gap fraction — essential for reproducibility and for
  the train/val/test boundaries.
- **DVC (optional, recommended).** Track `data/` and `models/` with **DVC**: Git stores small
  `*.dvc`/`dvc.lock` metafiles (md5 hashes), while the large binaries live in local or cloud
  remote storage. `dvc.yaml` encodes the **download → clean → window** pipeline so
  `dvc repro` rebuilds reproducibly; `dvc.lock` pins exact data versions.
- **Processed-array storage formats:**
  - **Parquet** (`pyarrow`) for tabular, time-indexed merged frames (`grid_5min.parquet`) —
    columnar, compressed, fast partial reads, pandas/Polars native.
  - **Zarr** for large N-D arrays (e.g. spectrogram-shaped flux, or chunked multi-year tensors)
    that you slice lazily with xarray/dask.
  - **NPZ** (`np.savez_compressed`) for the final model-ready `X, y, index` window tensors —
    simplest to load in training.
  - Keep raw immutable; treat `interim/` and `processed/` as **regenerable derivatives**
    (git-ignored, DVC-tracked).

---

## 8. References (URLs)

**CDF format & epochs**
- CDF leap seconds (epoch types, TT2000): https://cdf.gsfc.nasa.gov/html/leapseconds.html
- Requirements for handling leap seconds in CDF: https://cdf.gsfc.nasa.gov/html/leapseconds_requirements.html
- CDF_TIME_TT2000 (AGU 2011 presentation, SPDF): https://spdf.gsfc.nasa.gov/pub/documents/SPDF/presentations/CDF_AGU2011_Common-Data-Format-CDF-New-Time-Variable-CDF_TIME_TT2000a.pdf
- CDF home / reading CDF: https://cdf.gsfc.nasa.gov/ and https://cdf.gsfc.nasa.gov/html/reading_CDF.html
- CDF software developers (APIs): https://cdf.gsfc.nasa.gov/userQuestions/software_developers.html

**ISTP/IACG metadata conventions**
- ISTP guidelines / variable attributes (SPDF): https://spdf.gsfc.nasa.gov/istp_guide/istp_guide.html
- ISTP guidelines → SPASE attribute mapping (DEPEND_0, FILLVAL, VALIDMIN/MAX, VAR_TYPE): https://cdaweb.gsfc.nasa.gov/pub/documents/metadata/istp-guidelines/LeeBargatze_ISTP-Guidelines-SPASE-adapt_mapping_variable_attribute.pdf
- Concise Guide to CDF (PDS): https://pds-ppi.igpp.ucla.edu/doc/cdf/Concise-Guide-to-CDF-v2.pdf
- IMAP CDF file requirements (ISTP-conformant example): https://imap-processing.readthedocs.io/en/v1.0.0/external-tools/cdf/cdf_requirements.html

**Python libraries**
- cdflib docs (read/epoch): https://cdflib.readthedocs.io/en/latest/
- cdflib CDFepoch API (`to_datetime`, `encode`, `breakdown`, `unixtime`): https://cdflib.readthedocs.io/en/latest/api/cdflib.epochs.CDFepoch.html
- SpacePy pycdf: https://spacepy.github.io/pycdf.html
- PySPEDAS load routines / CDAWeb: https://pyspedas.readthedocs.io/en/latest/cdaweb.html and https://pyspedas.readthedocs.io/en/stable/wind.html
- cdasws (CDAS web services client, xarray output): https://pypi.org/project/cdasws and https://cdaweb.gsfc.nasa.gov/WebServices/REST/jupyter/CdasWsExampleXarray.html
- CDAS RESTful web services: https://cdaweb.gsfc.nasa.gov/WebServices/REST/

**GOES / Wind / OMNI products**
- CDAWeb "Notes G" (GOES dataset IDs incl. EP8 >2 MeV, EPS-MAGED): https://cdaweb.gsfc.nasa.gov/misc/NotesG.html
- CDAWeb "Notes W" (Wind SWE/MFI dataset IDs): https://cdaweb.gsfc.nasa.gov/misc/NotesW.html
- CDAWeb "Notes O" (OMNI HRO 1/5-min, OMNI2 hourly): https://cdaweb.gsfc.nasa.gov/misc/NotesO.html
- OMNI HRO documentation (variables, time-shift): https://omniweb.gsfc.nasa.gov/html/HROdocum.html
- GOES-R SEISS (MPS-HI, >2 MeV electrons), NCEI: https://www.ncei.noaa.gov/products/goes-r-space-environment-in-situ
- GOES-R SEISS (mission page): https://www.goes-r.gov/spacesegment/seiss.html
- GOES 1–15 space-weather instruments (EPS/EPEAD/MAGED), NCEI: https://www.ncei.noaa.gov/products/goes-1-15/space-weather-instruments
- GOES-16 SEISS MPS-HI L1b ReadMe (NCEI): https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l1b/docs/GOES-16_SEISS_MPS-HI_L1b_Full_Validation_Maturity_ReadMe.pdf
- EPEAD electron science reprocessing ATBD (contamination/QC): https://www.ngdc.noaa.gov/stp/satellite/goes/doc/EPEAD_Electron_Science_Reprocessing_ATBD_v1.0.pdf
- Rodriguez et al. 2025, GOES 8–15 >0.6/>4 MeV electron flux assessment: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2024SW004228
- Bowtie inversion for GOES-16/17 MPS-HI electron channels (Boudouridis 2020): https://ui.adsabs.harvard.edu/abs/2020SpWea..1802403B/abstract
- SWPC GOES electron flux product: https://www.swpc.noaa.gov/products/goes-electron-flux

**Time-shift / propagation**
- King & Papitashvili (2006) HRO / OMNI methodology (OMNIWeb HRO doc): https://omniweb.gsfc.nasa.gov/html/HROdocum.html
- Quantitative evaluation of solar-wind time-shifting methods: https://www.researchgate.net/publication/309181806_Quantitative_evaluation_of_solar_wind_time-shifting_methods
- PRIME (NN L1→bow-shock propagation, background on MVA/PFN): https://www.frontiersin.org/journals/astronomy-and-space-sciences/articles/10.3389/fspas.2023.1250779/full

**Preprocessing / despiking / leakage**
- Hampel filter (MAD, 1.4826 factor): https://towardsdatascience.com/outlier-detection-with-hampel-filter-85ddf523c73d/
- hampel-filter (Numba) PyPI: https://pypi.org/project/hampel-filter/ ; pyhampel: https://github.com/dwervin/pyhampel
- pandas resample/asfreq/interpolate (gap handling): https://pandas.pydata.org/docs/dev/user_guide/timeseries.html
- Avoiding data leakage in time series: https://towardsdatascience.com/avoiding-data-leakage-in-timeseries-101-25ea13fcb15f/
- Data preparation without leakage (fit scaler on train only): https://machinelearningmastery.com/data-preparation-without-data-leakage/
- Time-series forecasting as supervised learning (sliding windows): https://machinelearningmastery.com/time-series-forecasting-supervised-learning/
- Purging & embargo for time-series CV: https://abouttrading.substack.com/p/purging-and-embargo-two-tricks-that

**Project structure / tooling / versioning**
- Python project setup 2026 (uv + ruff): https://www.kdnuggets.com/python-project-setup-2026-uv-ruff-ty-polars
- ML project with uv + pyproject.toml: https://www.sarahglasmacher.com/how-i-set-up-a-machine-learning-project-with-uv-and-pyproject-toml/
- pyproject.toml standard config: https://pydevtools.com/handbook/reference/pyproject.toml/
- DVC complete guide: https://www.datacamp.com/tutorial/data-version-control-dvc
