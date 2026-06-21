# Fast Platform Engineering for Real-Time >2 MeV GEO Electron-Flux Forecasting

**BAH-2026 PS-14 — Research Note 04**
**Author:** ML-platform architecture research
**Date:** 2026-06-20
**Scope:** A low-latency, "O(1)" operational pipeline that ingests space-weather streams, computes online features in amortized-constant time, runs cached fast inference, and serves a multi-horizon nowcast/forecast through an API and a live dashboard.

---

## 0. Executive summary

The user wants "the fastest platform, O(1) techniques." For a real-time forecaster of the >2 MeV electron flux at geostationary orbit (GEO), "O(1)" is best interpreted operationally as: **per-sample online updates that cost amortized constant time, plus constant-time lookups for cached results and climatology**. The end-to-end pipeline is *not* dominated by floating-point math — it is dominated by (a) polling/ingestion cadence, (b) keeping a sliding window cheap, and (c) avoiding recomputation. The recommended design:

- **Ingest** NOAA SWPC real-time JSON products (GOES integral electrons >2 MeV, ACE/DSCOVR solar-wind plasma + IMF, planetary Kp) by polling on each product's natural cadence; optionally back the historical store with a HAPI client.
- **Maintain a sliding window with a ring buffer** (O(1) append/evict) and compute **online rolling features with Welford's algorithm** (running mean/variance, O(1)) and a **monotonic deque** (rolling min/max, amortized O(1)).
- **Cache forecasts in a hash map** keyed by a hash of the input feature vector (O(1) hit), and use **precomputed lookup tables** for climatology baselines (O(1)).
- **Accelerate inference with ONNX Runtime** (recommended over TorchScript for a small tabular/sequence model on CPU), with optional int8/fp16 quantization; TensorRT only if a GPU is available.
- **Store** 11 years of multi-satellite minute-cadence history as **partitioned Parquet** queried by **DuckDB**, with **Redis (or a plain Python dict)** as the hot in-memory cache for latest values and forecasts.
- **Serve** via **FastAPI + Uvicorn** (REST + WebSocket) with **APScheduler** refreshing the forecast every minute; containerize with Docker.
- **Demo** with **Streamlit** (fastest to build, `st.fragment(run_every=...)` for live panels) — or Plotly Dash if a more bespoke, callback-driven dashboard is wanted.
- **MLOps-lite:** Hydra/YAML config + MLflow tracking/registry + deterministic seeds + a Makefile/CLI.

The whole nowcast refresh (poll → window update → feature vector → inference → cache → push) fits comfortably in a **single-digit-to-low-tens-of-milliseconds** compute budget, far under the 1-minute refresh interval; the real latency floor is the upstream data cadence (1–5 min), not our code.

---

## 1. Real-time / streaming ingestion of space-weather data

### 1.1 NOAA SWPC real-time JSON products (`services.swpc.noaa.gov`)

SWPC publishes rolling JSON files that auto-update and always include the currently active spacecraft. The directly relevant endpoints for a >2 MeV GEO electron forecaster:

**Electron flux (GOES, primary spacecraft):** files live under `https://services.swpc.noaa.gov/json/goes/primary/` with the naming convention `[measurement]-[time-range].json` (time ranges: `6-hour`, `1-day`, `3-day`, `7-day`):

- `integral-electrons-6-hour.json`, `integral-electrons-1-day.json`, `integral-electrons-3-day.json`, `integral-electrons-7-day.json` — **integral electron flux**, which carries the **>0.8 MeV and >2 MeV** integral channels (the target variable). These are small (1-day ≈ 29 KB).
- `differential-electrons-{6-hour,1-day,3-day,7-day}.json` — differential channels if you want spectral inputs.
- Mapping/metadata files in the JSON service: `instrument-sources.json` (primary/secondary instrument→satellite mapping) and `satellite-longitudes.json` (GEO longitudes), plus a `secondary/` directory mirroring the same files for the secondary GOES.

The product page states SWPC provides **5-minute-averaged integral electron flux** (electrons cm⁻² s⁻¹ sr⁻¹) at >0.8 MeV and >2 MeV. (Source: SWPC GOES Electron Flux product page; SWPC JSON service.)

**Solar wind (L1: ACE/DSCOVR real-time solar wind, RTSW):** under `https://services.swpc.noaa.gov/products/solar-wind/`:

- Plasma: `plasma-5-minute.json`, `plasma-2-hour.json`, `plasma-6-hour.json`, `plasma-1-day.json`, `plasma-3-day.json`, `plasma-7-day.json` — columns `[time_tag, density, speed, temperature]`.
- IMF / magnetic field: `mag-5-minute.json`, `mag-1-day.json`, … `mag-7-day.json` — columns `[time_tag, bx_gsm, by_gsm, bz_gsm, lon_gsm, lat_gsm, bt]`.
- These JSON files cover up to the past 7 days and automatically include data from whichever RTSW spacecraft (DSCOVR or ACE) is active. **Solar-wind speed is the dominant physical driver of >2 MeV flux** — it is the single input to SWPC's operational REFM (see §3.4) — so `speed` from the plasma feed is the key predictor.

**Geomagnetic activity (Kp):**

- `https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json` — estimated planetary K-index (3-hour cadence; updated continuously).
- `https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json` — Kp forecast.
- Supplementary: `https://services.swpc.noaa.gov/products/kyoto-dst.json` (Dst), `.../10cm-flux-30-day.json` (F10.7), `.../alerts.json` (SWPC alerts/watches/warnings for alert-status display).

> Format note: SWPC "products" JSON is a **list-of-lists** with the header row first (e.g. `[["time_tag","density","speed","temperature"], ["2026-...", ...], ...]`), whereas the `json/goes/...` files are **lists of objects**. The ingester must handle both shapes.

### 1.2 HAPI — the streaming standard

HAPI (Heliophysics Application Programmer's Interface) is a **RESTful API + streaming-format specification** for digital time-series data, recommended by COSPAR (2018) as the common access API for space-science/space-weather data. It is the right backbone for **bulk historical backfill** and for any source not exposed as an SWPC rolling JSON.

- **Five endpoints:** `/capabilities` (which stream formats: csv/binary/json), `/catalog`, `/info` (dataset parameter list/metadata), `/data` (streams a dataset over a time range), and `/about`.
- **Requests are time-bounded** via `time.min` and `time.max`, and you can request a **subset of parameters** — i.e. server-side projection, which keeps transfers small.
- **Output formats:** CSV is mandatory (RFC 4180); **binary and JSON are optional** and faster to parse — prefer binary for large pulls.
- **Adoption:** CDAWeb (`https://cdaweb.gsfc.nasa.gov/hapi`), ESA ViRES/Swarm, ESA SWE HAPI server, etc., with mature Python clients.

For this project: use **SWPC rolling JSON for the live tip** (lowest-latency, no time-range bookkeeping) and a **HAPI/CDAWeb client for the 11-year GOES history** used to train and to build climatology tables.

### 1.3 Polling cadence and the sliding window

Each stream has a natural cadence; **poll just faster than it updates** and dedup by `time_tag`:

| Stream | Native cadence | Suggested poll |
|---|---|---|
| GOES integral electrons (>2 MeV) | ~1 min raw / 5-min averaged | 60 s |
| Solar-wind plasma + IMF (RTSW) | 1 min (5-min product) | 60 s |
| Planetary Kp | 3 h (estimated, refreshed often) | 5–15 min |
| F10.7 / Dst / alerts | daily / hourly | hourly |

Maintain a **fixed-duration sliding window** (e.g. last 30 days of solar-wind speed for REFM-style features, last few hours of flux) in a **ring buffer** (§2.1). On each poll, append only *new* timestamps (idempotent), evict anything older than the window, and incrementally update the rolling features (§2.2–2.3). This makes each refresh **O(new samples)**, not O(window size).

---

## 2. O(1) online feature computation

The core "O(1) techniques" the user asked for. Each new sample is folded into the feature set in amortized constant time — no re-scan of the window.

### 2.1 Ring buffer (circular buffer) — O(1) append + O(1) evict

A ring buffer is a fixed-size array with `head`/`tail` indices that wrap modulo capacity. For a sliding window of the last *N* minutes you preallocate `N` slots; appending overwrites the oldest slot and advances the pointer.

- **Complexity:** push O(1), pop O(1), indexed access O(1); **no reallocation, no shifting**, fixed memory footprint.
- **Why it matters here:** a Python `list.pop(0)` is O(N); a ring buffer is O(1) and keeps memory at a fixed address.
- **Implementation:** `collections.deque(maxlen=N)` gives O(1) append/auto-evict out of the box. For numeric speed, use a NumPy-backed ring buffer (e.g. `numpy_ringbuffer`, or `DvG_RingBuffer`, which keeps the buffer at a fixed memory address and is reported ~60× faster than `collections.deque` for NumPy/Numba/FFT workloads because the window is exposed as a contiguous `np.ndarray` with no copy).

```
capacity = N
buf = np.empty(capacity, dtype=np.float64)
idx, count = 0, 0
def push(x):
    global idx, count
    buf[idx] = x
    idx = (idx + 1) % capacity      # O(1)
    count = min(count + 1, capacity)
```

### 2.2 Welford's algorithm — O(1) running mean & variance

Welford (1962; Knuth TAOCP Vol. 2) gives **single-pass, numerically stable** mean and variance, updated per sample in O(1). It avoids the catastrophic cancellation of the naive "sum of squares minus square of sum" because the two subtracted terms stay the same order of magnitude.

For each new value `x` with running count `n`, mean `μ`, and aggregate `M2` (sum of squared deviations):

```
n      += 1
delta   = x - μ
μ      += delta / n
delta2  = x - μ                  # uses the UPDATED mean
M2     += delta * delta2
# variance (population)  = M2 / n
# variance (sample)      = M2 / (n - 1)
```

- **Complexity:** O(1) time, O(1) memory per tracked statistic; **no stored history** required.
- **Sliding-window variant:** for a *windowed* mean/variance you either (a) keep `M2`/`μ` on the ring-buffer contents and apply the symmetric "remove-then-add" update when a sample leaves the window, or (b) for strictly-windowed variance prefer a Kahan-compensated rolling sum or recompute over the (small, fixed-N) window — still O(window) but with constant *N*. For unbounded running stats and EWMA-style features, the pure Welford update is exactly O(1).
- **Use:** running mean/std of solar-wind speed/density, flux z-scores, normalization stats for the model, anomaly flags.

### 2.3 Monotonic deque — amortized O(1) rolling min/max

To compute a **rolling max/min over a sliding window** (e.g. peak solar-wind speed in the last 24 h, max flux in the last hour) without re-scanning, maintain a **double-ended queue of indices whose values are monotonic**.

**Invariant (for rolling max):** the deque holds candidate indices whose values are **non-increasing** from front to back; the **front is always the window maximum**. Any element that is smaller than a newer element can never be a future max (it expires no later than the newer one), so it is discarded.

Per new element at index `j` with value `x`:
1. **Evict from front** any index that has fallen out of the window (`front <= j - window`).
2. **Pop from back** while `value[back] <= x` (those can never be the max again).
3. **Push** `j` to the back.
4. **Read** `value[front]` = current window maximum in O(1).

```
from collections import deque
dq = deque()           # stores indices
def push_max(j, x, window):
    while dq and dq[0] <= j - window:     # drop expired
        dq.popleft()
    while dq and values[dq[-1]] <= x:     # drop dominated
        dq.pop()
    dq.append(j)
    return values[dq[0]]                  # window max, O(1)
```

- **Complexity:** each index is pushed exactly once and popped at most once → **amortized O(1) per sample, O(n) total**; querying the extremum is O(1). Worst-case single update is O(window) but amortizes to O(1). (Use the mirror condition `>=` for rolling min.)
- This is strictly better than a heap (O(log n)) or naive rescan (O(window)).

### 2.4 Net effect

With a ring buffer + Welford + monotonic deque, the **entire feature vector updates in amortized O(1) per incoming sample**, regardless of window length. Combine with cached inference (§3.5) and the per-minute refresh is dominated by network I/O, not computation.

---

## 3. Fast model inference

### 3.1 Export path: PyTorch → ONNX

Train in PyTorch, then export with `torch.onnx.export(model, dummy_input, "model.onnx", input_names=[...], output_names=[...], dynamic_axes={"input": {0: "batch"}}, opset_version=...)`. Best practices: call `model.eval()` first (freezes dropout/BN), pass a representative `dummy_input` of correct dtype/shape (tracing records shapes), and declare `dynamic_axes` for variable batch/sequence length. PyTorch 2.x exposes the newer `dynamo=True` exporter (`torch.export.ExportedProgram`) as the recommended path.

### 3.2 ONNX Runtime (recommended for this workload)

ONNX Runtime (ORT) runs the exported graph with cross-platform graph optimizations. For a **small tabular/sequence model served single-sample on CPU** (our case), tune `SessionOptions`:

- `graph_optimization_level = ORT_ENABLE_ALL` (operator fusion, constant folding, layout). Use **offline optimization** (serialize the optimized model) to cut session-init time.
- `intra_op_num_threads` = a small fixed number (sweep 1–4); `inter_op_num_threads = 1` for **consistent low tail latency**. Default thread counts contend with NumPy/BLAS and the web server and inflate p95 — fewer, well-chosen threads reduce contention and tail latency.
- Reuse a **single warm `InferenceSession`** for the process; bind inputs to avoid re-allocation.

### 3.3 Quantization & TorchScript & TensorRT

- **Quantization:** int8 dynamic quantization and fp16 reduce latency and memory; ORT fp16 conversion has been reported at up to ~2.88× throughput vs PyTorch, with gains growing at larger batch. *Caveat:* for tiny models int8 can occasionally be **slower** than fp32 (overhead dominates), so benchmark p50/p95 before committing.
- **TorchScript:** `torch.jit.script`/`trace` + dynamic quantization is a solid, dependency-light alternative and in some transformer CPU benchmarks dynamic-quantized TorchScript was the fastest engine. It keeps you inside the PyTorch runtime (simpler if you don't want an ONNX toolchain).
- **TensorRT (GPU only):** converts the graph to an optimized engine with FP16/INT8; large speedups for sizable models, but for a small MLP the **per-kernel launch overhead (~5–15 µs each)** can dominate, so TensorRT's edge over ORT/TorchScript is small for tiny networks served one sample at a time. Recommend TensorRT only if a GPU is in the deployment and the model is non-trivial; otherwise CPU ORT is simpler and predictable.

**Recommendation:** **ONNX Runtime on CPU** as the default (portable, easy quantization/graph-fusion, predictable tail latency, trivial to containerize). Keep **TorchScript** as a fallback if avoiding the ONNX export step is preferable. Reserve **TensorRT** for a GPU deployment with a larger model.

### 3.4 Domain anchor: what the model predicts

The operational benchmark is SWPC's **Relativistic Electron Forecast Model (REFM)** — a **linear prediction filter** using **30 days of L1 solar-wind speed** to forecast the **>2 MeV daily-averaged electron fluence at GEO for +1/+2/+3 days**, with an additive flux offset to handle non-HSS events. Recent literature (LSTM models, PreMeVE 2.0, time-series foundation models) beats REFM on day-2/day-3 fluence. This matters for the platform because the **feature pipeline is mostly "30-day solar-wind-speed window" statistics** — exactly the ring-buffer + Welford + monotonic-deque pattern above — and the model itself is small, which is *why* CPU ONNX inference is more than fast enough.

### 3.5 Caching & precomputed lookup tables — the real O(1) win

- **Memoize forecasts** in a hash map keyed by a hash of the (quantized) input feature vector: `key = hash(round(features, k))`. Identical/near-identical inputs (e.g. when no new sample has arrived) return in **O(1)** with zero model calls. Back it with Redis (shared across workers/restarts) or a process-local `dict`/`functools.lru_cache` (fastest, no network hop).
- **Latest-value cache:** store the most recent flux, solar-wind, Kp, and the current multi-horizon forecast under fixed keys → **O(1)** reads for the API/dashboard.
- **Climatology lookup tables:** precompute baseline flux quantiles by (day-of-year, Kp-bin, solar-wind-speed-bin, GEO-longitude) from the 11-year history and store as a NumPy array / dict → **O(1)** baseline retrieval for anomaly scoring and as a model fallback.

### 3.6 Typical latency numbers (orientation)

- Hash-map cache hit / dict lookup: sub-microsecond (in-process).
- Redis GET (same host): ~0.1–0.5 ms (network + memory), ~100k+ ops/s.
- ONNX/TorchScript single-sample CPU inference for a small MLP/LSTM: ~sub-ms to a few ms.
- GPU kernel launch overhead: ~5–15 µs per kernel (why tiny-model GPU wins are limited).

End-to-end compute per refresh (excluding upstream cadence): **low single-digit to low tens of milliseconds**, against a 60 s refresh budget.

---

## 4. Storage for time series (11 yr × multi-satellite × minute cadence)

11 years of minute cadence ≈ 5.8M rows/series/satellite; with several GOES satellites and a handful of channels this is **tens of millions of rows** — comfortably "medium data" that fits on one node.

| Option | Model | Strengths | Best role here |
|---|---|---|---|
| **Parquet** (partitioned) | Columnar files on disk | Compression, projection/predicate pushdown, row-group + page pruning via footer stats, ubiquitous ML support | **Primary historical store** (partition by year/month/satellite) |
| **Apache Arrow** | In-memory columnar | Zero-copy, memory-mapped, faster than reading Parquet from disk; the interchange layer | In-memory frames; bridge between DuckDB/Polars/ORT |
| **DuckDB** | Embedded OLAP engine | Vectorized, multithreaded, queries Parquet/Arrow *in place* with pushdown; "point at a 10 GB Parquet and `SELECT … LIMIT 10` almost instantly"; faster than Arrow which is faster than Parquet for scans | **Analytical query engine** over the Parquet lake (training pulls, backtests, climatology) |
| **Zarr + xarray** | Chunked N-D arrays | "Parquet for arrays"; ideal for (time × lat × lon × energy) cubes; cloud-native; NOAA/NASA standard | Only if you store **multi-dimensional** spectra/grids; overkill for flat tabular flux |
| **Redis / Python dict** | In-memory KV | O(1) latest-value & forecast cache, pub/sub for fan-out | **Hot cache + WebSocket fan-out** |

**Recommendation:** **Parquet (partitioned) as the durable lake + DuckDB as the query engine + Arrow as the zero-copy in-memory layer + Redis/dict as the O(1) hot cache.** Reach for **Zarr/xarray only if** you need the full differential-energy × time × satellite hypercube (n-dimensional); for the >2 MeV scalar target and a few solar-wind channels, flat Parquet is simpler and faster to integrate with the ML stack. (Benchmarks: DuckDB > Arrow > Parquet for scan latency; DuckDB query execution orders of magnitude faster than pandas via pushdown + multithreading.)

---

## 5. Fast dataframe / numerics

- **Polars vs pandas:** Polars (Rust, Arrow-backed, multithreaded, lazy) is reported **~4–10× faster** than pandas overall, **>20×** on aggregations, and markedly faster on **time-series resample + rolling** (one benchmark: 129 s pandas vs 26 s Polars for a moving-average task). The win comes from **contiguous columnar storage + vectorization + multi-core**, vs pandas' largely single-threaded ops. Use **Polars for batch feature engineering and backtests.**
- **NumPy vectorization** for the online math: operate on the ring buffer's contiguous array, not Python loops — vectorized/columnar processing decodes and computes many values per instruction (SIMD/BLAS), which is the same reason DuckDB/Arrow are fast.
- **Numba JIT** (`@njit`) compiles the hot inner loops (custom rolling features, the monotonic-deque update, Welford) to native code, eliminating Python interpreter overhead; pairs well with a fixed-address NumPy ring buffer.
- **Rule of thumb:** vectorized + columnar + (optionally) JIT-compiled beats Python loops by 1–2 orders of magnitude; keep all per-sample work either in a tiny O(1) Python update or a Numba kernel.

---

## 6. Serving & API

**Stack: FastAPI + Uvicorn + APScheduler + Docker.**

- **FastAPI (async, ASGI)** exposes:
  - `GET /forecast` (REST) → current multi-horizon forecast with uncertainty (O(1) cache read).
  - `GET /latest` → latest flux/solar-wind/Kp + alert status (O(1)).
  - `WS /ws/stream` (WebSocket, built into FastAPI via Starlette: `await ws.accept()` then push) → live updates to the dashboard.
- **Uvicorn** is a lightning-fast ASGI server on **uvloop + httptools**; run it as the app server (Gunicorn+Uvicorn workers for multiple processes).
- **APScheduler** runs a job **every 60 s**: poll SWPC → update ring buffer/features → run cached inference → write latest + forecast to Redis → publish to WebSocket subscribers. (Alternative: `asyncio.create_task` loop.) For multi-worker deployments, use **Redis pub/sub** to fan out (workers don't share memory; adds ~1 ms/message).
- **Containerization:** a single Dockerfile (slim Python base, pinned deps, the warm ONNX session loaded at startup); optionally a second Redis container via docker-compose.

**Latency budget — 30–45 min nowcast refreshed every minute:**

| Stage | Budget |
|---|---|
| Poll SWPC JSON (network) | ~50–300 ms |
| Parse + dedup + ring-buffer/feature update (O(new)) | < 5 ms |
| Inference (ORT CPU, cache-miss) | < 5 ms |
| Cache write + WebSocket push | < 5 ms (+~1 ms Redis) |
| **Total compute** | **≈ 10–20 ms** |

So the system trivially refreshes within the 60 s window; **wall-clock freshness is bounded by SWPC's 1–5 min cadence**, and dashboard reads are O(1) cache hits.

---

## 7. Visualization / demo dashboard

Requirements: live >2 MeV flux + solar-wind panels, the **multi-horizon forecast with uncertainty bands**, and an **alert-status** indicator — built fast and looking impressive for a hackathon.

| Tool | Build speed | Real-time | Best for |
|---|---|---|---|
| **Streamlit** | Fastest (pure Python script) | `st.fragment(run_every=...)` reruns just a panel on an interval; `st.cache_data(ttl=...)` for fresh data; `streamlit-autorefresh` as backup | **Hackathon demo** — minimal code, native Plotly, instant polish |
| **Plotly Dash** | Moderate | Stateless **callbacks** + `dcc.Interval` + partial updates; better when only specific sections refresh | Bespoke, multi-widget, "flagship" dashboards |
| **Grafana** | Low-code for metrics | Built for live observability (push flux/SW to a TSDB/Prometheus, panels auto-refresh, alerting built in) | Ops monitoring / alerting, less custom forecast viz |

**Recommendation:** **Streamlit** for the hackathon. It turns a Python script into an interactive app with the least effort; **fragments with `run_every`** (Streamlit ≥1.37) let the live flux/solar-wind/forecast panels refresh on a timer **without rerunning the whole script**, and Plotly inside Streamlit gives interactive uncertainty-band charts. Wire it to the FastAPI `/forecast` + `/latest` endpoints (or read Redis directly). Use **Plotly Dash** if you want fine-grained callback control and a more production-looking multi-panel layout; use **Grafana** only if the deliverable is ops-style monitoring/alerting rather than a forecast UI. (A common pattern: Streamlit/Dash for the analytical forecast UI, Grafana for system health.)

---

## 8. Reproducibility / MLOps-lite (hackathon-appropriate)

- **Config-driven:** **Hydra + YAML** — compose configs (data sources, window sizes, model hyperparams, horizons), override on the CLI, and run sweeps. Hydra writes the resolved config per run for exact reproduction.
- **Experiment tracking + registry:** **MLflow** (self-host locally with `mlflow ui`) for params/metrics/artifacts and the **Model Registry** (stage models, load the "production" version by name at serving time). **Weights & Biases** is a slick hosted alternative if internet/account is available. Glue libraries like **HydraFlow** auto-log the Hydra config as an MLflow artifact for full reproducibility.
- **Determinism:** set all seeds (`random`, `numpy`, `torch`), `torch.use_deterministic_algorithms(True)` / `trainer.deterministic=True` (note the perf cost), and pin library + CUDA versions.
- **CLI / Makefile:** `make data` (ingest/backfill), `make train`, `make export` (→ ONNX), `make serve` (Uvicorn), `make dashboard` (Streamlit), `make eval` (backtest vs REFM). A thin `click`/`typer` CLI mirrors these.
- **Pin everything:** lockfile (uv/poetry/conda), the exact ONNX opset, and the trained-model + scaler artifacts committed to the registry so the served pipeline is byte-reproducible.

---

## 9. Annotated real-time architecture

```
                         ┌──────────────────────────────────────────────────────────┐
                         │                 UPSTREAM (space weather)                  │
                         │  SWPC JSON: integral-electrons-*.json (>2 MeV, GOES)      │
                         │            solar-wind/plasma-*.json, mag-*.json (L1)      │
                         │            noaa-planetary-k-index*.json, alerts.json      │
                         │  HAPI / CDAWeb: 11-yr GOES + SW history (binary stream)   │
                         └───────────────┬───────────────────────┬──────────────────┘
                                         │ poll 60 s (live tip)   │ batch backfill
                                         ▼                        ▼
                  ┌───────────────────────────────┐     ┌────────────────────────────┐
                  │  INGEST (APScheduler job)      │     │  HISTORICAL LAKE           │
                  │  • fetch JSON, dedup by time   │     │  Parquet (year/sat parts)  │
                  │  • normalize both JSON shapes  │────▶│  queried by DuckDB         │
                  └───────────────┬───────────────┘     │  Arrow zero-copy in mem    │
                                  │ new samples (O(new)) │  → climatology LUTs (O(1)) │
                                  ▼                      └────────────────────────────┘
                  ┌───────────────────────────────────────────────┐
                  │  ONLINE FEATURES  (amortized O(1) per sample)  │
                  │  • Ring buffer  (O(1) append/evict, 30-d SW)   │
                  │  • Welford      (O(1) running mean / variance) │
                  │  • Monotonic deque (amortized O(1) min / max)  │
                  │  • Numba/NumPy vectorized inner loops          │
                  └───────────────┬───────────────────────────────┘
                                  │ feature vector
                                  ▼
                  ┌───────────────────────────────────────────────┐
                  │  CACHED O(1) INFERENCE                         │
                  │  key = hash(round(features))                  │
                  │   ├─ HIT  → return cached forecast  (O(1))    │
                  │   └─ MISS → ONNX Runtime (CPU, fp16/int8)     │
                  │             single warm InferenceSession      │
                  │  Climatology LUT fallback (O(1))              │
                  └───────────────┬───────────────────────────────┘
                                  │ multi-horizon forecast + uncertainty + alert flag
                                  ▼
                  ┌───────────────────────────────┐   write latest/forecast (O(1))
                  │  HOT CACHE: Redis / dict       │◀───────────────────────────────┐
                  └───────────────┬───────────────┘                                 │
                                  │ pub/sub                                          │
                                  ▼                                                  │
                  ┌───────────────────────────────────────────────┐                 │
                  │  SERVING: FastAPI + Uvicorn (Docker)          │                 │
                  │  GET /forecast  GET /latest   WS /ws/stream   │─────────────────┘
                  └───────────────┬───────────────────────────────┘
                                  │ REST / WebSocket
                                  ▼
                  ┌───────────────────────────────────────────────┐
                  │  DASHBOARD: Streamlit (st.fragment run_every)  │
                  │  live flux + solar wind, forecast ± bands,     │
                  │  alert status, Plotly interactive charts       │
                  └───────────────────────────────────────────────┘
```

**Why this is "O(1)":** the per-minute hot path touches only (a) the *new* samples (O(new)), (b) constant-time online-feature updates (ring buffer, Welford, monotonic deque), (c) a constant-time cache lookup (hit) or one small model call (miss), and (d) constant-time cache writes/reads for the API and dashboard. Nothing in the serving path scales with the 11-year history or the window length.

---

## 10. Recommended tech-stack table

| Layer | Choice | Why / latency note |
|---|---|---|
| Live ingestion | SWPC rolling JSON (`services.swpc.noaa.gov`) | No time-range bookkeeping, auto-active spacecraft; poll 60 s; ~50–300 ms/fetch |
| Bulk/history | HAPI / CDAWeb client (binary) | COSPAR standard; time-bounded, parameter-subset (server-side projection) |
| Window | NumPy ring buffer (`DvG_RingBuffer`) / `deque(maxlen)` | O(1) append/evict, fixed memory, contiguous for Numba |
| Rolling mean/var | Welford | O(1)/sample, numerically stable, no stored history |
| Rolling min/max | Monotonic deque | Amortized O(1)/sample, O(1) query |
| Batch dataframe | **Polars** | 4–10× (>20× aggregations) vs pandas; multithreaded Arrow |
| Hot loops | NumPy + **Numba** | SIMD/native; kill Python-loop overhead |
| Historical store | **Parquet** (partitioned) | Pushdown + page pruning; ML-ubiquitous |
| Query engine | **DuckDB** | Vectorized OLAP over Parquet/Arrow in place; DuckDB > Arrow > Parquet scans |
| N-D cubes (optional) | Zarr + xarray | Only if storing (time×energy×sat) hypercube |
| Inference | **ONNX Runtime (CPU)** + int8/fp16 | Portable, graph fusion, predictable p95; sub-ms–few-ms; TorchScript fallback; TensorRT only on GPU |
| Forecast cache | hash-map (`dict`/`lru_cache`) + Redis | O(1) hit; dict sub-µs, Redis ~0.1–0.5 ms |
| Climatology | Precomputed LUT (NumPy/dict) | O(1) baseline lookup |
| API | **FastAPI + Uvicorn** | Async REST + WebSocket; uvloop/httptools |
| Scheduler | **APScheduler** (60 s) | Refresh forecast + push |
| Container | **Docker** (+ compose for Redis) | Reproducible deploy |
| Dashboard | **Streamlit** (`st.fragment(run_every)`) | Fastest to build, live panels, Plotly; Dash if bespoke; Grafana for ops |
| Config | **Hydra + YAML** | Composable, CLI overrides, sweeps |
| Tracking/registry | **MLflow** (or W&B) | Params/metrics/artifacts + model registry |
| Determinism | seeds + `use_deterministic_algorithms` + pinned deps | Reproducible runs |
| Orchestration | **Makefile / typer CLI** | `make ingest/train/export/serve/dashboard/eval` |

---

## References (URLs)

**SWPC real-time products**
- GOES Electron Flux (product page, channels & 5-min cadence): https://www.swpc.noaa.gov/products/goes-electron-flux and https://www.spaceweather.gov/products/goes-electron-flux
- SWPC JSON service root (products dir listing): https://services.swpc.noaa.gov/products/
- GOES primary JSON (integral/differential electrons): https://services.swpc.noaa.gov/json/goes/primary/
- Solar wind JSON dir (plasma/mag): https://services.swpc.noaa.gov/products/solar-wind/
- `plasma-1-day.json`: https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json
- `mag-1-day.json`: https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json
- Planetary K-index: https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json ; forecast: https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json
- Real-Time Solar Wind (RTSW, DSCOVR/ACE): https://www.spaceweather.gov/products/real-time-solar-wind
- Relativistic Electron Forecast Model (REFM, the operational baseline): https://www.swpc.noaa.gov/products/relativistic-electron-forecast-model

**HAPI**
- HAPI home: https://hapi-server.org/
- HAPI data-access spec (v3.2.0): https://github.com/hapi-server/data-specification/blob/master/hapi-3.2.0/HAPI-data-access-spec-3.2.0.md
- CDAWeb HAPI server: https://cdaweb.gsfc.nasa.gov/hapi
- Weigel et al. 2021, "HAPI: An API Standard for Accessing Heliophysics Time Series Data," JGR Space Physics: https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2021JA029534

**O(1) online algorithms**
- Sliding window min/max (monotonic deque), Nayuki: https://www.nayuki.io/page/sliding-window-minimum-maximum-algorithm
- Sliding Window Maximum (monotonic deque, amortized O(1)): https://algomaster.io/learn/dsa/sliding-window-maximum and https://algo.monster/liteproblems/239
- Welford's method (numerically stable online mean/var), Sachs: https://www.embeddedrelated.com/showarticle/785.php
- Welford derivation/update: https://changyaochen.github.io/welford/
- NumPy ring buffer (fixed-address, ~60× deque): https://github.com/Dennis-van-Gils/python-dvg-ringbuffer ; https://pypi.org/project/numpy-ringbuffer/0.2.0
- `collections.deque` (O(1) ends): https://www.pythoncentral.io/understanding-pythons-deque/

**Inference acceleration**
- `torch.onnx` export (dynamic_axes, opset, dynamo): https://docs.pytorch.org/docs/2.9/onnx.html
- ONNX Runtime graph optimizations: https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html
- ONNX Runtime perf-tuning (threads, p50/p95): https://onnxruntime.ai/docs/performance/ and https://medium.com/@Modexa/8-onnx-runtime-tricks-for-low-latency-python-inference-baee6e535445
- ORT fp16 ~2.88× throughput (Microsoft): https://opensource.microsoft.com/blog/2022/04/19/scaling-up-pytorch-inference-serving-billions-of-daily-nlp-inferences-with-onnx-runtime/
- ONNX vs TorchScript (engine comparison): https://medium.com/@2nick2patel2/onnx-vs-torchscript-pick-the-faster-lane-9dc07cc2637b
- int8 sometimes slower than fp32 (ORT issue): https://github.com/microsoft/onnxruntime/issues/10135
- TensorRT (kernel-launch overhead ~5–15 µs for small ops): https://developer.nvidia.com/blog/adaptive-inference-in-nvidia-tensorrt-for-rtx-enables-automatic-optimization/ ; https://www.baseten.co/blog/high-performance-ml-inference-with-nvidia-tensorrt/

**Storage & dataframes**
- Querying Parquet with millisecond latency (pushdown/pruning/late materialization), Apache Arrow: https://arrow.apache.org/blog/2022/12/26/querying-parquet-with-millisecond-latency/
- DuckDB vs dataframe libs benchmark: https://www.codecentric.de/en/knowledge-hub/blog/duckdb-vs-dataframe-libraries
- DuckDB > Arrow > Parquet scan ordering: https://www.christophenicault.com/post/large_dataframe_arrow_duckdb/
- Zarr ("Parquet for arrays") / xarray: https://www.earthmover.io/blog/what-is-zarr/ ; https://tutorial.xarray.dev/intermediate/intro-to-zarr.html
- Zarr vs Parquet (NWM benchmark): https://element84.com/software-engineering/benchmarking-zarr-and-parquet-data-retrieval-using-the-national-water-model-nwm-in-a-cloud-native-environment/
- Polars vs pandas benchmarks: https://www.statology.org/pandas-vs-polars-performance-benchmarks-for-common-data-operations/ ; https://www.databricks.com/blog/polars-vs-pandas
- Redis vs in-memory cache (latency): https://blog.nashtechglobal.com/redis-cache-vs-in-memory-cache-when-to-use-what/ ; https://realpython.com/python-redis/

**Serving & dashboard**
- FastAPI WebSocket + background tasks: https://websocket.org/guides/frameworks/fastapi/ ; https://hexshift.medium.com/implementing-background-tasks-with-websockets-in-fastapi-034cdf803430
- WebSockets at scale (Uvicorn workers + Redis pub/sub, ~1 ms/msg): https://medium.com/@bhagyarana80/websockets-at-scale-with-fastapi-and-uvicorn-workers-building-real-time-systems-that-dont-break-ac2dada6cae9
- Schedule tasks with FastAPI (APScheduler): https://sentry.io/answers/schedule-tasks-with-fastapi/
- Streamlit fragments / `run_every`: https://docs.streamlit.io/develop/concepts/architecture/fragments ; https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment
- Streamlit + Plotly real-time dashboard: https://workingoutjournal.com/building-real-time-dashboards-with-streamlit-and-plotly/
- Streamlit vs Dash (2025/2026): https://www.squadbase.dev/en/blog/streamlit-vs-dash-in-2025-comparing-data-app-frameworks ; https://reflex.dev/blog/streamlit-vs-dash-python-dashboards/
- Streamlit vs Dash vs Grafana (showdown): https://medium.com/@anfaalbhatti71/data-visualization-showdown-exploring-plotly-dash-powerbi-streamlit-and-grafana-for-dashboard-31315115f2c5

**MLOps-lite & domain models**
- Hydra + MLflow (HydraFlow): https://github.com/daizutabi/hydraflow ; https://pypi.org/project/hydraflow/
- Lightning + Hydra template (config-driven ML): https://github.com/ashleve/lightning-hydra-template
- MLflow + Hydra sweeps: https://towardsdatascience.com/hyperparameters-tuning-with-mlflow-and-hydra-sweeps-7253d97d7897/
- LSTM beats REFM for ≥2 MeV daily fluence (MDPI 2023): https://www.mdpi.com/2072-4292/15/10/2538
- ML forecasting of MeV electron flux (foundation model, 2026): https://arxiv.org/html/2605.15752
- REFM vs SNB3GEO comparison: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4995643/
