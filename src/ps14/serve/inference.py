"""Low-latency inference core: O(1) online features, cache, climatology LUT, predictor.

The real O(1) win on the serving hot path (R4 §2-3, ARCHITECTURE.md (f)):

* :class:`OnlineFeatureState` — folds each incoming sample into a sliding-window feature
  vector in amortized O(1) using the :mod:`ps14.features.online` primitives (RingBuffer,
  Welford, MonotonicDeque) — no re-scan of the window.
* :class:`ForecastCache` — a hash-map memoizer keyed by the *quantized* feature vector;
  identical / near-identical inputs return in O(1) with zero model calls. Bounded size.
* :class:`ClimatologyLUT` — a precomputed per-local-time lookup table giving an O(1)
  climatology baseline / fallback forecast.
* :class:`Predictor` — wraps a model: a warm ONNX Runtime session (lazy ``onnxruntime``)
  if a ``.onnx`` exists, else a :class:`ps14.models.base.Forecaster` loaded from disk, else
  the climatology LUT. :meth:`Predictor.predict` checks the cache first (O(1)) then runs
  inference then caches.
* :class:`ForecastPayload` — the typed serving payload (CONTRACTS.md §7 / ARCHITECTURE
  (i.5)).

Heavy / optional dependencies (``onnxruntime``, ``torch``, ``onnx``) are imported lazily
inside the functions that need them so ``import ps14`` works with only the core deps.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ps14.constants import (
    HARSH_PFU,
    HORIZON_LEAD_MINUTES,
    HORIZON_NAMES,
    LOG_HARSH,
    QUANTILES,
)
from ps14.datasets import schema
from ps14.features.online import MonotonicDeque, RingBuffer, Welford

# Channel index of the autoregressive target within the encoder feature vector.
_TARGET_CHANNEL: int = schema.FEATURE_COLUMNS.index(schema.TARGET)
_VSW_CHANNEL: int = schema.FEATURE_COLUMNS.index("vsw")
_KP_CHANNEL: int = schema.FEATURE_COLUMNS.index("kp")

# Expose for convenience / backwards-compatibility with the scaffold.
HORIZONS: list[str] = HORIZON_NAMES
LOG_THRESHOLD: float = LOG_HARSH


# ======================================================================================
# Typed payload (CONTRACTS.md §7 / ARCHITECTURE.md (i.5))
# ======================================================================================
class HorizonForecast(BaseModel):
    """Per-horizon forecast block (quantiles in log10 pfu + exceedance + alert)."""

    lead_min: int = Field(..., description="Lead time of this horizon in minutes.")
    p10: float = Field(..., description="10th-percentile log10 flux.")
    p50: float = Field(..., description="Median (50th-percentile) log10 flux.")
    p90: float = Field(..., description="90th-percentile log10 flux.")
    flux_p50_pfu: float = Field(..., description="Median flux in linear pfu (10**p50).")
    p_exceed_1000pfu: float = Field(
        ..., ge=0.0, le=1.0, description="Calibrated P(flux >= 1000 pfu)."
    )
    alert: bool = Field(..., description="True if P(exceed) >= 0.5 (harsh-alert flag).")


class ForecastPayload(BaseModel):
    """Multi-horizon serving payload (CONTRACTS.md §7, ARCHITECTURE.md (i.5)).

    Matches the documented ``/forecast`` JSON: top-level ``issued_utc``, ``satellite``,
    a ``horizons`` map keyed by ``HORIZON_NAMES`` with P10/P50/P90 (log10 pfu),
    ``flux_p50_pfu`` (linear), ``p_exceed_1000pfu`` and ``alert``, plus ``threshold_pfu``,
    ``model`` and ``source``. ``latency_ms`` records the compute time for the refresh.
    """

    issued_utc: str = Field(..., description="ISO-8601 UTC time the forecast was issued.")
    satellite: str = Field("synthetic", description="Source satellite identifier.")
    horizons: dict[str, HorizonForecast] = Field(
        ..., description="Per-horizon forecast keyed by HORIZON_NAMES."
    )
    threshold_pfu: float = Field(HARSH_PFU, description="Harsh alert threshold (linear pfu).")
    model: str = Field("climatology", description="Model that produced the forecast.")
    source: str = Field("synthetic", description="Data source: 'synthetic' or 'swpc'.")
    latency_ms: float = Field(0.0, ge=0.0, description="Compute latency of this forecast (ms).")

    @property
    def model_name(self) -> str:
        """Alias for :attr:`model` (the task interface names it ``model_name``)."""
        return self.model

    @property
    def timestamp(self) -> str:
        """Alias for :attr:`issued_utc`."""
        return self.issued_utc


def _utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string with a trailing ``Z``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_payload(
    quantiles_log: dict[float, np.ndarray],
    proba_exceed: np.ndarray,
    *,
    satellite: str = "synthetic",
    model_name: str = "tft-dualhead-v1",
    source: str = "synthetic",
    latency_ms: float = 0.0,
    issued_utc: str | None = None,
) -> ForecastPayload:
    """Assemble the ``/forecast`` payload from quantile + exceedance arrays.

    Parameters
    ----------
    quantiles_log:
        ``{tau: array}`` of log10-flux quantiles. Each array is shape ``[n_h]`` (a single
        sample) or ``[N, n_h]`` (the last row is used). Must contain the 0.1/0.5/0.9 keys.
    proba_exceed:
        ``P(flux >= HARSH_PFU)`` per horizon, shape ``[n_h]`` or ``[N, n_h]``.
    satellite, model_name, source, latency_ms:
        Metadata for the payload header.
    issued_utc:
        Override the issue time (defaults to ``now`` in UTC).

    Returns
    -------
    ForecastPayload
        The validated multi-horizon payload. Converts log10 P50 to linear pfu, thresholds
        ``p_exceed`` at 0.5 for the boolean ``alert`` (CONTRACTS.md §7).
    """

    def _row(arr: np.ndarray) -> np.ndarray:
        a = np.atleast_2d(np.asarray(arr, dtype="float64"))
        return a[-1]  # most-recent sample if a batch was supplied

    p10 = _row(quantiles_log[0.1])
    p50 = _row(quantiles_log[0.5])
    p90 = _row(quantiles_log[0.9])
    pe = _row(proba_exceed)

    horizons: dict[str, HorizonForecast] = {}
    for i, name in enumerate(HORIZON_NAMES):
        pe_i = float(np.clip(pe[i], 0.0, 1.0))
        horizons[name] = HorizonForecast(
            lead_min=HORIZON_LEAD_MINUTES[name],
            p10=float(p10[i]),
            p50=float(p50[i]),
            p90=float(p90[i]),
            flux_p50_pfu=float(10.0 ** p50[i]),
            p_exceed_1000pfu=pe_i,
            alert=bool(pe_i >= 0.5),
        )

    return ForecastPayload(
        issued_utc=issued_utc or _utc_now_iso(),
        satellite=satellite,
        horizons=horizons,
        threshold_pfu=HARSH_PFU,
        model=model_name,
        source=source,
        latency_ms=float(latency_ms),
    )


# ======================================================================================
# O(1) online feature state
# ======================================================================================
class OnlineFeatureState:
    """Sliding-window online feature accumulator with amortized O(1) updates.

    Maintains, for the autoregressive target log-flux, a :class:`RingBuffer` (the raw
    window), a :class:`Welford` accumulator (running mean / variance / z-score) and two
    :class:`MonotonicDeque` instances (rolling min / max). Each :meth:`update` folds one
    new sample into all of them in amortized O(1) and emits a fixed-length feature vector
    aligned to :data:`ps14.datasets.schema.FEATURE_COLUMNS` channel order.

    The vector is allocated once and reused (allocation-light hot path); the per-channel
    values that the offline pipeline would compute from the full window are approximated
    online where a cheap O(1) statistic exists, and otherwise carried forward from the
    most recent observed sample.

    Parameters
    ----------
    window:
        Sliding-window length in 5-min samples (defaults to the rolling-feature span).
    warmup:
        Minimum number of samples before :meth:`is_warm` is True.
    """

    #: Sample-dict keys that map directly onto a feature channel (others are derived).
    _DIRECT_KEYS = (
        "log_flux_e2",
        "log_flux_seed",
        "vsw",
        "density",
        "pdyn",
        "bz_gsm",
        "bt",
        "ae",
        "al",
        "kp",
        "sym_h",
        "f107",
    )

    def __init__(self, window: int = 72, warmup: int = 1) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = int(window)
        self.warmup = max(1, int(warmup))
        self.n_features = len(schema.FEATURE_COLUMNS)

        # O(1) accumulators for the autoregressive target log-flux.
        self._flux_buf = RingBuffer(self.window, dtype="float64")
        self._flux_welford = Welford()
        self._flux_max = MonotonicDeque(self.window, kind="max")
        self._flux_min = MonotonicDeque(self.window, kind="min")

        # Reusable feature vector + the last raw sample (carry-forward for derived cols).
        self._vec = np.zeros(self.n_features, dtype="float32")
        self._last: dict[str, float] = {}
        self._count = 0

        # Precompute the channel indices we write so the hot path is a tight loop.
        cols = schema.FEATURE_COLUMNS
        self._idx = {c: cols.index(c) for c in cols}

    def is_warm(self) -> bool:
        """True once at least ``warmup`` samples have been folded in."""
        return self._count >= self.warmup

    @property
    def count(self) -> int:
        """Number of samples folded in so far."""
        return self._count

    def update(self, sample: dict[str, float]) -> np.ndarray:
        """Fold one sample into the window and return the feature vector (amortized O(1)).

        Parameters
        ----------
        sample:
            Mapping with at least ``log_flux_e2`` (and ideally the other base channels in
            :attr:`_DIRECT_KEYS`). Missing keys carry forward their last observed value.

        Returns
        -------
        np.ndarray
            The reused float32 feature vector of length ``len(FEATURE_COLUMNS)``. Callers
            that retain it across updates should copy it.
        """
        self._count += 1

        # 1) O(1) accumulator updates for the autoregressive target.
        flux = float(sample.get("log_flux_e2", self._last.get("log_flux_e2", LOG_HARSH - 1.0)))
        if self._flux_buf.is_full:
            self._flux_welford.remove(self._flux_buf[0])  # evict oldest from running stats
        self._flux_buf.append(flux)
        self._flux_welford.update(flux)
        roll_max = self._flux_max.push(flux)
        roll_min = self._flux_min.push(flux)

        # 2) Carry-forward the most recent observed value for each base channel.
        for key in self._DIRECT_KEYS:
            if key in sample and sample[key] == sample[key]:  # present and not NaN
                self._last[key] = float(sample[key])

        vec = self._vec
        idx = self._idx

        # 3) Observed-past base channels (direct copy / carry-forward).
        for key in self._DIRECT_KEYS:
            vec[idx[key]] = self._last.get(key, 0.0)

        # 4) Lag features approximated from the ring buffer (O(1) indexed reads).
        roll_mean = self._flux_welford.mean
        vec[idx["log_flux_e2_lag_1"]] = self._buf_lag(1, flux)
        vec[idx["log_flux_e2_lag_6"]] = self._buf_lag(6, flux)
        vec[idx["log_flux_e2_lag_72"]] = self._buf_lag(72, flux)
        vec[idx["log_flux_e2_lag_288"]] = self._buf_lag(288, flux)
        vec[idx["log_flux_e2_lag_576"]] = self._buf_lag(576, flux)
        vsw_now = self._last.get("vsw", 0.0)
        vec[idx["vsw_lag_288"]] = vsw_now
        vec[idx["vsw_lag_576"]] = vsw_now

        # 5) Rolling features from the O(1) accumulators.
        vec[idx["log_flux_e2_rollmean_12"]] = roll_mean
        vec[idx["log_flux_e2_rollmean_72"]] = roll_mean
        vec[idx["log_flux_e2_rollmean_288"]] = roll_mean
        std = self._flux_welford.std
        std = std if std == std else 0.0  # NaN-safe
        vec[idx["log_flux_e2_rollstd_72"]] = std
        vec[idx["log_flux_e2_rollstd_288"]] = std
        vec[idx["log_flux_e2_rollmin_72"]] = roll_min
        vec[idx["log_flux_e2_rollmax_72"]] = roll_max
        vec[idx["vsw_rollmean_576"]] = vsw_now
        vec[idx["ae_rollmean_288"]] = self._last.get("ae", 0.0)

        # 6) Coupling functions: use provided values, else a cheap derived estimate.
        vec[idx["vbs"]] = float(sample.get("vbs", self._vbs()))
        vec[idx["newell"]] = float(sample.get("newell", 0.0))
        vec[idx["epsilon"]] = float(sample.get("epsilon", 0.0))
        vec[idx["clock_angle"]] = float(sample.get("clock_angle", 0.0))
        vec[idx["r0_standoff"]] = float(sample.get("r0_standoff", 0.0))

        return vec

    def _buf_lag(self, steps: int, default: float) -> float:
        """Value ``steps`` samples back in the ring buffer (default if not yet available)."""
        if len(self._flux_buf) > steps:
            return self._flux_buf[-1 - steps]
        return default

    def _vbs(self) -> float:
        """Half-wave-rectified dawn-dusk E-field estimate ``v * max(-Bz, 0)`` (mV/m-ish)."""
        vsw = self._last.get("vsw", 0.0)
        bz = self._last.get("bz_gsm", 0.0)
        return vsw * max(-bz, 0.0) * 1e-3


# ======================================================================================
# O(1) forecast cache
# ======================================================================================
class ForecastCache:
    """Bounded hash-map memoizer for forecasts keyed by a quantized feature vector.

    The cache key is ``hash(np.round(feature_vector, decimals).tobytes())`` — a constant
    work hash of a fixed-length vector — so :meth:`get` / :meth:`put` are **O(1)** average
    (Python ``dict`` lookups). Near-identical inputs (e.g. when no new sample has arrived)
    collapse to the same key and return with zero model calls (R4 §3.5).

    An :class:`~collections.OrderedDict` gives LRU-ish eviction: on overflow the
    least-recently-used entry is popped (still O(1) per operation).

    Parameters
    ----------
    capacity:
        Maximum number of cached forecasts (``<= 0`` means unbounded).
    decimals:
        Rounding precision applied to the feature vector before hashing.
    """

    def __init__(self, capacity: int = 4096, decimals: int = 3) -> None:
        self.capacity = int(capacity)
        self.decimals = int(decimals)
        self._store: OrderedDict[int, ForecastPayload] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key_for(features: np.ndarray, decimals: int) -> int:
        """Compute the O(1) cache key for a feature vector (rounded then hashed)."""
        rounded = np.round(np.asarray(features, dtype="float64"), decimals)
        return hash(rounded.tobytes())

    def get(self, features: np.ndarray) -> ForecastPayload | None:
        """Return the cached payload for ``features`` (O(1)), or None on a miss."""
        k = self.key_for(features, self.decimals)
        payload = self._store.get(k)
        if payload is None:
            self.misses += 1
            return None
        self._store.move_to_end(k)  # mark most-recently-used (O(1))
        self.hits += 1
        return payload

    def put(self, features: np.ndarray, payload: ForecastPayload) -> None:
        """Insert / refresh a forecast for ``features`` (O(1)); evicts LRU on overflow."""
        k = self.key_for(features, self.decimals)
        self._store[k] = payload
        self._store.move_to_end(k)
        if self.capacity > 0:
            while len(self._store) > self.capacity:
                self._store.popitem(last=False)  # evict least-recently-used

    def clear(self) -> None:
        """Empty the cache and reset hit/miss counters."""
        self._store.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._store)


# ======================================================================================
# Climatology lookup table (O(1) fallback)
# ======================================================================================
class ClimatologyLUT:
    """Precomputed climatology lookup table for an O(1) baseline / fallback forecast.

    Keyed by binned context ``(doy_bin, kp_bin, vsw_bin, longitude)`` per CONTRACTS.md §7;
    the simplest useful variant (and the default when no history is available) is a
    per-local-time-bin table of ``(p10, p50, p90)`` log10-flux quantiles. Lookups digitize
    the inputs into bins and index the table — constant work, no scan (R4 §3.5).

    Parameters
    ----------
    table:
        Array of shape ``[n_bins, 3]`` of ``(p10, p50, p90)`` log10-flux per local-time
        bin. If None, a smooth diurnal default is synthesized.
    n_lt_bins:
        Number of local-time bins (the diurnal cycle index).
    edges:
        Optional bin-edge metadata (for the full multi-key variant).
    """

    def __init__(
        self,
        table: np.ndarray | None = None,
        *,
        n_lt_bins: int = 24,
        edges: dict[str, Any] | None = None,
    ) -> None:
        self.n_lt_bins = int(n_lt_bins)
        self._edges = edges or {}
        if table is None:
            table = self._default_table(self.n_lt_bins)
        self._table = np.asarray(table, dtype="float64")
        if self._table.ndim != 2 or self._table.shape[1] != 3:
            raise ValueError("climatology table must have shape [n_bins, 3] (p10,p50,p90)")

    @staticmethod
    def _default_table(n_lt_bins: int) -> np.ndarray:
        """Synthesize a smooth diurnal climatology (max near local noon, min at midnight).

        The >2 MeV flux at GEO has a ~1-order-of-magnitude diurnal cycle peaking near local
        noon (R1 §0). We centre P50 around ~2.4 dex (≈250 pfu) with a 0.5-dex amplitude and
        a fixed ±0.4-dex P10/P90 spread — a physically plausible baseline below the
        1000-pfu (3.0-dex) alert line.
        """
        lt = np.arange(n_lt_bins) / n_lt_bins  # 0..1 over the day
        # Cosine peaking at local noon (lt = 0.5).
        diurnal = 0.5 * np.cos(2.0 * math.pi * (lt - 0.5))  # +0.5 at noon, -0.5 at midnight
        p50 = 2.4 + diurnal
        p10 = p50 - 0.4
        p90 = p50 + 0.4
        return np.column_stack([p10, p50, p90])

    @classmethod
    def load(cls, path: str | Path) -> ClimatologyLUT:
        """Load a LUT from ``models/climatology_lut.npz`` (falls back to default on absence)."""
        path = Path(path)
        if not path.exists():
            return cls()
        with np.load(path, allow_pickle=True) as data:
            table = data["table"] if "table" in data else None
            n_lt_bins = int(data["n_lt_bins"]) if "n_lt_bins" in data else 24
            edges = data["edges"].item() if "edges" in data else None
        return cls(table=table, n_lt_bins=n_lt_bins, edges=edges)

    def save(self, path: str | Path) -> None:
        """Persist the LUT to ``path`` (NPZ; CONTRACTS.md §8)."""
        np.savez_compressed(path, table=self._table, n_lt_bins=self.n_lt_bins)

    def _lt_bin(self, mlt: float) -> int:
        """Map a magnetic-local-time (0-24 h) to a table bin index (O(1))."""
        frac = (float(mlt) % 24.0) / 24.0
        return min(self.n_lt_bins - 1, int(frac * self.n_lt_bins))

    def lookup(
        self, *, mlt: float = 12.0, doy: float = 0.0, kp: float = 0.0, vsw: float = 0.0
    ) -> dict[str, float]:
        """Return baseline ``{p10, p50, p90}`` (log10 flux) for the context (O(1)).

        Only ``mlt`` is used by the default diurnal table; ``doy``/``kp``/``vsw`` are
        accepted for the full multi-key variant and ignored otherwise.
        """
        row = self._table[self._lt_bin(mlt)]
        return {"p10": float(row[0]), "p50": float(row[1]), "p90": float(row[2])}

    def forecast(
        self,
        *,
        mlt: float = 12.0,
        doy: float = 0.0,
        kp: float = 0.0,
        vsw: float = 0.0,
        satellite: str = "synthetic",
        source: str = "synthetic",
        latency_ms: float = 0.0,
    ) -> ForecastPayload:
        """Build a full :class:`ForecastPayload` from the climatology baseline (O(1))."""
        base = self.lookup(mlt=mlt, doy=doy, kp=kp, vsw=vsw)
        n_h = len(HORIZON_NAMES)
        p10 = np.full(n_h, base["p10"])
        p50 = np.full(n_h, base["p50"])
        p90 = np.full(n_h, base["p90"])
        # Smooth exceedance estimate from the baseline P50 vs the log-threshold.
        spread = max(base["p90"] - base["p10"], 1e-6)
        proba = 1.0 / (1.0 + np.exp(-(p50 - LOG_HARSH) / (0.5 * spread)))
        return build_payload(
            {0.1: p10, 0.5: p50, 0.9: p90},
            proba,
            satellite=satellite,
            model_name="climatology",
            source=source,
            latency_ms=latency_ms,
        )


# ======================================================================================
# Predictor (ONNX -> Forecaster -> climatology fallback) with O(1) cache
# ======================================================================================
class Predictor:
    """Wrap a model behind an O(1) forecast cache and a climatology fallback.

    Resolution order at construction time:

    1. a warm ``onnxruntime`` session if ``onnx_path`` exists (lazy ``onnxruntime`` import);
    2. else a :class:`ps14.models.base.Forecaster` loaded from ``model_path``;
    3. else (or on any load failure) the :class:`ClimatologyLUT` baseline.

    :meth:`predict` first checks the :class:`ForecastCache` (O(1)); on a miss it runs the
    resolved backend, builds a :class:`ForecastPayload`, caches it and returns it.

    Parameters
    ----------
    onnx_path, model_path:
        Optional artifact paths. Missing / unusable artifacts fall through to the next tier.
    lut:
        Climatology fallback (a default diurnal LUT is created if omitted).
    cache:
        Forecast cache (a default bounded cache is created if omitted).
    threads_intra, threads_inter:
        ONNX Runtime thread settings (``inter=1`` for low tail latency, R4 §3.2).
    satellite, source:
        Payload metadata.
    """

    def __init__(
        self,
        *,
        onnx_path: str | Path | None = None,
        model_path: str | Path | None = None,
        lut: ClimatologyLUT | None = None,
        cache: ForecastCache | None = None,
        threads_intra: int = 2,
        threads_inter: int = 1,
        satellite: str = "synthetic",
        source: str = "synthetic",
    ) -> None:
        self.onnx_path = Path(onnx_path) if onnx_path else None
        self.model_path = Path(model_path) if model_path else None
        self.lut = lut if lut is not None else ClimatologyLUT()
        self.cache = cache if cache is not None else ForecastCache()
        self.threads_intra = int(threads_intra)
        self.threads_inter = int(threads_inter)
        self.satellite = satellite
        self.source = source

        self._session: Any | None = None
        self._model: Any | None = None
        self.backend: str = "climatology"
        self.model_name: str = "climatology"

        self._resolve_backend()

    # ---- backend resolution -----------------------------------------------------------
    def _resolve_backend(self) -> None:
        """Pick the best available backend (ONNX -> Forecaster -> climatology)."""
        if self.onnx_path is not None and self.onnx_path.exists():
            try:
                self._load_onnx_session()
                self.backend = "onnx"
                self.model_name = self.onnx_path.stem
                return
            except Exception:  # noqa: BLE001 - any ORT/load failure falls through
                self._session = None
        if self.model_path is not None and self.model_path.exists():
            try:
                self._model = self._load_forecaster(self.model_path)
                self.backend = "forecaster"
                self.model_name = getattr(self._model, "name", self.model_path.stem)
                return
            except Exception:  # noqa: BLE001 - fall back to climatology
                self._model = None
        self.backend = "climatology"
        self.model_name = "climatology"

    def _load_onnx_session(self) -> None:
        """Create a single warm ``onnxruntime.InferenceSession`` (lazy import, R4 §3.2)."""
        import onnxruntime as ort  # lazy: optional [dl] extra

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = self.threads_intra
        opts.inter_op_num_threads = self.threads_inter
        self._session = ort.InferenceSession(
            str(self.onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )

    @staticmethod
    def _load_forecaster(path: Path) -> Any:
        """Load a persisted :class:`Forecaster` (tries known subclasses)."""
        from ps14.models.baselines import (
            Climatology,
            LightGBMForecaster,
            Persistence,
            RefmLinearFilter,
        )

        for cls in (Persistence, Climatology, LightGBMForecaster, RefmLinearFilter):
            try:
                return cls.load(path)
            except Exception:  # noqa: BLE001 - try the next class
                continue
        raise RuntimeError(f"no Forecaster subclass could load {path}")

    # ---- inference --------------------------------------------------------------------
    def predict(
        self,
        feature_vector: np.ndarray,
        *,
        future: np.ndarray | None = None,
        context: dict[str, float] | None = None,
    ) -> ForecastPayload:
        """Return the multi-horizon forecast for one feature vector.

        Cache hit -> O(1) return; miss -> one backend call -> cache + return. ``context``
        supplies ``mlt``/``kp``/``vsw``/``doy`` for the climatology backend / fallback.

        Parameters
        ----------
        feature_vector:
            Encoder feature vector aligned to ``schema.FEATURE_COLUMNS`` (1-D ``[F]`` or a
            single-row ``[1, F]``). For ONNX, a ``[1, 1, F]`` encoder window is built.
        future:
            Optional known-future covariates ``[H, F_kf]`` for the Forecaster backend.
        context:
            Optional scalar context for the climatology backend.

        Returns
        -------
        ForecastPayload
            The cached or freshly computed forecast.
        """
        features = np.asarray(feature_vector, dtype="float32").reshape(-1)
        cached = self.cache.get(features)
        if cached is not None:
            return cached

        t0 = time.perf_counter()
        if self.backend == "onnx" and self._session is not None:
            payload = self._predict_onnx(features)
        elif self.backend == "forecaster" and self._model is not None:
            payload = self._predict_forecaster(features, future)
        else:
            payload = self._predict_climatology(features, context)
        payload.latency_ms = (time.perf_counter() - t0) * 1000.0

        self.cache.put(features, payload)
        return payload

    def _predict_onnx(self, features: np.ndarray) -> ForecastPayload:
        """Run the warm ONNX session and parse quantile / exceedance outputs."""
        # Encoder expects [N, L, F]; serve a single-step window.
        x = features.reshape(1, 1, -1).astype("float32")
        inp_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {inp_name: x})
        return self._payload_from_arrays(outputs)

    def _payload_from_arrays(self, outputs: list[np.ndarray]) -> ForecastPayload:
        """Build a payload from raw model outputs (quantiles [, exceedance]).

        Accepts either ``[quantiles]`` of shape ``[N, n_q, n_h]`` (ordered by QUANTILES)
        or ``[quantiles, proba]``. Falls back to a degenerate spread if shapes are unusual.
        """
        n_h = len(HORIZON_NAMES)
        q_arr = np.asarray(outputs[0], dtype="float64").reshape(-1)
        if q_arr.size >= len(QUANTILES) * n_h:
            q_mat = q_arr[: len(QUANTILES) * n_h].reshape(len(QUANTILES), n_h)
            p10, p50, p90 = q_mat[0], q_mat[1], q_mat[2]
        else:  # treat the output as a point forecast
            point = q_arr[:n_h] if q_arr.size >= n_h else np.full(n_h, q_arr.mean())
            p10, p50, p90 = point - 0.3, point, point + 0.3
        if len(outputs) > 1:
            proba = np.asarray(outputs[1], dtype="float64").reshape(-1)[:n_h]
            proba = np.clip(proba, 0.0, 1.0)
        else:
            proba = (p50 >= LOG_HARSH).astype("float64")
        return build_payload(
            {0.1: p10, 0.5: p50, 0.9: p90},
            proba,
            satellite=self.satellite,
            model_name=self.model_name,
            source=self.source,
        )

    def _predict_forecaster(
        self, features: np.ndarray, future: np.ndarray | None
    ) -> ForecastPayload:
        """Run a :class:`Forecaster` backend over a single-step encoder window."""
        x = features.reshape(1, 1, -1).astype("float32")
        n_h = len(HORIZON_NAMES)
        f_kf = len(schema.KNOWN_FUTURE_COLUMNS)
        if future is None:
            x_future = np.zeros((1, 1, f_kf), dtype="float32")
        else:
            fut = np.asarray(future, dtype="float32")
            x_future = fut.reshape(1, fut.shape[0], -1) if fut.ndim == 2 else fut
        q = self._model.predict_quantiles(x, x_future)
        proba = np.asarray(self._model.predict_proba_exceed(x, x_future)).reshape(-1)[:n_h]
        return build_payload(
            {tau: np.asarray(arr).reshape(-1)[:n_h] for tau, arr in q.items()},
            proba,
            satellite=self.satellite,
            model_name=self.model_name,
            source=self.source,
        )

    def _predict_climatology(
        self, features: np.ndarray, context: dict[str, float] | None
    ) -> ForecastPayload:
        """Fall back to the climatology LUT, deriving context from the feature vector."""
        ctx = dict(context or {})
        # Pull a sensible vsw/kp from the feature vector if not supplied explicitly.
        vsw = float(features[_VSW_CHANNEL]) if features.size > _VSW_CHANNEL else 0.0
        kp = float(features[_KP_CHANNEL]) if features.size > _KP_CHANNEL else 0.0
        ctx.setdefault("vsw", vsw)
        ctx.setdefault("kp", kp)
        ctx.setdefault("mlt", 12.0)
        ctx.setdefault("doy", 0.0)
        return self.lut.forecast(
            mlt=ctx["mlt"],
            doy=ctx["doy"],
            kp=ctx["kp"],
            vsw=ctx["vsw"],
            satellite=self.satellite,
            source=self.source,
        )


# ======================================================================================
# ONNX export helper (lazy torch / onnx)
# ======================================================================================
def export_to_onnx(
    model: Any,
    path: str | Path,
    *,
    lookback: int = 1,
    n_features: int | None = None,
    opset: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """Export a ``torch.nn.Module`` to ONNX for serving (lazy ``torch`` import, R4 §3.1).

    Calls ``model.eval()``, traces with a representative dummy ``[1, lookback, F]`` input
    and declares a dynamic batch axis. Only meaningful for torch models; raises if torch is
    unavailable or ``model`` is not an ``nn.Module``.

    Parameters
    ----------
    model:
        A ``torch.nn.Module`` to export.
    path:
        Destination ``.onnx`` path.
    lookback, n_features:
        Dummy-input encoder shape (``n_features`` defaults to ``len(FEATURE_COLUMNS)``).
    opset:
        ONNX opset version.
    dynamic_batch:
        Declare a dynamic batch (and sequence) axis.

    Returns
    -------
    Path
        The written ONNX path.
    """
    import torch  # lazy: optional [dl] extra

    if not isinstance(model, torch.nn.Module):
        raise TypeError("export_to_onnx requires a torch.nn.Module")

    f = int(n_features if n_features is not None else len(schema.FEATURE_COLUMNS))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy = torch.zeros(1, int(lookback), f, dtype=torch.float32)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {"input": {0: "batch", 1: "lookback"}, "output": {0: "batch"}}
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )
    return path


# Backwards-compatible alias for the scaffold's class name.
CachedForecaster = Predictor


__all__ = [
    "HorizonForecast",
    "ForecastPayload",
    "OnlineFeatureState",
    "ForecastCache",
    "ClimatologyLUT",
    "Predictor",
    "CachedForecaster",
    "build_payload",
    "export_to_onnx",
    "HORIZONS",
    "LOG_THRESHOLD",
]
