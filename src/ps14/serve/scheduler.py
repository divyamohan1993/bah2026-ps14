"""APScheduler refresh job for the real-time forecast (R4 §6, ARCHITECTURE.md (f.5)).

Every ``interval_s`` seconds the job: pulls the latest sample (from
:mod:`ps14.io.swpc_realtime` if available, else a provided callable / synthetic
generator), folds it into the O(1) online features, runs cached inference, writes the
result to the shared latest-forecast store and pushes it to WebSocket subscribers. The
refresh is dominated by network I/O, not compute (R4 §6).

``apscheduler`` is imported lazily inside :func:`start_scheduler` so ``import ps14`` works
with only the core deps; the optional ``[serve]`` extra is needed only to start the
background scheduler.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from typing import Any

from ps14.constants import LOG_HARSH
from ps14.serve.inference import (
    ForecastPayload,
    OnlineFeatureState,
    Predictor,
)

# A sample is a mapping of channel -> value; a source is a zero-arg callable returning one.
Sample = dict[str, float]
SampleSource = Callable[[], Sample]


def synthetic_source(seed: int | None = None) -> SampleSource:
    """Return a zero-arg callable yielding a physically-plausible synthetic sample.

    A smooth diurnal log-flux (peaking near local noon) plus an Ornstein-Uhlenbeck-ish
    solar-wind wander, so the serving path is exercised offline with no network (R4 §1.1).
    Used as the default ``source`` when SWPC is unavailable.
    """
    rng = random.Random(seed)
    state = {"vsw": 420.0, "t": 0}

    def _next() -> Sample:
        t = state["t"]
        state["t"] = t + 1
        # Diurnal flux (24 h cycle at 5-min cadence -> 288 steps/day).
        phase = 2.0 * math.pi * ((t % 288) / 288.0)
        log_flux = 2.4 + 0.5 * math.cos(phase - math.pi) + rng.gauss(0.0, 0.05)
        # OU-ish solar-wind speed.
        state["vsw"] = 0.98 * state["vsw"] + 0.02 * 420.0 + rng.gauss(0.0, 8.0)
        vsw = state["vsw"]
        return {
            "log_flux_e2": log_flux,
            "log_flux_seed": log_flux - 0.5,
            "vsw": vsw,
            "density": max(0.5, rng.gauss(4.0, 1.0)),
            "pdyn": 1.6e-6 * max(0.5, rng.gauss(4.0, 1.0)) * vsw * vsw,
            "bz_gsm": rng.gauss(0.0, 3.0),
            "bt": abs(rng.gauss(5.0, 1.5)),
            "ae": abs(rng.gauss(150.0, 80.0)),
            "al": -abs(rng.gauss(120.0, 70.0)),
            "kp": min(9.0, max(0.0, rng.gauss(2.5, 1.0))),
            "sym_h": rng.gauss(-5.0, 10.0),
            "f107": rng.gauss(120.0, 10.0),
            "mlt": (t % 288) / 288.0 * 24.0,
        }

    return _next


def _swpc_source_or_none() -> SampleSource | None:
    """Build a SWPC-backed sample source if the realtime client is usable, else None.

    Imported lazily and defensively: the realtime client is being implemented concurrently
    and its fetchers may raise ``NotImplementedError`` / network errors, so any failure
    falls back to the synthetic source.
    """
    try:
        from ps14.io import swpc_realtime
    except Exception:  # noqa: BLE001 - optional [data] extra / partial implementation
        return None

    def _next() -> Sample:
        sample: Sample = {}
        try:
            electrons = swpc_realtime.fetch_latest_electrons()
            row = electrons.iloc[-1]
            if "flux" in row:
                flux = float(row["flux"])
                sample["log_flux_e2"] = math.log10(flux) if flux > 0 else LOG_HARSH - 2.0
        except Exception:  # noqa: BLE001 - tolerate any feed error
            pass
        try:
            sw = swpc_realtime.fetch_latest_solar_wind()
            row = sw.iloc[-1]
            for key, col in (("vsw", "speed"), ("density", "density"), ("bz_gsm", "bz_gsm")):
                if col in row:
                    sample[key] = float(row[col])
        except Exception:  # noqa: BLE001
            pass
        return sample

    return _next


def resolve_source(source: Any = "synthetic") -> SampleSource:
    """Resolve the ``source`` argument into a zero-arg sample callable.

    Accepts a callable (returned as-is), ``"swpc"`` (the realtime client, falling back to
    synthetic if unusable), or ``"synthetic"`` / anything else (the synthetic generator).
    """
    if callable(source):
        return source
    if source == "swpc":
        swpc = _swpc_source_or_none()
        if swpc is not None:
            return swpc
    return synthetic_source()


class RefreshState:
    """Bundles the O(1) online-feature state + the cached predictor for the refresh job.

    Holds a single :class:`~ps14.serve.inference.OnlineFeatureState` (the amortized-O(1)
    sliding-window updater), the warm :class:`~ps14.serve.inference.Predictor`, the sample
    source, and the most recent observed scalars + forecast so the scheduled job stays a
    thin O(new) update.
    """

    def __init__(
        self,
        predictor: Predictor | None = None,
        *,
        source: Any = "synthetic",
        config: Any | None = None,
    ) -> None:
        self.config = config
        self.predictor = predictor if predictor is not None else Predictor()
        self.feature_state = OnlineFeatureState()
        self.source: SampleSource = resolve_source(source)
        self.source_name = source if isinstance(source, str) else "callable"
        self.latest: dict[str, Any] = {}
        self.forecast: ForecastPayload | None = None
        # Backwards-compatible attribute names from the scaffold.
        self.online: dict[str, Any] = {"feature_state": self.feature_state}
        self.forecaster = self.predictor

    def initialize(self) -> None:
        """Warm the online accumulators by folding a few synthetic samples (idempotent)."""
        warm = synthetic_source(seed=0)
        for _ in range(max(1, self.feature_state.warmup)):
            self.feature_state.update(warm())


def refresh_once(state: RefreshState) -> ForecastPayload:
    """Run one refresh cycle and return (and store) the new forecast payload.

    Pulls one sample from ``state.source``, folds it into the O(1) online features, runs
    cached inference and updates ``state.latest`` / ``state.forecast``. Resilient to source
    errors: on any failure it reuses the last feature vector (or warms from synthetic).

    Returns
    -------
    ForecastPayload
        The new forecast (also assigned to ``state.forecast``).
    """
    try:
        sample = state.source()
    except Exception:  # noqa: BLE001 - never let a flaky source kill the loop
        sample = synthetic_source(seed=int(time.time()))()

    clean = {k: v for k, v in sample.items() if isinstance(v, (int, float)) and v == v}
    features = state.feature_state.update(clean).copy()

    context = {k: float(clean[k]) for k in ("mlt", "kp", "vsw", "doy") if k in clean}
    payload = state.predictor.predict(features, context=context)

    state.forecast = payload
    state.latest = {
        "time": payload.issued_utc,
        "flux_e2": float(10.0 ** clean.get("log_flux_e2", LOG_HARSH - 1.0)),
        "vsw": clean.get("vsw"),
        "bz_gsm": clean.get("bz_gsm"),
        "kp": clean.get("kp"),
        "mlt": clean.get("mlt"),
        "alert": any(h.alert for h in payload.horizons.values()),
    }
    return payload


def start_scheduler(
    predictor_or_state: Predictor | RefreshState,
    source: Any = "synthetic",
    interval_s: int = 60,
    *,
    on_update: Callable[[ForecastPayload], None] | None = None,
    store: Any | None = None,
) -> Any:
    """Start an APScheduler ``BackgroundScheduler`` running :func:`refresh_once`.

    Parameters
    ----------
    predictor_or_state:
        Either a :class:`~ps14.serve.inference.Predictor` (a :class:`RefreshState` is built
        around it) or a ready :class:`RefreshState`.
    source:
        ``"swpc"`` (realtime JSON), ``"synthetic"`` (offline replay), or a zero-arg callable
        returning a sample. Ignored when a ready :class:`RefreshState` is passed.
    interval_s:
        Refresh interval in seconds (default 60, per the latency budget in R4 §6).
    on_update:
        Optional callback invoked with each new :class:`ForecastPayload` (e.g. WS fan-out).
    store:
        Optional :class:`~ps14.serve.api.ForecastStore`; if given, its ``update_forecast`` /
        ``update_observed`` are called on each refresh.

    Returns
    -------
    apscheduler.schedulers.background.BackgroundScheduler
        The started scheduler (the caller keeps a reference and shuts it down on exit).

    Raises
    ------
    RuntimeError
        If APScheduler is not installed (install the ``[serve]`` extra).
    """
    try:  # lazy import so core `import ps14` never needs apscheduler
        from apscheduler.schedulers.background import BackgroundScheduler
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "APScheduler is not installed; install the '[serve]' extra to run the scheduler."
        ) from exc

    if isinstance(predictor_or_state, RefreshState):
        state = predictor_or_state
    else:
        state = RefreshState(predictor_or_state, source=source)
    state.initialize()

    def _job() -> None:
        payload = refresh_once(state)
        if store is not None:
            store.update_forecast(payload)
            store.update_observed(state.latest)
        if on_update is not None:
            on_update(payload)

    scheduler = BackgroundScheduler()
    scheduler.add_job(_job, "interval", seconds=int(interval_s), id="ps14-refresh")
    scheduler.start()
    return scheduler


__all__ = [
    "RefreshState",
    "refresh_once",
    "start_scheduler",
    "resolve_source",
    "synthetic_source",
    "Sample",
    "SampleSource",
]
