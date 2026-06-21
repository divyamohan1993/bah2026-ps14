"""FastAPI serving app for the >2 MeV GEO electron forecast (R4 §6, ARCHITECTURE.md (f.5)).

Endpoints:
  GET  /health            liveness + model / source / uptime.
  GET  /latest            most recent observed flux / solar-wind / Kp + alert (O(1) cache).
  GET  /forecast          current multi-horizon forecast with uncertainty (O(1) cache).
  GET  /forecast/{horizon} a single horizon block from the current forecast.
  POST /forecast          run inference on a posted feature/sample payload.
  WS   /ws                streams the forecast object as it refreshes.

``fastapi`` is imported lazily inside :func:`create_app` so ``import ps14`` (and importing
this module) works with only the core deps installed — the optional ``[serve]`` extra is
needed only to actually build / run the app.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np

from ps14.constants import HORIZON_NAMES
from ps14.datasets import schema
from ps14.serve.inference import (
    ForecastPayload,
    OnlineFeatureState,
    Predictor,
)


class ForecastStore:
    """Shared in-memory hot store for the latest forecast + observed values (O(1) reads).

    Holds the most recent :class:`ForecastPayload`, a small dict of latest observed
    scalars, the set of connected WebSocket subscribers, and the process start time. All
    reads/writes are constant-time dict / attribute access (R4 §3.5).
    """

    def __init__(self) -> None:
        self.latest_forecast: ForecastPayload | None = None
        self.latest_observed: dict[str, Any] = {}
        self.subscribers: set[Any] = set()
        self.started_at: float = time.time()

    def update_forecast(self, payload: ForecastPayload) -> None:
        """Store the newest forecast (O(1))."""
        self.latest_forecast = payload

    def update_observed(self, observed: dict[str, Any]) -> None:
        """Merge the newest observed scalars (flux/vsw/bz/kp/mlt/time) (O(1))."""
        self.latest_observed.update(observed)

    def uptime_s(self) -> float:
        """Seconds since the store (process) started."""
        return time.time() - self.started_at


def _sample_to_features(sample: dict[str, Any], state: OnlineFeatureState) -> np.ndarray:
    """Turn a posted payload into a feature vector.

    Accepts either a ready feature vector (key ``features`` / ``feature_vector``) or a raw
    observation dict that is folded through ``state`` (the O(1) online updater).
    """
    if "features" in sample or "feature_vector" in sample:
        vec = np.asarray(sample.get("features", sample.get("feature_vector")), dtype="float32")
        return vec.reshape(-1)
    return state.update({k: v for k, v in sample.items() if isinstance(v, (int, float))}).copy()


def create_app(config: Any | None = None, predictor: Predictor | None = None) -> Any:
    """Build the FastAPI app with the forecast / latest / health / ws endpoints.

    Parameters
    ----------
    config:
        A :class:`ps14.config.Settings` (or None to load the default).
    predictor:
        A :class:`~ps14.serve.inference.Predictor` to inject (a climatology-fallback one is
        created if omitted, so the app is runnable with no trained artifact).

    Returns
    -------
    fastapi.FastAPI
        The configured application. ``app.state`` carries ``store``, ``predictor``,
        ``feature_state`` and ``config``.

    Raises
    ------
    RuntimeError
        If FastAPI is not installed (install the ``[serve]`` extra).
    """
    try:  # lazy import so core `import ps14` never needs fastapi
        from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "FastAPI is not installed; install the '[serve]' extra to build the API."
        ) from exc

    if config is None:
        try:
            from ps14.config import load_config

            config = load_config()
        except Exception:  # noqa: BLE001 - config is optional for serving
            config = None

    source = getattr(getattr(config, "serving", None), "source", "synthetic")
    app = FastAPI(title="PS-14 >2 MeV GEO electron forecast", version="1.0")

    store = ForecastStore()
    if predictor is None:
        predictor = Predictor(source=source)
    feature_state = OnlineFeatureState()

    app.state.store = store
    app.state.predictor = predictor
    app.state.feature_state = feature_state
    app.state.config = config

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Liveness + model / source / uptime (CONTRACTS.md §7)."""
        return {
            "status": "ok",
            "model": predictor.model_name,
            "source": source,
            "uptime_s": round(store.uptime_s(), 3),
        }

    @app.get("/latest")
    def latest() -> dict[str, Any]:
        """Most recent observed values + current alert flag (O(1) cache read)."""
        observed = dict(store.latest_observed)
        fc = store.latest_forecast
        observed["alert"] = bool(fc is not None and any(h.alert for h in fc.horizons.values()))
        return observed

    @app.get("/forecast")
    def forecast() -> dict[str, Any]:
        """Return the current multi-horizon forecast (O(1) cache read)."""
        if store.latest_forecast is None:
            raise HTTPException(status_code=503, detail="no forecast available yet")
        return store.latest_forecast.model_dump()

    @app.get("/forecast/{horizon}")
    def forecast_horizon(horizon: str) -> dict[str, Any]:
        """Return one horizon block (``nowcast`` / ``6h`` / ``12h``) of the forecast."""
        if horizon not in HORIZON_NAMES:
            raise HTTPException(
                status_code=404, detail=f"unknown horizon {horizon!r}; expected {HORIZON_NAMES}"
            )
        if store.latest_forecast is None:
            raise HTTPException(status_code=503, detail="no forecast available yet")
        return store.latest_forecast.horizons[horizon].model_dump()

    @app.post("/forecast")
    def post_forecast(payload: dict[str, Any]) -> dict[str, Any]:
        """Run inference on a posted feature/sample payload and return a forecast.

        The body is either ``{"features": [...]}`` (a ready feature vector aligned to
        ``schema.FEATURE_COLUMNS``) or a raw observation dict (``log_flux_e2``, ``vsw``, …)
        that is folded through the O(1) online updater.
        """
        features = _sample_to_features(payload, feature_state)
        expected = len(schema.FEATURE_COLUMNS)
        if features.size != expected:
            raise HTTPException(
                status_code=422,
                detail=f"feature vector has {features.size} values; expected {expected}",
            )
        context = {k: float(payload[k]) for k in ("mlt", "kp", "vsw", "doy") if k in payload}
        result = predictor.predict(features, context=context)
        store.update_forecast(result)
        return result.model_dump()

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        """Stream the forecast object as it refreshes (push on each store update)."""
        await websocket.accept()
        store.subscribers.add(websocket)
        try:
            # Send the current snapshot immediately, then poll the shared store for changes.
            last_sent = None
            if store.latest_forecast is not None:
                await websocket.send_json(store.latest_forecast.model_dump())
                last_sent = store.latest_forecast.issued_utc
            while True:
                fc = store.latest_forecast
                if fc is not None and fc.issued_utc != last_sent:
                    await websocket.send_json(fc.model_dump())
                    last_sent = fc.issued_utc
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:  # pragma: no cover - network teardown
            pass
        finally:
            store.subscribers.discard(websocket)

    return app


def run(config: Any | None = None) -> None:  # pragma: no cover - launches a server
    """Launch the app with Uvicorn (used by ``ps14 serve`` / ``make serve``)."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "Uvicorn is not installed; install the '[serve]' extra to run the API."
        ) from exc

    if config is None:
        from ps14.config import load_config

        config = load_config()
    host = getattr(getattr(config, "serving", None), "host", "0.0.0.0")
    port = getattr(getattr(config, "serving", None), "port", 8000)
    uvicorn.run(create_app(config), host=host, port=port)


__all__ = ["ForecastStore", "create_app", "run"]
