"""FastAPI application skeleton (R4 §6, ARCHITECTURE.md (f.5)).

Endpoints:
  GET  /health   liveness + model/source/uptime.
  GET  /latest   most recent observed flux/solar-wind/Kp + alert status (O(1) cache).
  GET  /forecast current multi-horizon forecast with uncertainty (O(1) cache).
  WS   /ws       pushes the forecast object on each 60 s refresh.

Importable even without FastAPI installed (the ``[serve]`` extra): :func:`create_app`
raises a clear error if FastAPI is missing, but importing this module never fails — so
``python -c "import ps14"`` works in the core environment.
"""

from __future__ import annotations

from typing import Any

try:  # optional dependency (the [serve] extra)
    from fastapi import FastAPI  # type: ignore

    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover - exercised only without the extra
    FastAPI = None  # type: ignore
    _HAS_FASTAPI = False


def create_app(config: Any | None = None) -> FastAPI:
    """Build the FastAPI app with the forecast/latest/health/ws endpoints.

    Parameters
    ----------
    config:
        A :class:`ps14.config.Settings` (or None to load the default).

    Returns
    -------
    FastAPI
        The configured application.

    Raises
    ------
    RuntimeError
        If FastAPI is not installed (install the ``[serve]`` extra).
    """
    if not _HAS_FASTAPI:
        raise RuntimeError("FastAPI is not installed; install the '[serve]' extra to run the API.")
    raise NotImplementedError(
        "TODO: app = FastAPI(title='PS-14 >2 MeV GEO electron forecast'); register the "
        "hot-cache-backed /health, /latest, /forecast routes and the /ws WebSocket; wire the "
        "APScheduler refresh job from ps14.serve.scheduler (R4 §6)."
    )


def run(config: Any | None = None) -> None:
    """Launch the app with Uvicorn (used by ``ps14 serve`` / ``make serve``)."""
    if not _HAS_FASTAPI:
        raise RuntimeError("FastAPI/Uvicorn not installed; install the '[serve]' extra.")
    raise NotImplementedError(
        "TODO: import uvicorn; "
        "uvicorn.run(create_app(config), host=cfg.serving.host, port=cfg.serving.port)."
    )


__all__ = ["create_app", "run"]
