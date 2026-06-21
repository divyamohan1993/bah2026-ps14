"""Serving: ONNX inference + cache, FastAPI app, APScheduler refresh (R4 §3/§6).

The real-time hot path uses the O(1) online primitives (ps14.features.online), cached
ONNX inference, and a climatology LUT fallback, exposed via FastAPI and refreshed every
60 s by APScheduler. See ARCHITECTURE.md (f).
"""

from __future__ import annotations

from ps14.serve.inference import (
    ClimatologyLUT,
    ForecastCache,
    ForecastPayload,
    HorizonForecast,
    OnlineFeatureState,
    Predictor,
    build_payload,
    export_to_onnx,
)

__all__ = [
    "OnlineFeatureState",
    "ForecastCache",
    "ClimatologyLUT",
    "Predictor",
    "ForecastPayload",
    "HorizonForecast",
    "build_payload",
    "export_to_onnx",
]
