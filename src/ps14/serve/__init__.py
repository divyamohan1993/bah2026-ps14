"""Serving: ONNX inference + cache, FastAPI app, APScheduler refresh (R4 §3/§6).

The real-time hot path uses the O(1) online primitives (ps14.features.online), cached
ONNX inference, and a climatology LUT fallback, exposed via FastAPI and refreshed every
60 s by APScheduler. See ARCHITECTURE.md (f).
"""

from __future__ import annotations

__all__: list[str] = []
