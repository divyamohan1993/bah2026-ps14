"""ps14 — Forecasting the >2 MeV energetic electron radiation environment at GEO.

BAH-2026 Problem Statement 14. End-to-end AI/ML system that reads/processes/visualizes
CDF-format space-weather data and forecasts the >2 MeV integral electron flux at
geostationary orbit at 30-45 min (nowcast), 6 h and 12 h, plus a calibrated probability
of exceeding the NOAA "harsh" alert threshold (1000 pfu).

See ARCHITECTURE.md for the design and CONTRACTS.md for the binding data contracts.

Sub-packages
------------
io          CDF reading, CDAWeb/HAPI/SWPC fetch, synthetic data generation.
preprocess  Cleaning, despiking, gap handling, resampling, alignment, transforms.
features    Offline feature engineering + O(1) online primitives for serving.
datasets    Supervised windowing, chronological splits, canonical schema.
models      Forecaster ABC, baselines, TFT, N-HiTS, foundation-model backups.
serve       ONNX inference + cache, FastAPI app, APScheduler refresh.
dashboard   Streamlit live demo.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
