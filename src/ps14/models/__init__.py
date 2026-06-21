"""Models: the Forecaster interface, baselines, the primary TFT, and backups.

All models implement :class:`ps14.models.base.Forecaster` so train/evaluate/serve treat
them interchangeably (CONTRACTS.md §5). See ARCHITECTURE.md (e).
"""

from __future__ import annotations

from ps14.models.base import Forecaster

__all__ = ["Forecaster"]
