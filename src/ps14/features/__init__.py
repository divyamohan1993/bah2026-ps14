"""Feature engineering: offline batch features + online O(1) primitives.

``offline`` builds the lag/rolling/coupling/cyclic feature matrix (CONTRACTS.md §3) for
training. ``online`` provides the amortized-O(1) streaming primitives (ring buffer,
Welford, monotonic deque) for the serving hot path (R4 §2).
"""

from __future__ import annotations

from ps14.features.online import MonotonicDeque, RingBuffer, Welford

__all__ = ["RingBuffer", "Welford", "MonotonicDeque"]
