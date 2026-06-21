"""Online O(1) streaming feature primitives for the real-time serving hot path.

Three small, well-defined primitives that fold each incoming sample into the feature set
in amortized constant time, with no re-scan of the window (R4 §2). These are FULLY
IMPLEMENTED and unit-tested (see tests/test_features_online.py):

* :class:`RingBuffer`     — O(1) append / O(1) evict fixed-size circular buffer.
* :class:`Welford`        — O(1) numerically stable running mean / variance.
* :class:`MonotonicDeque` — amortized O(1) rolling min/max over a sliding window.

Together they keep the per-sample feature update amortized O(1) regardless of window
length, which is what makes the per-minute nowcast refresh dominated by network I/O
rather than computation (R4 §2.4).
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


class RingBuffer:
    """Fixed-size circular buffer over a preallocated NumPy array.

    Complexity
    ----------
    ``append``: O(1) (overwrites the oldest slot once full).
    ``__getitem__`` / ``values``: O(1) / O(capacity) for the ordered copy.

    Notes
    -----
    A plain ``list.pop(0)`` is O(N); this is O(1) with a fixed memory footprint and a
    contiguous backing store suitable for vectorized reads (R4 §2.1).
    """

    def __init__(self, capacity: int, dtype: str = "float64") -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._buf = np.empty(self.capacity, dtype=dtype)
        self._head = 0  # index of the next write
        self._count = 0  # number of valid elements (<= capacity)

    def append(self, x: float) -> None:
        """Append one sample (O(1)); overwrites the oldest element once full."""
        self._buf[self._head] = x
        self._head = (self._head + 1) % self.capacity
        self._count = min(self._count + 1, self.capacity)

    def extend(self, xs) -> None:
        """Append many samples (O(len(xs)))."""
        for x in xs:
            self.append(x)

    @property
    def is_full(self) -> bool:
        return self._count == self.capacity

    def __len__(self) -> int:
        return self._count

    def values(self) -> np.ndarray:
        """Return the buffered values in chronological (oldest -> newest) order."""
        if self._count < self.capacity:
            return self._buf[: self._count].copy()
        # Full buffer: oldest element is at _head.
        return np.concatenate((self._buf[self._head :], self._buf[: self._head]))

    def latest(self) -> float:
        """Return the most recently appended value (O(1)). Raises if empty."""
        if self._count == 0:
            raise IndexError("RingBuffer is empty")
        return float(self._buf[(self._head - 1) % self.capacity])

    def __getitem__(self, i: int) -> float:
        """Index in chronological order: ``0`` = oldest, ``-1`` = newest (O(1))."""
        if not -self._count <= i < self._count:
            raise IndexError("ring buffer index out of range")
        if i < 0:
            i += self._count
        if self._count < self.capacity:
            return float(self._buf[i])
        return float(self._buf[(self._head + i) % self.capacity])


class Welford:
    """Welford's algorithm: O(1) numerically stable running mean and variance.

    For each new value ``x`` with running count ``n`` (R4 §2.2)::

        n += 1; delta = x - mean; mean += delta / n
        delta2 = x - mean; M2 += delta * delta2

    Complexity
    ----------
    ``update``: O(1) time, O(1) memory; no stored history. Avoids the catastrophic
    cancellation of the naive sum-of-squares form.

    A windowed mean/variance can be obtained by also calling :meth:`remove` with the
    value leaving a fixed-N window (symmetric remove-then-add update).
    """

    __slots__ = ("n", "mean", "_m2")

    def __init__(self) -> None:
        self.n: int = 0
        self.mean: float = 0.0
        self._m2: float = 0.0

    def update(self, x: float) -> None:
        """Fold one new value into the running statistics (O(1))."""
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._m2 += delta * delta2

    def remove(self, x: float) -> None:
        """Remove one value (for a fixed-window variant); inverse of :meth:`update`.

        Raises
        ------
        ValueError
            If there are no samples to remove.
        """
        if self.n == 0:
            raise ValueError("cannot remove from an empty Welford accumulator")
        if self.n == 1:
            self.n = 0
            self.mean = 0.0
            self._m2 = 0.0
            return
        prev_mean = (self.n * self.mean - x) / (self.n - 1)
        self._m2 -= (x - self.mean) * (x - prev_mean)
        self.mean = prev_mean
        self.n -= 1
        # Guard tiny negative drift from floating-point round-off.
        if self._m2 < 0.0:
            self._m2 = 0.0

    @property
    def variance(self) -> float:
        """Population variance ``M2 / n`` (NaN if no samples)."""
        if self.n == 0:
            return math.nan
        return self._m2 / self.n

    @property
    def sample_variance(self) -> float:
        """Sample variance ``M2 / (n - 1)`` (NaN if < 2 samples)."""
        if self.n < 2:
            return math.nan
        return self._m2 / (self.n - 1)

    @property
    def std(self) -> float:
        """Population standard deviation."""
        v = self.variance
        return math.sqrt(v) if v == v else math.nan  # NaN-safe

    def zscore(self, x: float) -> float:
        """Z-score of ``x`` against the running mean/std (0.0 if std is 0/NaN)."""
        s = self.std
        if not s or s != s:
            return 0.0
        return (x - self.mean) / s


class MonotonicDeque:
    """Monotonic deque for amortized O(1) rolling min OR max over a sliding window.

    Maintains a deque of ``(index, value)`` pairs whose values are monotonic so the front
    is always the current window extremum. Each index is pushed once and popped at most
    once -> amortized O(1) per sample; querying the extremum is O(1) — strictly better
    than a heap (O(log n)) or a rescan (O(window)) (R4 §2.3).

    Parameters
    ----------
    window:
        Sliding-window length in samples.
    kind:
        ``"max"`` (front = window maximum) or ``"min"`` (front = window minimum).
    """

    def __init__(self, window: int, kind: str = "max") -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        if kind not in ("max", "min"):
            raise ValueError("kind must be 'max' or 'min'")
        self.window = int(window)
        self.kind = kind
        self._dq: deque[tuple[int, float]] = deque()
        self._j = -1  # index of the most recent push

    def push(self, x: float) -> float:
        """Push the next sample and return the current window extremum (amortized O(1))."""
        self._j += 1
        j = self._j
        # 1) Evict from the front any index that has fallen out of the window.
        while self._dq and self._dq[0][0] <= j - self.window:
            self._dq.popleft()
        # 2) Pop from the back any element dominated by the new one.
        if self.kind == "max":
            while self._dq and self._dq[-1][1] <= x:
                self._dq.pop()
        else:
            while self._dq and self._dq[-1][1] >= x:
                self._dq.pop()
        # 3) Push the new element and (4) return the front extremum.
        self._dq.append((j, x))
        return self._dq[0][1]

    @property
    def extreme(self) -> float:
        """Current window extremum without pushing (O(1)). Raises if empty."""
        if not self._dq:
            raise IndexError("MonotonicDeque is empty")
        return self._dq[0][1]


__all__ = ["RingBuffer", "Welford", "MonotonicDeque"]
