"""Real tests for the O(1) online primitives (ps14.features.online).

Each primitive is validated against a brute-force reference so the amortized-O(1)
implementations are provably correct (R4 §2).
"""

from __future__ import annotations

import numpy as np
import pytest

from ps14.features.online import MonotonicDeque, RingBuffer, Welford

# --------------------------------------------------------------------------------------
# RingBuffer
# --------------------------------------------------------------------------------------


def test_ring_buffer_basic_append_and_order():
    rb = RingBuffer(3)
    assert len(rb) == 0
    rb.append(1.0)
    rb.append(2.0)
    assert len(rb) == 2
    assert rb.latest() == 2.0
    np.testing.assert_array_equal(rb.values(), np.array([1.0, 2.0]))


def test_ring_buffer_wraparound_keeps_last_n():
    rb = RingBuffer(3)
    rb.extend([1.0, 2.0, 3.0, 4.0, 5.0])
    assert rb.is_full
    assert len(rb) == 3
    # Oldest evicted: only the last 3 remain in chronological order.
    np.testing.assert_array_equal(rb.values(), np.array([3.0, 4.0, 5.0]))
    assert rb.latest() == 5.0


def test_ring_buffer_indexing_chronological():
    rb = RingBuffer(3)
    rb.extend([10.0, 20.0, 30.0, 40.0])  # buffer now [20,30,40]
    assert rb[0] == 20.0  # oldest
    assert rb[-1] == 40.0  # newest
    assert rb[1] == 30.0
    with pytest.raises(IndexError):
        _ = rb[5]


def test_ring_buffer_empty_latest_raises():
    rb = RingBuffer(2)
    with pytest.raises(IndexError):
        rb.latest()


def test_ring_buffer_invalid_capacity():
    with pytest.raises(ValueError):
        RingBuffer(0)


# --------------------------------------------------------------------------------------
# Welford
# --------------------------------------------------------------------------------------


def test_welford_matches_numpy_mean_var():
    rng = np.random.default_rng(0)
    data = rng.normal(100.0, 15.0, size=5000)
    w = Welford()
    for x in data:
        w.update(x)
    assert w.n == len(data)
    assert w.mean == pytest.approx(np.mean(data), rel=1e-10)
    assert w.variance == pytest.approx(np.var(data), rel=1e-9)
    assert w.sample_variance == pytest.approx(np.var(data, ddof=1), rel=1e-9)
    assert w.std == pytest.approx(np.std(data), rel=1e-9)


def test_welford_numerically_stable_large_offset():
    # Naive sum-of-squares would catastrophically cancel here; Welford must not.
    data = 1.0e9 + np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    w = Welford()
    for x in data:
        w.update(x)
    assert w.variance == pytest.approx(np.var(data), rel=1e-6)


def test_welford_remove_is_inverse_of_update():
    rng = np.random.default_rng(1)
    data = rng.normal(size=200)
    w = Welford()
    for x in data:
        w.update(x)
    # Remove the last 50; should equal stats over the first 150.
    for x in data[-50:][::-1]:
        w.remove(x)
    assert w.n == 150
    assert w.mean == pytest.approx(np.mean(data[:150]), rel=1e-9)
    assert w.variance == pytest.approx(np.var(data[:150]), rel=1e-7)


def test_welford_zscore_and_empty():
    w = Welford()
    assert np.isnan(w.variance)
    assert w.zscore(0.0) == 0.0  # std undefined -> 0
    with pytest.raises(ValueError):
        w.remove(1.0)


def test_welford_windowed_via_ring_and_remove():
    # Combine RingBuffer + Welford.remove for a fixed-window running mean/var.
    rng = np.random.default_rng(2)
    data = rng.normal(size=300)
    window = 50
    rb = RingBuffer(window)
    w = Welford()
    ref_means = []
    for x in data:
        if rb.is_full:
            w.remove(rb[0])  # value about to be evicted
        rb.append(x)
        w.update(x)
        ref_means.append(w.mean)
    # Compare the last value to a direct rolling mean.
    expected = np.mean(data[-window:])
    assert ref_means[-1] == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------------------
# MonotonicDeque
# --------------------------------------------------------------------------------------


def _brute_rolling(data, window, kind):
    out = []
    for j in range(len(data)):
        lo = max(0, j - window + 1)
        seg = data[lo : j + 1]
        out.append(max(seg) if kind == "max" else min(seg))
    return out


@pytest.mark.parametrize("kind", ["max", "min"])
def test_monotonic_deque_matches_bruteforce(kind):
    rng = np.random.default_rng(3)
    data = rng.normal(size=1000)
    window = 37
    md = MonotonicDeque(window, kind=kind)
    got = [md.push(float(x)) for x in data]
    expected = _brute_rolling(data.tolist(), window, kind)
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_monotonic_deque_extreme_property():
    md = MonotonicDeque(3, kind="max")
    md.push(1.0)
    md.push(3.0)
    assert md.extreme == 3.0
    md.push(2.0)
    assert md.extreme == 3.0
    md.push(2.0)  # window now [3? no -> 3 expired]; window covers indices 1..3 -> {3,2,2}
    assert md.extreme == 3.0
    md.push(2.0)  # window indices 2..4 -> {2,2,2}
    assert md.extreme == 2.0


def test_monotonic_deque_validation():
    with pytest.raises(ValueError):
        MonotonicDeque(0)
    with pytest.raises(ValueError):
        MonotonicDeque(3, kind="median")
    with pytest.raises(IndexError):
        _ = MonotonicDeque(3).extreme
