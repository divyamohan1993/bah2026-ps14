"""Tests for the synthetic generator + the physics helpers it relies on.

The coupling-function and cyclic-encoding helpers (used by the generator and feature
layer) are FULLY IMPLEMENTED and tested here; the generator orchestration functions are
stubs (asserted to raise until implemented).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ps14.features import offline
from ps14.io import synthetic
from ps14.utils import timeops

# --------------------------------------------------------------------------------------
# Coupling functions (physics the generator + features share)
# --------------------------------------------------------------------------------------


def test_vbs_is_zero_for_northward_bz():
    v = np.array([400.0, 400.0])
    bz = np.array([5.0, -5.0])  # north, south
    out = offline.vbs(v, bz)
    assert out[0] == 0.0  # northward -> no merging E-field
    assert out[1] > 0.0  # southward -> positive vBs


def test_dynamic_pressure_scaling():
    # Doubling speed quadruples dynamic pressure (Pdyn ~ N V^2).
    p1 = offline.dynamic_pressure(np.array([5.0]), np.array([400.0]))
    p2 = offline.dynamic_pressure(np.array([5.0]), np.array([800.0]))
    assert p2[0] == pytest.approx(4.0 * p1[0], rel=1e-9)
    # Sanity: ~5 cm^-3 at 400 km/s gives a few nPa.
    assert 1.0 < p1[0] < 5.0


def test_newell_coupling_nonnegative_and_zero_when_no_transverse_field():
    v = np.array([500.0])
    out_zero = offline.newell_coupling(v, np.array([0.0]), np.array([10.0]))  # purely northward
    out_south = offline.newell_coupling(v, np.array([0.0]), np.array([-10.0]))
    assert out_zero[0] == pytest.approx(0.0)  # sin(theta_c/2)=0 for north
    assert out_south[0] > 0.0


def test_clock_angle_sign():
    # Bz north -> ~0; Bz south -> ~pi.
    assert offline.clock_angle(np.array([0.0]), np.array([5.0]))[0] == pytest.approx(0.0)
    assert abs(offline.clock_angle(np.array([0.0]), np.array([-5.0]))[0]) == pytest.approx(np.pi)


def test_epsilon_nonnegative():
    out = offline.epsilon_coupling(np.array([500.0]), np.array([10.0]), np.array([np.pi]))
    assert out[0] > 0.0


def test_shue_standoff_compresses_with_pressure():
    # Higher dynamic pressure -> smaller standoff distance.
    r_low = offline.shue_standoff(np.array([1.0]), np.array([0.0]))
    r_high = offline.shue_standoff(np.array([10.0]), np.array([0.0]))
    assert r_high[0] < r_low[0]
    assert 5.0 < r_low[0] < 12.0  # plausible R_E range


# --------------------------------------------------------------------------------------
# Cyclic encodings (known-future covariates)
# --------------------------------------------------------------------------------------


def test_cyclic_encode_unit_circle():
    s, c = timeops.cyclic_encode(np.array([0.0, 6.0, 12.0, 18.0]), period=24.0)
    np.testing.assert_allclose(s**2 + c**2, np.ones(4), atol=1e-12)
    # 0h -> (sin,cos)=(0,1); 6h (quarter) -> (1,0).
    assert s[0] == pytest.approx(0.0) and c[0] == pytest.approx(1.0)
    assert s[1] == pytest.approx(1.0, abs=1e-12)


def test_time_of_day_and_doy_encoding_shapes():
    idx = pd.date_range("2020-01-01", periods=48, freq="h", name="time")
    tod_s, tod_c = timeops.time_of_day_encoding(idx)
    doy_s, doy_c = timeops.day_of_year_encoding(idx)
    assert tod_s.shape == (48,) and doy_c.shape == (48,)
    np.testing.assert_allclose(tod_s**2 + tod_c**2, np.ones(48), atol=1e-12)


def test_ballistic_lag_minutes_plausible():
    lag = timeops.ballistic_lag_minutes(np.array([400.0, 800.0]))
    # ~1.5e6 km / 400 km/s / 60 ~ 62 min; faster wind -> shorter lag.
    assert lag[0] > lag[1]
    assert 40.0 < lag[0] < 80.0


# --------------------------------------------------------------------------------------
# Generator stubs
# --------------------------------------------------------------------------------------


def test_synthetic_params_defaults():
    p = synthetic.SyntheticParams()
    assert p.cadence == "5min"
    assert p.vsw_to_flux_lag_days == pytest.approx(1.5)


def test_generate_stub_raises():
    with pytest.raises(NotImplementedError):
        synthetic.generate(synthetic.SyntheticParams())
