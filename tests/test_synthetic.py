"""Tests for the synthetic generator + the physics helpers it relies on.

The coupling-function and cyclic-encoding helpers (used by the generator and feature
layer) are FULLY IMPLEMENTED and tested here, alongside real tests of the physically
plausible synthetic generator (schema conformance, exceedance calibration,
reproducibility, MLT range).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ps14.constants import HARSH_PFU
from ps14.datasets import schema
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
# Synthetic generator
# --------------------------------------------------------------------------------------

# A short span re-used across tests (10 days at 5-min = 2880 samples).
_START = "2014-03-01"
_END = "2014-03-11"


@pytest.fixture(scope="module")
def short_dataset() -> pd.DataFrame:
    """A reproducible 10-day synthetic merged frame at the project cadence."""
    return synthetic.generate_dataset(_START, _END, seed=0, longitude_deg=83.0)


def test_synthetic_params_defaults():
    p = synthetic.SyntheticParams()
    assert p.cadence == "5min"
    assert p.vsw_to_flux_lag_days == pytest.approx(1.5)


def test_generate_dataset_conforms_to_merged_schema(short_dataset):
    # validate_merged raises SchemaError on any violation; a clean return is the assertion.
    problems = schema.validate_merged(short_dataset, raise_on_error=False)
    assert problems == [], f"merged-schema violations: {problems}"


def test_generate_dataset_index_is_uniform_5min(short_dataset):
    idx = short_dataset.index
    assert idx.name == "time"
    assert idx.is_monotonic_increasing
    assert not idx.has_duplicates
    deltas = idx.to_series().diff().dropna().unique()
    assert len(deltas) == 1 and deltas[0] == pd.Timedelta("5min")
    # 10 days at 5-min, half-open grid.
    assert len(short_dataset) == 10 * 24 * 12


def test_generate_dataset_required_columns_present(short_dataset):
    for col in schema.MERGED_REQUIRED:
        assert col in short_dataset.columns
    assert short_dataset["sat_id"].dtype == "category"
    assert short_dataset["flux_e2_imputed"].dtype == np.int8


def test_flux_exceeds_harsh_threshold_sometimes(short_dataset):
    flux = short_dataset["flux_e2"].to_numpy()
    valid = flux[~np.isnan(flux)]
    exceed = (valid >= HARSH_PFU).mean()
    # The generator must produce positive exceedance examples (storm peaks > 1000 pfu)
    # without being saturated -- a realistic minority of samples.
    assert exceed > 0.0, "no samples exceed 1000 pfu (no positive exceedance examples)"
    assert exceed < 0.6, f"implausibly high exceedance fraction {exceed:.2%}"


def test_flux_dynamic_range_is_physical(short_dataset):
    log_flux = short_dataset["log_flux_e2"].dropna().to_numpy()
    # Quiet periods sit around 10^1-10^2 pfu; storm peaks reach but do not wildly exceed
    # the relativistic-electron ceiling (~10^5 pfu).
    assert log_flux.min() >= np.log10(0.01) - 1e-9  # the log floor
    assert log_flux.max() < 5.5
    assert np.nanmedian(short_dataset["flux_e2"].to_numpy()) < HARSH_PFU


def test_generate_dataset_is_reproducible():
    a = synthetic.generate_dataset(_START, _END, seed=0)
    b = synthetic.generate_dataset(_START, _END, seed=0)
    # NaN-aware exact equality on the target and a driver.
    pd.testing.assert_series_equal(a["flux_e2"], b["flux_e2"])
    pd.testing.assert_series_equal(a["vsw"], b["vsw"])


def test_seed_changes_the_realization():
    a = synthetic.generate_dataset(_START, _END, seed=0)
    b = synthetic.generate_dataset(_START, _END, seed=1)
    assert not a["flux_e2"].fillna(-1.0).equals(b["flux_e2"].fillna(-1.0))


def test_mlt_in_range(short_dataset):
    mlt = short_dataset["mlt"].dropna().to_numpy()
    assert (mlt >= 0.0).all() and (mlt < 24.0).all()


def test_log_columns_match_log10_floor(short_dataset):
    from ps14.preprocess.transform import log10_floor

    expected = np.asarray(log10_floor(short_dataset["flux_e2"], floor=0.01))
    got = short_dataset["log_flux_e2"].to_numpy()
    both = ~(np.isnan(expected) | np.isnan(got))
    np.testing.assert_allclose(got[both], expected[both], rtol=1e-9, atol=1e-9)


def test_with_gaps_injects_nans_and_clean_frame_has_none():
    gapped = synthetic.generate_dataset(_START, _END, seed=0, with_gaps=True)
    clean = synthetic.generate_dataset(_START, _END, seed=0, with_gaps=False, with_spikes=False)
    assert gapped["flux_e2"].isna().any(), "with_gaps=True should null some samples"
    assert not clean["flux_e2"].isna().any(), "with_gaps=False should be NaN-free"
    # *_imputed masks stay 0 -- imputation is a downstream preprocess responsibility.
    assert (gapped["flux_e2_imputed"] == 0).all()


def test_pdyn_consistent_with_density_and_speed():
    df = synthetic.generate_dataset(_START, _END, seed=0, with_gaps=False, with_spikes=False)
    expected = offline.dynamic_pressure(
        df["density"].to_numpy(), df["vsw"].to_numpy()
    )
    np.testing.assert_allclose(df["pdyn"].to_numpy(), expected, rtol=1e-6)


def test_indices_have_correct_sign_conventions():
    df = synthetic.generate_dataset(_START, _END, seed=0, with_gaps=False, with_spikes=False)
    assert (df["ae"].to_numpy() >= 0.0).all(), "AE must be non-negative"
    assert (df["al"].to_numpy() <= 0.0).all(), "AL must be non-positive"
    assert df["sym_h"].min() < 0.0, "SYM-H should go negative during storms"


def test_exceedance_fraction_helper(short_dataset):
    frac = synthetic.exceedance_fraction(short_dataset)
    assert 0.0 < frac < 0.6
