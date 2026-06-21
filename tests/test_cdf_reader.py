"""Tests for the CDF reader (ps14.io.cdf_reader).

The pure ``mask_invalid`` helper is tested in isolation; the cdflib-backed reader is
exercised end-to-end by round-tripping the CDFs written by ``ps14.io.synthetic`` and
comparing values within tolerance. Tests needing ``cdflib`` are guarded with
``pytest.importorskip`` so they skip cleanly when the optional wheel is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ps14.io import cdf_reader

cdflib = pytest.importorskip("cdflib")


def test_mask_invalid_fillval():
    data = np.array([1.0, -1.0e31, 3.0, -1.0e31])
    out = cdf_reader.mask_invalid(data, fillval=-1.0e31, validmin=None, validmax=None)
    assert np.isnan(out[1]) and np.isnan(out[3])
    assert out[0] == 1.0 and out[2] == 3.0


def test_mask_invalid_valid_range():
    data = np.array([-5.0, 0.0, 50.0, 200.0])
    out = cdf_reader.mask_invalid(data, fillval=None, validmin=0.0, validmax=100.0)
    assert np.isnan(out[0])  # below min
    assert np.isnan(out[3])  # above max
    assert out[1] == 0.0 and out[2] == 50.0


def test_mask_invalid_does_not_mutate_input():
    data = np.array([1.0, -1.0e31])
    _ = cdf_reader.mask_invalid(data, fillval=-1.0e31, validmin=None, validmax=None)
    assert data[1] == -1.0e31  # original untouched


def test_mask_invalid_no_attrs_is_noop():
    data = np.array([1.0, 2.0, 3.0])
    out = cdf_reader.mask_invalid(data, fillval=None, validmin=None, validmax=None)
    np.testing.assert_array_equal(out, data)


def test_variable_meta_dataclass():
    meta = cdf_reader.CdfVariableMeta(name="E2", depend_0="Epoch", units="cm^-2 s^-1 sr^-1")
    assert meta.name == "E2"
    assert meta.depend_0 == "Epoch"
    assert meta.extra == {}


# --------------------------------------------------------------------------------------
# Round-trip: write CDFs with the synthetic generator, then read them back.
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def written_cdfs(tmp_path_factory):
    """Generate a short merged frame and write GOES + OMNI CDFs; return (df, paths)."""
    from ps14.io import synthetic

    out_dir = tmp_path_factory.mktemp("synthetic_cdf")
    df = synthetic.generate_dataset("2014-03-01", "2014-03-06", seed=0, longitude_deg=83.0)
    params = synthetic.SyntheticParams(longitude_deg=83.0)
    goes = synthetic.write_cdf(df, out_dir / "goes.cdf", kind="goes", params=params)
    omni = synthetic.write_cdf(df, out_dir / "omni.cdf", kind="omni", params=params)
    return df, {"goes": goes, "omni": omni}


def test_epoch_roundtrip_is_leap_aware():
    # A span crossing the 2015-06-30 leap second must round-trip to the nanosecond.
    idx = pd.date_range("2015-06-30 23:55", periods=4, freq="5min", name="time").as_unit("ns")
    from ps14.io.synthetic import _datetime_to_tt2000

    tt2000 = _datetime_to_tt2000(idx)
    back = pd.DatetimeIndex(cdf_reader.epoch_to_datetime(tt2000))
    assert (back == idx).all()
    assert back.dtype == np.dtype("datetime64[ns]")


def test_list_data_variables_finds_science_vars(written_cdfs):
    _, paths = written_cdfs
    goes_vars = cdf_reader.list_data_variables(paths["goes"])
    assert set(goes_vars) == {"flux_e2", "flux_seed", "mlt"}
    # The epoch support variable is excluded from the data-variable listing.
    assert "Epoch" not in goes_vars


def test_get_variable_meta_reads_istp_attrs(written_cdfs):
    _, paths = written_cdfs
    meta = cdf_reader.get_variable_meta(paths["goes"], "flux_e2")
    assert meta.depend_0 == "Epoch"
    assert meta.var_type == "data"
    assert meta.fillval == pytest.approx(-1.0e31)
    assert "cm^-2" in meta.units


def test_read_cdf_roundtrips_goes_values(written_cdfs):
    df, paths = written_cdfs
    got = cdf_reader.read_cdf(paths["goes"])
    assert got.index.name == "time"
    assert got.index.dtype == np.dtype("datetime64[ns]")
    assert (got.index == df.index).all()
    for col in ("flux_e2", "flux_seed", "mlt"):
        src = df[col].to_numpy()
        out = got[col].to_numpy()
        both = ~(np.isnan(src) | np.isnan(out))
        np.testing.assert_allclose(out[both], src[both], rtol=1e-9, atol=1e-6)
        # FILLVAL round-trips to NaN: missing positions must coincide exactly.
        np.testing.assert_array_equal(np.isnan(src), np.isnan(out))


def test_read_cdf_roundtrips_omni_values(written_cdfs):
    df, paths = written_cdfs
    got = cdf_reader.read_cdf(paths["omni"])
    for col in ("vsw", "density", "bz_gsm", "bt", "ae", "al", "kp", "sym_h", "f107"):
        src = df[col].to_numpy()
        out = got[col].to_numpy()
        both = ~(np.isnan(src) | np.isnan(out))
        np.testing.assert_allclose(out[both], src[both], rtol=1e-9, atol=1e-6)


def test_read_cdf_variable_carries_metadata(written_cdfs):
    _, paths = written_cdfs
    series = cdf_reader.read_cdf_variable(paths["goes"], "flux_e2")
    assert series.index.name == "time"
    assert series.attrs["source_var"] == "flux_e2"
    assert "cm^-2" in series.attrs["units"]


def test_read_cdf_subset_of_variables(written_cdfs):
    _, paths = written_cdfs
    got = cdf_reader.read_cdf(paths["omni"], variables=["vsw", "kp"])
    assert list(got.columns) == ["vsw", "kp"]


def test_read_cdf_to_dataframe_alias(written_cdfs):
    _, paths = written_cdfs
    assert cdf_reader.read_cdf_to_dataframe is cdf_reader.read_cdf_to_frame
    got = cdf_reader.read_cdf_to_dataframe(paths["goes"], variables=["mlt"])
    assert list(got.columns) == ["mlt"]
