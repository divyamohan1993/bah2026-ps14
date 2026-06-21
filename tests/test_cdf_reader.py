"""Tests for the CDF reader (ps14.io.cdf_reader).

The pure ``mask_invalid`` helper is FULLY IMPLEMENTED and tested; the cdflib-backed
reader functions are stubs (asserted to raise until implemented) so the contract surface
is exercised without requiring real CDF files.
"""

from __future__ import annotations

import numpy as np
import pytest

from ps14.io import cdf_reader


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


@pytest.mark.parametrize(
    "func, args",
    [
        (cdf_reader.list_data_variables, ("missing.cdf",)),
        (cdf_reader.get_variable_meta, ("missing.cdf", "E2")),
        (cdf_reader.read_cdf_variable, ("missing.cdf", "E2")),
        (cdf_reader.epoch_to_datetime, (np.array([0]),)),
    ],
)
def test_cdf_reader_stubs_raise(func, args):
    with pytest.raises(NotImplementedError):
        func(*args)
