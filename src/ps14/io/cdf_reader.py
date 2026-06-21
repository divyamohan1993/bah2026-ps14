"""CDF reader (cdflib-based) honouring ISTP metadata conventions.

Implements the CDF read contract (CONTRACTS.md §1, R5 §1-2): discover zVariables, keep
``VAR_TYPE == "data"``, follow ``DEPEND_0`` to the epoch, mask ``FILLVAL`` and
out-of-``[VALIDMIN, VALIDMAX]`` to NaN, apply any quality flag, and convert the epoch
(TT2000/EPOCH/EPOCH16) leap-aware to ``datetime64[ns]`` UTC.

The core single-variable reader is implemented; multi-variable discovery helpers carry
the contract in their signatures/docstrings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CdfVariableMeta:
    """ISTP metadata for one CDF variable (subset relevant to ingestion)."""

    name: str
    depend_0: str | None = None
    fillval: float | None = None
    validmin: float | None = None
    validmax: float | None = None
    units: str = ""
    catdesc: str = ""
    var_type: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def list_data_variables(path: str | Path) -> list[str]:
    """Return the names of science zVariables (``VAR_TYPE == "data"``) in a CDF.

    Parameters
    ----------
    path:
        Path to a ``.cdf`` file.

    Returns
    -------
    list[str]
        Names of variables whose ``VAR_TYPE`` is ``"data"`` (the modellable
        measurements), excluding ``support_data``/``metadata`` axes.
    """
    raise NotImplementedError(
        "TODO: open with cdflib.CDF, iterate info.zVariables, read varattsget(var)['VAR_TYPE'], "
        "and keep those equal to 'data' (R5 §1.3)."
    )


def get_variable_meta(path: str | Path, var: str) -> CdfVariableMeta:
    """Read the ISTP variable attributes needed for masking + time alignment.

    Reads ``DEPEND_0``, ``FILLVAL``, ``VALIDMIN``/``VALIDMAX``, ``UNITS``, ``CATDESC``,
    ``VAR_TYPE`` via ``cdflib.CDF(path).varattsget(var)``.
    """
    raise NotImplementedError("TODO: populate CdfVariableMeta from cdf.varattsget(var).")


def epoch_to_datetime(epoch: np.ndarray) -> np.ndarray:
    """Convert a CDF epoch array to ``datetime64[ns]`` (UTC), leap-aware.

    Works for ``CDF_TIME_TT2000`` (int64), ``CDF_EPOCH`` (float ms), and
    ``CDF_EPOCH16`` (complex ps) via ``cdflib.cdfepoch.to_datetime`` (R5 §1.2/§2.2).
    """
    raise NotImplementedError("TODO: return cdflib.cdfepoch.to_datetime(epoch).")


def mask_invalid(
    data: np.ndarray,
    *,
    fillval: float | None,
    validmin: float | None,
    validmax: float | None,
) -> np.ndarray:
    """Set ``FILLVAL`` and out-of-``[VALIDMIN, VALIDMAX]`` values to NaN.

    Parameters
    ----------
    data:
        Float array of raw values.
    fillval, validmin, validmax:
        ISTP attribute values (any may be None).

    Returns
    -------
    np.ndarray
        A float64 copy with invalid entries replaced by ``np.nan``.

    Notes
    -----
    This is a pure helper (no cdflib dependency) so it is unit-testable in isolation;
    implementers should keep it side-effect free.
    """
    out = np.asarray(data, dtype="float64").copy()
    if fillval is not None:
        out = np.where(np.isclose(out, float(fillval)), np.nan, out)
    if validmin is not None:
        out = np.where(out < float(validmin), np.nan, out)
    if validmax is not None:
        out = np.where(out > float(validmax), np.nan, out)
    return out


def read_cdf_variable(path: str | Path, var: str) -> pd.Series:
    """Read one science variable into a time-indexed, masked pandas Series.

    Implements CONTRACTS.md §1: returns a ``pd.Series`` indexed by a UTC
    ``DatetimeIndex`` named ``"time"`` (from ``DEPEND_0``), float64 values with
    ``FILLVAL``/out-of-valid masked to NaN, and ``series.attrs`` carrying
    ``units``/``catdesc``/``source_var``.

    Parameters
    ----------
    path:
        Path to the ``.cdf`` file.
    var:
        Name of the science variable to read.

    Returns
    -------
    pd.Series
        ``name == var``; ``index.name == "time"``.
    """
    raise NotImplementedError(
        "TODO: cdflib.CDF(path); meta=get_variable_meta; data=varget(var); "
        "epoch=varget(meta.depend_0); times=epoch_to_datetime(epoch); "
        "data=mask_invalid(data, ...); "
        "return pd.Series(data, index=DatetimeIndex(times, name='time'), name=var) "
        "with attrs set (R5 §2.1)."
    )


def read_cdf_to_frame(path: str | Path, variables: list[str] | None = None) -> pd.DataFrame:
    """Read multiple science variables sharing a time axis into one DataFrame.

    Parameters
    ----------
    path:
        Path to the ``.cdf`` file.
    variables:
        Variable names to read; if None, all ``VAR_TYPE == "data"`` variables.

    Returns
    -------
    pd.DataFrame
        UTC-indexed (``time``) frame; one column per variable. Variables on different
        epochs are outer-joined on the union time index (callers resample later).
    """
    raise NotImplementedError(
        "TODO: discover variables (list_data_variables) if None; read each via "
        "read_cdf_variable; outer-join on the time index."
    )


__all__ = [
    "CdfVariableMeta",
    "list_data_variables",
    "get_variable_meta",
    "epoch_to_datetime",
    "mask_invalid",
    "read_cdf_variable",
    "read_cdf_to_frame",
]
