"""CDF reader (cdflib-based) honouring ISTP metadata conventions.

Implements the CDF read contract (CONTRACTS.md §1, R5 §1-2): discover zVariables, keep
``VAR_TYPE == "data"``, follow ``DEPEND_0`` to the epoch, mask ``FILLVAL`` and
out-of-``[VALIDMIN, VALIDMAX]`` to NaN, apply any quality flag, and convert the epoch
(TT2000/EPOCH/EPOCH16) leap-aware to ``datetime64[ns]`` UTC.

``cdflib`` is a pure-Python dependency (a core dependency of the package), so it is
imported lazily inside the functions only to keep ``import ps14`` cheap and to give a
clear error if the wheel is somehow absent.
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


def _import_cdflib():
    """Lazily import :mod:`cdflib`, raising an actionable error if it is missing."""
    try:
        import cdflib  # noqa: PLC0415  (lazy import is intentional)
    except ImportError as exc:  # pragma: no cover - cdflib is a core dependency
        raise ImportError(
            "cdflib is required to read CDF files. Install it with `pip install cdflib` "
            "(it is a core dependency declared in pyproject.toml)."
        ) from exc
    return cdflib


def _coerce_scalar(value: Any) -> float | None:
    """Coerce a possibly-array ISTP attribute (FILLVAL/VALIDMIN/...) to a float scalar.

    ISTP scalar attributes are sometimes returned by cdflib as 0-d/1-element arrays or
    strings. Returns ``None`` when the value is absent or not numerically interpretable.
    """
    if value is None:
        return None
    arr = np.asarray(value).ravel()
    if arr.size == 0:
        return None
    try:
        return float(arr[0])
    except (TypeError, ValueError):
        return None


def _attr_to_str(value: Any) -> str:
    """Coerce an ISTP string attribute (possibly list/array) to a plain ``str``."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value).ravel()
        return str(arr[0]) if arr.size else ""
    return str(value)


def list_variables(path: str | Path) -> list[str]:
    """Return the names of all zVariables in a CDF (data + support axes).

    Parameters
    ----------
    path:
        Path to a ``.cdf`` file.

    Returns
    -------
    list[str]
        Every zVariable name, in file order (use :func:`list_data_variables` to filter to
        the modellable measurements).
    """
    cdflib = _import_cdflib()
    info = cdflib.CDF(str(path)).cdf_info()
    return list(info.zVariables)


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
        measurements), excluding ``support_data``/``metadata`` axes. If no variable
        declares ``VAR_TYPE`` (non-ISTP file) every non-epoch zVariable is returned so the
        reader still does something useful.
    """
    cdflib = _import_cdflib()
    cdf = cdflib.CDF(str(path))
    info = cdf.cdf_info()
    data_vars: list[str] = []
    saw_var_type = False
    for var in info.zVariables:
        atts = cdf.varattsget(var)
        var_type = _attr_to_str(atts.get("VAR_TYPE", "")).lower()
        if var_type:
            saw_var_type = True
        if var_type == "data":
            data_vars.append(var)
    if not saw_var_type:
        # Non-ISTP file: treat any variable that is not an epoch axis as data.
        data_vars = [v for v in info.zVariables if "epoch" not in v.lower()]
    return data_vars


def get_variable_meta(path: str | Path, var: str) -> CdfVariableMeta:
    """Read the ISTP variable attributes needed for masking + time alignment.

    Reads ``DEPEND_0``, ``FILLVAL``, ``VALIDMIN``/``VALIDMAX``, ``UNITS``, ``CATDESC``,
    ``VAR_TYPE`` via ``cdflib.CDF(path).varattsget(var)``.
    """
    cdflib = _import_cdflib()
    atts = cdflib.CDF(str(path)).varattsget(var)
    known = {"DEPEND_0", "FILLVAL", "VALIDMIN", "VALIDMAX", "UNITS", "CATDESC", "VAR_TYPE"}
    extra = {k: v for k, v in atts.items() if k not in known}
    depend_0 = atts.get("DEPEND_0")
    return CdfVariableMeta(
        name=var,
        depend_0=str(depend_0) if depend_0 is not None else None,
        fillval=_coerce_scalar(atts.get("FILLVAL")),
        validmin=_coerce_scalar(atts.get("VALIDMIN")),
        validmax=_coerce_scalar(atts.get("VALIDMAX")),
        units=_attr_to_str(atts.get("UNITS")),
        catdesc=_attr_to_str(atts.get("CATDESC")),
        var_type=_attr_to_str(atts.get("VAR_TYPE")),
        extra=extra,
    )


def epoch_to_datetime(epoch: np.ndarray) -> np.ndarray:
    """Convert a CDF epoch array to ``datetime64[ns]`` (UTC), leap-aware.

    Works for ``CDF_TIME_TT2000`` (int64), ``CDF_EPOCH`` (float ms), and
    ``CDF_EPOCH16`` (complex ps) via ``cdflib.cdfepoch.to_datetime`` (R5 §1.2/§2.2).
    Always returns nanosecond resolution to satisfy the canonical time contract.
    """
    cdflib = _import_cdflib()
    dt = np.asarray(cdflib.cdfepoch.to_datetime(epoch))
    return dt.astype("datetime64[ns]")


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
    cdflib = _import_cdflib()
    cdf = cdflib.CDF(str(path))
    meta = get_variable_meta(path, var)

    data = np.asarray(cdf.varget(var), dtype="float64")
    if meta.depend_0 is not None:
        epoch = cdf.varget(meta.depend_0)
    else:
        # Fall back to the first epoch-like variable in the file.
        info = cdf.cdf_info()
        epoch_var = next((v for v in info.zVariables if "epoch" in v.lower()), None)
        if epoch_var is None:
            raise ValueError(f"variable {var!r} has no DEPEND_0 and no epoch variable found")
        epoch = cdf.varget(epoch_var)

    times = epoch_to_datetime(epoch)
    data = mask_invalid(data, fillval=meta.fillval, validmin=meta.validmin, validmax=meta.validmax)

    series = pd.Series(data, index=pd.DatetimeIndex(times, name="time"), name=var, dtype="float64")
    series.attrs["units"] = meta.units
    series.attrs["catdesc"] = meta.catdesc
    series.attrs["source_var"] = var
    return series


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
    if variables is None:
        variables = list_data_variables(path)
    if not variables:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="time"))

    frame: pd.DataFrame | None = None
    for var in variables:
        series = read_cdf_variable(path, var)
        # Collapse duplicate timestamps (defensive) keeping the last observation.
        if series.index.has_duplicates:
            series = series[~series.index.duplicated(keep="last")]
        col = series.to_frame()
        frame = col if frame is None else frame.join(col, how="outer")

    assert frame is not None  # noqa: S101 (variables is non-empty here)
    frame = frame.sort_index()
    frame.index.name = "time"
    return frame


def read_cdf(path: str | Path, variables: list[str] | None = None) -> pd.DataFrame:
    """Read a CDF into a tidy, UTC-indexed dataframe (CONTRACTS.md §1).

    Thin alias over :func:`read_cdf_to_frame`: discovers ``VAR_TYPE == "data"``
    variables (or uses the requested subset), follows each ``DEPEND_0`` to its epoch,
    masks fill/valid, and converts to a ``datetime64[ns]`` index named ``"time"``.

    Parameters
    ----------
    path:
        Path to the ``.cdf`` file.
    variables:
        Optional subset of variable names; defaults to all science variables.

    Returns
    -------
    pd.DataFrame
        Time-indexed frame, one column per variable.
    """
    return read_cdf_to_frame(path, variables)


# Backwards-compatible alias requested by the data-layer contract.
read_cdf_to_dataframe = read_cdf_to_frame


__all__ = [
    "CdfVariableMeta",
    "list_variables",
    "list_data_variables",
    "get_variable_meta",
    "epoch_to_datetime",
    "mask_invalid",
    "read_cdf_variable",
    "read_cdf_to_frame",
    "read_cdf_to_dataframe",
    "read_cdf",
]
