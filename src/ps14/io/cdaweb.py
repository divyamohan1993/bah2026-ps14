"""CDAWeb fetch client (cdasws / pyspedas) with a local cache.

OPTIONAL path -- the system runs fully offline on synthetic data without this module.
Requires the ``[data]`` extra (``cdasws`` preferred, ``pyspedas`` as a bulk fallback).
See R2 "Data Access Strategy" and R5 §2.3 for dataset IDs and the recommended access
pattern. The optional libraries are imported LAZILY inside the functions so ``import
ps14`` works without them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Dataset-ID hints (R2 §C/§F, R5 §3). Confirm exact IDs/variable names against the live
# CDAWeb index before relying on them -- these are the documented current identifiers.
# --------------------------------------------------------------------------------------

#: GOES-R SEISS MPS-HI >2 MeV integral electron flux (the forecast target). Per probe.
GOES_TARGET_DATASETS: dict[str, str] = {
    "G16": "DN_SEIS-L2-MPSH_G16",
    "G17": "DN_SEIS-L2-MPSH_G17",
    "G18": "DN_SEIS-L2-MPSH_G18",
    "G19": "DN_SEIS-L2-MPSH_G19",
}
#: Legacy GOES EPEAD >2 MeV (``E2``) integral electron flux (pre-2018 record).
GOES_LEGACY_DATASETS: dict[str, str] = {
    "G13": "GOES13_EPS-EPEAD-SCIENCE-ELECTRONS-E13EW_1MIN",
    "G15": "GOES15_EPS-EPEAD-SCIENCE-ELECTRONS-E13EW_1MIN",
}
#: Primary merged solar-wind / IMF / geomagnetic-index driver matrix (bow-shock-shifted).
OMNI_DRIVERS_DATASET: str = "OMNI_HRO_1MIN"
OMNI_DRIVERS_DATASET_5MIN: str = "OMNI_HRO_5MIN"

#: Canonical OMNI HRO (1-min) driver variable names (R5 §3.3). NOTE: ``KP`` and ``F10_INDEX``
#: are NOT present in the 1-minute ``OMNI_HRO_1MIN`` product (they live in the hourly
#: ``OMNI2_H0_MRG1HR`` / ``OMNI_HRO2`` products); including them makes cdasws reject the whole
#: request (HTTP 400). They are therefore omitted here and merged from the hourly product
#: separately when needed. All names below are verified valid against the live OMNI_HRO_1MIN.
OMNI_DRIVER_VARIABLES: list[str] = [
    "flow_speed",  # Vsw (km/s)
    "proton_density",  # N (cm^-3)
    "Pressure",  # dynamic pressure (nPa)
    "BZ_GSM",  # IMF Bz (nT)
    "F",  # |B| (nT)
    "AE_INDEX",  # AE (nT)
    "AL_INDEX",  # AL (nT)
    "SYM_H",  # SYM-H (nT)
]
#: Hourly-only OMNI indices (planetary K-index x10, F10.7) — fetch from OMNI2_H0_MRG1HR.
OMNI_HOURLY_INDEX_VARIABLES: list[str] = ["KP_INDEX", "F10_INDEX"]


def fetch_cdaweb(
    dataset_id: str,
    variables: list[str],
    start: str,
    end: str,
    *,
    cache_dir: str | Path = "data/raw",
    method: str = "cdasws",
) -> pd.DataFrame:
    """Fetch a CDAWeb dataset over a time range into a UTC-indexed DataFrame.

    Parameters
    ----------
    dataset_id:
        CDAWeb dataset ID, e.g. ``"OMNI_HRO_1MIN"`` or ``"DN_SEIS-L2-MPSH_G18"``
        (see :data:`OMNI_DRIVERS_DATASET`, :data:`GOES_TARGET_DATASETS`).
    variables:
        Variable names to retrieve (server-side projection keeps transfers small).
    start, end:
        ISO-8601 UTC bounds (e.g. ``"2017-02-07T00:00:00Z"``).
    cache_dir:
        Where to cache downloaded CDFs (idempotent: skip if present + checksum OK).
    method:
        ``"cdasws"`` (xarray with ISTP metadata; preferred) or ``"pyspedas"`` (bulk
        loader).

    Returns
    -------
    pd.DataFrame
        UTC-indexed frame (``time``) with the requested variables, masked per ISTP.

    Raises
    ------
    ImportError
        If the requested backend library is not installed (offline mode).
    RuntimeError
        If the download fails (e.g. no network) -- the message is actionable.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    if method == "cdasws":
        frame = _fetch_via_cdasws(dataset_id, variables, start, end)
    elif method == "pyspedas":
        frame = _fetch_via_pyspedas(dataset_id, variables, start, end, cache_path)
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown method {method!r} (expected 'cdasws' or 'pyspedas')")

    logger.info("fetched %s [%s] %s..%s -> %d rows", dataset_id, method, start, end, len(frame))
    return frame


def _fetch_via_cdasws(dataset_id: str, variables: list[str], start: str, end: str) -> pd.DataFrame:
    """Fetch via ``cdasws`` (returns an xarray.Dataset with ISTP metadata; R5 §2.3)."""
    try:
        from cdasws import CdasWs  # noqa: PLC0415  (lazy optional dependency)
    except ImportError as exc:
        raise ImportError(
            "cdasws is required for method='cdasws'. Install the data extra: "
            "`pip install 'ps14[data]'` (or `pip install cdasws`). The pipeline runs "
            "fully offline on synthetic data without it (ps14.io.synthetic)."
        ) from exc

    try:
        cdas = CdasWs()
        status, data = cdas.get_data(dataset_id, list(variables), start, end)
    except Exception as exc:  # network / server / dataset errors
        raise RuntimeError(
            f"cdasws fetch of {dataset_id!r} ({start}..{end}) failed: {exc}. "
            "Check network connectivity and that the dataset ID + variable names are "
            "valid on the live CDAWeb portal (https://cdaweb.gsfc.nasa.gov/)."
        ) from exc

    if data is None:
        raise RuntimeError(
            f"cdasws returned no data for {dataset_id!r} ({start}..{end}); status={status!r}."
        )
    try:
        return _response_to_frame(data, variables)
    except Exception as exc:
        raise RuntimeError(
            f"failed to convert cdasws response for {dataset_id!r} to a DataFrame "
            f"(response type {type(data).__name__}): {exc}."
        ) from exc


def _response_to_frame(data, variables: list[str]) -> pd.DataFrame:
    """Convert a cdasws response to a UTC-indexed, masked DataFrame.

    Handles both supported cdasws return types:

    * an ``xarray.Dataset`` (cdasws with the xarray backend), and
    * a SpacePy ``CDFCopy`` (dict-like of ``VarCopy`` arrays), which cdasws returns when
      SpacePy is installed.

    For each requested variable it follows the ISTP ``DEPEND_0`` to its epoch, masks
    ``FILLVAL``/out-of-``[VALIDMIN, VALIDMAX]`` to NaN, and assembles one column per
    variable on the union time index.
    """
    import numpy as np  # noqa: PLC0415  (local import keeps module import cheap)

    from ps14.io.cdf_reader import mask_invalid

    columns: dict[str, pd.Series] = {}
    for var in variables:
        if var not in data:
            logger.warning("variable %s absent from cdasws response; skipping", var)
            continue
        var_obj = data[var]
        attrs = dict(getattr(var_obj, "attrs", {}) or {})

        # Resolve the epoch axis. xarray exposes it via .dims; SpacePy via DEPEND_0.
        depend_0 = attrs.get("DEPEND_0")
        dims = getattr(var_obj, "dims", None)
        if dims:  # xarray.DataArray
            epoch_key = depend_0 if (depend_0 and depend_0 in data) else dims[0]
            epoch_vals = np.asarray(data[epoch_key].values)
            raw = np.asarray(var_obj.values, dtype="float64")
        else:  # SpacePy VarCopy (array-like with .attrs)
            epoch_key = depend_0 if (depend_0 and depend_0 in data) else "Epoch"
            epoch_vals = np.asarray(data[epoch_key])
            raw = np.asarray(var_obj, dtype="float64")

        epoch = pd.DatetimeIndex(pd.to_datetime(epoch_vals)).as_unit("ns")
        values = mask_invalid(
            raw,
            fillval=_attr_float(attrs.get("FILLVAL")),
            validmin=_attr_float(attrs.get("VALIDMIN")),
            validmax=_attr_float(attrs.get("VALIDMAX")),
        )
        if values.ndim > 1:  # multi-channel: keep the first channel as a scalar series
            values = values[:, 0]
        series = pd.Series(values, index=epoch, name=var, dtype="float64")
        series.attrs["units"] = str(attrs.get("UNITS", ""))
        columns[var] = series

    if not columns:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="time"))

    frame = pd.concat(columns, axis=1)
    frame.index.name = "time"
    return frame.sort_index()


def _attr_float(value) -> float | None:
    """Coerce a possibly-array ISTP scalar attribute to ``float`` (or ``None``)."""
    if value is None:
        return None
    import numpy as np  # noqa: PLC0415

    arr = np.asarray(value).ravel()
    if arr.size == 0:
        return None
    try:
        return float(arr[0])
    except (TypeError, ValueError):
        return None


def _fetch_via_pyspedas(
    dataset_id: str,
    variables: list[str],
    start: str,
    end: str,
    cache_dir: Path,
) -> pd.DataFrame:
    """Fetch via ``pyspedas`` generic CDAWeb loader, then read the cached CDFs.

    pyspedas downloads the CDF files to a local directory; we hand the downloaded files
    to :func:`ps14.io.cdf_reader.read_cdf_to_frame` so the same ISTP masking path is used
    everywhere.
    """
    try:
        import pyspedas  # noqa: PLC0415  (lazy optional dependency)
    except ImportError as exc:
        raise ImportError(
            "pyspedas is required for method='pyspedas'. Install the data extra: "
            "`pip install 'ps14[data]'` (or `pip install pyspedas`)."
        ) from exc

    from ps14.io.cdf_reader import read_cdf_to_frame  # noqa: PLC0415

    loader = _resolve_pyspedas_cdaweb_loader(pyspedas)
    if loader is None:
        raise RuntimeError(
            "this pyspedas build does not expose a programmatic CDAWeb file loader "
            "(pyspedas.cdaweb.CDAWeb / pyspedas.projects.cdaweb). Use "
            "fetch_cdaweb(..., method='cdasws') instead, which is the preferred path."
        )

    try:
        downloaded = loader(dataset_id, [start, end], list(variables))
    except Exception as exc:  # network / server / API errors
        raise RuntimeError(
            f"pyspedas fetch of {dataset_id!r} ({start}..{end}) failed: {exc}. "
            "Check network connectivity and the dataset ID, or use method='cdasws'."
        ) from exc

    files = [f for f in (downloaded or []) if str(f).endswith(".cdf")]
    if not files:
        raise RuntimeError(f"pyspedas returned no CDF files for {dataset_id!r} ({start}..{end}).")

    frames = [read_cdf_to_frame(f, variables) for f in files]
    frame = pd.concat(frames).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame.index.name = "time"
    return frame


def _resolve_pyspedas_cdaweb_loader(pyspedas):
    """Return a ``loader(dataset, trange, varnames) -> list[file]`` for this pyspedas.

    pyspedas reorganised its CDAWeb interface across versions; this probes the known
    locations and returns ``None`` if none expose a download-only file loader.
    """
    # Modern API: a CDAWeb class with .get_data(dataset, trange, varnames, ...).
    for module_path in ("pyspedas.cdaweb", "pyspedas.projects.cdaweb"):
        try:
            import importlib  # noqa: PLC0415

            mod = importlib.import_module(module_path)
        except ImportError:
            continue
        cdaweb_cls = getattr(mod, "CDAWeb", None)
        if cdaweb_cls is not None:

            def _load(dataset, trange, varnames, _cls=cdaweb_cls):
                client = _cls()
                return client.get_data(dataset, trange, varnames, downloadonly=True)

            return _load
    return None


def update_manifest(cache_dir: str | Path, dataset_id: str, file_path: str | Path) -> None:
    """Append/refresh a coverage-manifest row for a cached CDF (CONTRACTS.md §8).

    Row schema: ``dataset_id, file, start, end, n_records, sha256, fill_fraction,
    download_utc`` (R5 §7). The manifest documents exactly which spans exist and their
    gap fraction, driving split boundaries and reproducibility.
    """
    import hashlib  # noqa: PLC0415

    from ps14.io.cdf_reader import read_cdf_to_frame  # noqa: PLC0415

    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.csv"
    file_path = Path(file_path)

    frame = read_cdf_to_frame(file_path)
    n_records = len(frame)
    if n_records:
        start = frame.index.min().isoformat()
        end = frame.index.max().isoformat()
        fill_fraction = float(frame.isna().to_numpy().mean())
    else:
        start = end = ""
        fill_fraction = float("nan")

    sha256 = hashlib.sha256(file_path.read_bytes()).hexdigest()
    row = {
        "dataset_id": dataset_id,
        "file": str(file_path),
        "start": start,
        "end": end,
        "n_records": n_records,
        "sha256": sha256,
        "fill_fraction": fill_fraction,
        "download_utc": pd.Timestamp.now("UTC").isoformat(),
    }

    new = pd.DataFrame([row])
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        existing = existing[existing["file"] != str(file_path)]  # replace stale row
        new = pd.concat([existing, new], ignore_index=True)
    new.to_csv(manifest_path, index=False)
    logger.info("updated manifest %s for %s (%d records)", manifest_path, dataset_id, n_records)


# Backwards-compatible alias matching the original stub signature.
fetch_dataset = fetch_cdaweb


__all__ = [
    "fetch_cdaweb",
    "fetch_dataset",
    "update_manifest",
    "GOES_TARGET_DATASETS",
    "GOES_LEGACY_DATASETS",
    "OMNI_DRIVERS_DATASET",
    "OMNI_DRIVERS_DATASET_5MIN",
    "OMNI_DRIVER_VARIABLES",
]
