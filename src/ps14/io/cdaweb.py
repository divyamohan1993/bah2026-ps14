"""CDAWeb fetch client (cdasws / pyspedas) with a local cache.

OPTIONAL path — the system runs fully offline on synthetic data without this module.
Requires the ``[data]`` extra (cdasws, pyspedas). See R2 "Data Access Strategy" and
R5 §2.3 for dataset IDs and the recommended access pattern.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def fetch_dataset(
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
        CDAWeb dataset ID, e.g. ``"OMNI_HRO_1MIN"`` or ``"DN_SEIS-L2-MPSH_G18"``.
    variables:
        Variable names to retrieve (server-side projection keeps transfers small).
    start, end:
        ISO-8601 UTC bounds (e.g. ``"2017-02-07T00:00:00Z"``).
    cache_dir:
        Where to cache downloaded CDFs (idempotent: skip if present + checksum OK).
    method:
        ``"cdasws"`` (xarray with ISTP metadata) or ``"pyspedas"`` (bulk loader).

    Returns
    -------
    pd.DataFrame
        UTC-indexed frame with the requested variables.

    Notes
    -----
    Implementation should: (1) check the cache/manifest; (2) on miss, call
    ``cdasws.CdasWs().get_data(...)`` or ``pyspedas`` and persist the CDF; (3) hand the
    local CDF to :func:`ps14.io.cdf_reader.read_cdf_to_frame` so the same ISTP masking
    path is used everywhere.
    """
    raise NotImplementedError(
        "TODO: implement cdasws/pyspedas fetch + local cache + manifest update (R2, R5 §2.3)."
    )


def update_manifest(cache_dir: str | Path, dataset_id: str, file_path: str | Path) -> None:
    """Append/refresh a coverage-manifest row for a cached CDF (CONTRACTS.md §8).

    Row schema: ``dataset_id, file, start, end, n_records, sha256, fill_fraction,
    download_utc`` (R5 §7).
    """
    raise NotImplementedError("TODO: compute coverage + sha256 and write data/manifest.csv row.")


__all__ = ["fetch_dataset", "update_manifest"]
