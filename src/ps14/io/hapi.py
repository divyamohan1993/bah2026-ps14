"""HAPI fetch client (hapiclient) for streaming time-series subsets.

OPTIONAL path (``[data]`` extra). HAPI is the COSPAR-recommended cross-archive
time-series API; good for bulk historical backfill and any source not exposed as an
SWPC rolling JSON (R4 §1.2, R2 "HAPI server base URLs"). ``hapiclient`` is imported
LAZILY inside the function so ``import ps14`` works without it.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Common HAPI servers (R2 "HAPI server base URLs"). CDAWeb holds GOES SEISS, Wind, OMNI,
# RBSP, THEMIS, MMS, Arase, etc.
HAPI_SERVERS: dict[str, str] = {
    "cdaweb": "https://cdaweb.gsfc.nasa.gov/hapi",
    "sscweb": "https://sscweb.gsfc.nasa.gov/WS/hapi",
    "lisird": "https://lasp.colorado.edu/lisird/hapi",
    "vires": "https://vires.services/hapi",
    "amda": "http://amda.irap.omp.eu/service/hapi",
    "intermagnet": "https://imag-data.bgs.ac.uk/GIN_V1/hapi",
    "ccmc": "https://iswa.gsfc.nasa.gov/IswaSystemWebApp/hapi",
}


def _resolve_server(server: str) -> str:
    """Map a server key (e.g. ``"cdaweb"``) to its base URL, or pass a full URL through."""
    if server in HAPI_SERVERS:
        return HAPI_SERVERS[server]
    if server.startswith("http://") or server.startswith("https://"):
        return server
    raise ValueError(
        f"unknown HAPI server {server!r}; pass a key from {sorted(HAPI_SERVERS)} or a URL."
    )


def fetch_hapi(
    server: str,
    dataset: str,
    parameters: list[str] | str,
    start: str,
    stop: str,
) -> pd.DataFrame:
    """Fetch a HAPI dataset over a time range into a UTC-indexed DataFrame.

    Parameters
    ----------
    server:
        Key into :data:`HAPI_SERVERS` (e.g. ``"cdaweb"``) or a full base URL.
    dataset:
        HAPI dataset ID on the chosen server (e.g. ``"OMNI_HRO_1MIN"``).
    parameters:
        Parameter name(s) to retrieve (server-side projection). A list or a single
        comma-separated string; ``Time`` is always included by HAPI.
    start, stop:
        ISO-8601 UTC bounds (e.g. ``"2017-02-07T00:00:00Z"``).

    Returns
    -------
    pd.DataFrame
        UTC-indexed frame (``time``) with the requested parameters as float64 columns.

    Raises
    ------
    ImportError
        If ``hapiclient`` is not installed (offline mode).
    RuntimeError
        If the request fails (e.g. no network) -- the message is actionable.
    """
    try:
        from hapiclient import hapi  # noqa: PLC0415  (lazy optional dependency)
    except ImportError as exc:
        raise ImportError(
            "hapiclient is required for HAPI fetching. Install the data extra: "
            "`pip install 'ps14[data]'` (or `pip install hapiclient`). The pipeline runs "
            "fully offline on synthetic data without it (ps14.io.synthetic)."
        ) from exc

    server_url = _resolve_server(server)
    params_str = parameters if isinstance(parameters, str) else ",".join(parameters)

    try:
        data, meta = hapi(server_url, dataset, params_str, start, stop)
    except Exception as exc:  # network / server / dataset errors
        raise RuntimeError(
            f"HAPI fetch of {dataset!r} ({params_str}) from {server_url} failed: {exc}. "
            "Check network connectivity and that the server/dataset/parameters are valid "
            "(browse https://hapi-server.org/servers/)."
        ) from exc

    return _hapi_to_frame(data, meta)


def _hapi_to_frame(data: np.ndarray, meta: dict) -> pd.DataFrame:
    """Convert a hapiclient structured array + metadata into a UTC-indexed DataFrame."""
    if data is None or len(data) == 0:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="time"))

    names = list(data.dtype.names) if data.dtype.names else []
    time_name = names[0] if names else "Time"
    times = pd.to_datetime(
        np.asarray(data[time_name]).astype("U"), utc=True, format="ISO8601"
    ).tz_localize(None)
    index = pd.DatetimeIndex(times, name="time").as_unit("ns")

    columns: dict[str, np.ndarray] = {}
    fill_by_name = {p["name"]: p.get("fill") for p in meta.get("parameters", [])}
    for name in names[1:]:
        values = np.asarray(data[name], dtype="float64")
        if values.ndim > 1:  # multi-channel parameter: keep the first channel
            values = values[:, 0]
        fill = fill_by_name.get(name)
        if fill is not None:
            try:
                values = np.where(np.isclose(values, float(fill)), np.nan, values)
            except (TypeError, ValueError):
                pass
        columns[name] = values

    frame = pd.DataFrame(columns, index=index)
    frame.index.name = "time"
    return frame.sort_index()


__all__ = ["fetch_hapi", "HAPI_SERVERS"]
