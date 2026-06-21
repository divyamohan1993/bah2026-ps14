"""HAPI fetch client (hapiclient) for streaming time-series subsets.

OPTIONAL path (``[data]`` extra). HAPI is the COSPAR-recommended cross-archive
time-series API; good for bulk historical backfill and any source not exposed as an
SWPC rolling JSON (R4 §1.2, R2 "HAPI server base URLs").
"""

from __future__ import annotations

import pandas as pd

# Common HAPI servers (R2). CDAWeb holds GOES SEISS, Wind, OMNI, RBSP, etc.
HAPI_SERVERS: dict[str, str] = {
    "cdaweb": "https://cdaweb.gsfc.nasa.gov/hapi",
    "sscweb": "https://sscweb.gsfc.nasa.gov/WS/hapi",
    "lisird": "https://lasp.colorado.edu/lisird/hapi",
    "vires": "https://vires.services/hapi",
}


def fetch_hapi(
    dataset_id: str,
    parameters: list[str],
    start: str,
    end: str,
    *,
    server: str = "cdaweb",
) -> pd.DataFrame:
    """Fetch a HAPI dataset over a time range into a UTC-indexed DataFrame.

    Parameters
    ----------
    dataset_id:
        HAPI dataset ID on the chosen server.
    parameters:
        Parameter names to retrieve (server-side projection).
    start, end:
        ISO-8601 UTC bounds.
    server:
        Key into :data:`HAPI_SERVERS` or a full base URL.

    Returns
    -------
    pd.DataFrame
        UTC-indexed frame with the requested parameters.
    """
    raise NotImplementedError(
        "TODO: call hapiclient.hapi(server_url, dataset_id, ','.join(parameters), start, end), "
        "convert the structured array to a UTC-indexed DataFrame (R4 §1.2)."
    )


__all__ = ["fetch_hapi", "HAPI_SERVERS"]
