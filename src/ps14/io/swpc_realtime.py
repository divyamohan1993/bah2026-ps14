"""NOAA SWPC real-time JSON client for nowcasting (R4 §1.1).

Fetches the rolling JSON products that auto-update and always include the active
spacecraft: GOES >2 MeV integral electrons, L1 (ACE/DSCOVR) solar-wind plasma + IMF,
planetary Kp, and alerts. Used by the serving path; OPTIONAL ``[data]`` extra (requests).

Two JSON shapes must be handled (R4 §1.1):
- ``json/goes/...`` files are **lists of objects** (one dict per record).
- ``products/...`` files are **list-of-lists** with the header row first.
"""

from __future__ import annotations

import pandas as pd

# Canonical endpoints (mirror config/default.yaml -> data.sources).
SWPC_ENDPOINTS: dict[str, str] = {
    "electrons": "https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-1-day.json",
    "plasma": "https://services.swpc.noaa.gov/products/solar-wind/plasma-5-minute.json",
    "mag": "https://services.swpc.noaa.gov/products/solar-wind/mag-5-minute.json",
    "kp": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
    "alerts": "https://services.swpc.noaa.gov/products/alerts.json",
    "instrument_sources": "https://services.swpc.noaa.gov/json/goes/primary/instrument-sources.json",
}


def parse_products_json(payload: list[list]) -> pd.DataFrame:
    """Parse an SWPC 'products' list-of-lists payload (header row first).

    Parameters
    ----------
    payload:
        e.g. ``[["time_tag","density","speed","temperature"], ["2026-...", ...], ...]``.

    Returns
    -------
    pd.DataFrame
        Columns from the header row; ``time_tag`` parsed to a UTC ``DatetimeIndex``.
    """
    raise NotImplementedError(
        "TODO: header = payload[0]; rows = payload[1:]; build DataFrame, coerce numerics, "
        "set time_tag -> DatetimeIndex (UTC, name='time')."
    )


def parse_records_json(payload: list[dict]) -> pd.DataFrame:
    """Parse an SWPC 'json/goes' list-of-objects payload into a DataFrame.

    Filters to the >2 MeV integral electron channel where applicable and returns a
    UTC-indexed frame.
    """
    raise NotImplementedError(
        "TODO: pd.DataFrame(payload); coerce 'time_tag'; filter energy=='>=2 MeV' for electrons."
    )


def fetch_latest_electrons(url: str | None = None) -> pd.DataFrame:
    """Fetch the latest GOES >2 MeV integral electron flux (target nowcast feed)."""
    raise NotImplementedError(
        "TODO: requests.get(url or SWPC_ENDPOINTS['electrons']) -> parse_records_json."
    )


def fetch_latest_solar_wind(
    plasma_url: str | None = None, mag_url: str | None = None
) -> pd.DataFrame:
    """Fetch + merge the latest L1 solar-wind plasma and IMF onto one time index."""
    raise NotImplementedError(
        "TODO: fetch plasma + mag products (parse_products_json), merge on time_tag."
    )


def fetch_latest_kp(url: str | None = None) -> pd.DataFrame:
    """Fetch the latest estimated planetary Kp (3-hourly)."""
    raise NotImplementedError("TODO: requests.get -> parse_products_json.")


def fetch_alerts(url: str | None = None) -> list[dict]:
    """Fetch current SWPC alerts/watches/warnings (for the dashboard alert panel)."""
    raise NotImplementedError("TODO: requests.get(SWPC_ENDPOINTS['alerts']).json().")


__all__ = [
    "SWPC_ENDPOINTS",
    "parse_products_json",
    "parse_records_json",
    "fetch_latest_electrons",
    "fetch_latest_solar_wind",
    "fetch_latest_kp",
    "fetch_alerts",
]
