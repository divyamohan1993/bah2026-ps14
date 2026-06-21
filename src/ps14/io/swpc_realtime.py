"""NOAA SWPC real-time JSON client for nowcasting (R4 §1.1, R2 "stream real-time").

Fetches the rolling JSON products that auto-update and always include the active
spacecraft: GOES >2 MeV integral electrons, L1 (ACE/DSCOVR) solar-wind plasma + IMF,
planetary Kp, and alerts. Used by the serving path; OPTIONAL ``[data]`` extra. ``requests``
is imported LAZILY inside the functions so ``import ps14`` works without it.

Two JSON shapes must be handled (R4 §1.1):
- ``json/goes/...`` files are **lists of objects** (one dict per record).
- ``products/...`` files are **list-of-lists** with the header row first.

Returned frames are mapped toward the canonical merged-schema column names
(``flux_e2``, ``vsw``, ``density``, ``bz_gsm``, ``bt``, ``kp``) so the serving path can
align them with the offline contract (CONTRACTS.md §2).
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Canonical endpoints (mirror config/default.yaml -> data.sources; exact URLs from R2).
SWPC_ENDPOINTS: dict[str, str] = {
    "electrons": "https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-1-day.json",
    "differential_electrons": (
        "https://services.swpc.noaa.gov/json/goes/primary/differential-electrons-1-day.json"
    ),
    "plasma": "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json",
    "mag": "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json",
    "kp": "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
    "alerts": "https://services.swpc.noaa.gov/products/alerts.json",
    "instrument_sources": (
        "https://services.swpc.noaa.gov/json/goes/primary/instrument-sources.json"
    ),
}

# Identifier of the >2 MeV integral electron channel in the GOES integral-electrons feed.
_GE_2MEV_LABELS: tuple[str, ...] = (">=2 MeV", ">2 MeV", "2MeV", ">=2MeV")

# Default request timeout (seconds) -- fail fast when offline.
_TIMEOUT_S: float = 20.0


def _get_json(url: str):
    """GET a URL and return parsed JSON, raising an actionable error when offline."""
    try:
        import requests  # noqa: PLC0415  (lazy optional dependency)
    except ImportError as exc:
        raise ImportError(
            "requests is required for the SWPC real-time client. Install the data extra: "
            "`pip install 'ps14[data]'` (or `pip install requests`). The pipeline runs "
            "fully offline on synthetic data without it (ps14.io.synthetic.replay_stream)."
        ) from exc

    try:
        response = requests.get(url, timeout=_TIMEOUT_S)
        response.raise_for_status()
        return response.json()
    except Exception as exc:  # network / HTTP / decode errors
        raise RuntimeError(
            f"SWPC fetch of {url} failed: {exc}. Check network connectivity; for offline "
            "operation use ps14.io.synthetic.replay_stream() instead."
        ) from exc


def parse_products_json(payload: list[list]) -> pd.DataFrame:
    """Parse an SWPC 'products' list-of-lists payload (header row first).

    Parameters
    ----------
    payload:
        e.g. ``[["time_tag","density","speed","temperature"], ["2026-...", ...], ...]``.

    Returns
    -------
    pd.DataFrame
        Columns from the header row; ``time_tag`` parsed to a UTC ``DatetimeIndex`` named
        ``"time"``; numeric columns coerced to float.
    """
    if not payload or len(payload) < 2:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="time"))
    header = [str(h) for h in payload[0]]
    frame = pd.DataFrame(payload[1:], columns=header)
    return _finalize_time_frame(frame, time_col="time_tag")


def parse_records_json(payload: list[dict]) -> pd.DataFrame:
    """Parse an SWPC 'json/goes' list-of-objects payload into a DataFrame.

    Returns a UTC-indexed frame; numeric columns are coerced to float. (Channel
    filtering for the electron feed is applied by :func:`fetch_latest_electrons`.)
    """
    if not payload:
        return pd.DataFrame(index=pd.DatetimeIndex([], name="time"))
    frame = pd.DataFrame(payload)
    return _finalize_time_frame(frame, time_col="time_tag")


def _finalize_time_frame(frame: pd.DataFrame, *, time_col: str) -> pd.DataFrame:
    """Set the time column as a UTC ``DatetimeIndex`` and coerce other columns to float."""
    if time_col not in frame.columns:
        # Some feeds use 'time' instead of 'time_tag'.
        time_col = "time" if "time" in frame.columns else frame.columns[0]
    index = pd.DatetimeIndex(
        pd.to_datetime(frame[time_col], utc=True, format="ISO8601").dt.tz_localize(None),
        name="time",
    ).as_unit("ns")
    out = frame.drop(columns=[time_col])
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = pd.to_numeric(out[col], errors="ignore")
    out.index = index
    out.index.name = "time"
    return out.sort_index()


def fetch_latest_electrons(url: str | None = None) -> pd.DataFrame:
    """Fetch the latest GOES >2 MeV integral electron flux (target nowcast feed).

    Returns
    -------
    pd.DataFrame
        UTC-indexed frame with a ``flux_e2`` column (the >2 MeV integral electron flux in
        pfu) and a ``satellite`` column, filtered to the >2 MeV channel.
    """
    payload = _get_json(url or SWPC_ENDPOINTS["electrons"])
    frame = parse_records_json(payload)
    if frame.empty:
        return frame

    if "energy" in frame.columns:
        mask = (
            frame["energy"]
            .astype(str)
            .str.replace(" ", "")
            .isin([label.replace(" ", "") for label in _GE_2MEV_LABELS])
        )
        frame = frame[mask]

    out = pd.DataFrame(index=frame.index)
    if "flux" in frame.columns:
        out["flux_e2"] = pd.to_numeric(frame["flux"], errors="coerce").astype("float64")
    if "satellite" in frame.columns:
        out["satellite"] = frame["satellite"].to_numpy()
    out.index.name = "time"
    return out.sort_index()


def fetch_latest_solar_wind(
    plasma_url: str | None = None, mag_url: str | None = None
) -> pd.DataFrame:
    """Fetch + merge the latest L1 solar-wind plasma and IMF onto one time index.

    Returns
    -------
    pd.DataFrame
        UTC-indexed frame mapped toward the merged schema: ``vsw`` (km/s), ``density``
        (cm^-3), ``temperature`` (K), ``bt`` (|B|, nT), ``bz_gsm`` (nT), and the GSM
        components when present.
    """
    plasma = parse_products_json(_get_json(plasma_url or SWPC_ENDPOINTS["plasma"]))
    mag = parse_products_json(_get_json(mag_url or SWPC_ENDPOINTS["mag"]))

    plasma = plasma.rename(
        columns={"speed": "vsw", "density": "density", "temperature": "temperature"}
    )
    mag = mag.rename(
        columns={
            "bt": "bt",
            "bz_gsm": "bz_gsm",
            "bx_gsm": "bx_gsm",
            "by_gsm": "by_gsm",
            "lon_gsm": "lon_gsm",
            "lat_gsm": "lat_gsm",
        }
    )

    keep_plasma = [c for c in ("vsw", "density", "temperature") if c in plasma.columns]
    keep_mag = [c for c in ("bt", "bz_gsm", "bx_gsm", "by_gsm") if c in mag.columns]
    merged = plasma[keep_plasma].join(mag[keep_mag], how="outer")
    for col in merged.columns:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("float64")
    merged.index.name = "time"
    return merged.sort_index()


def fetch_latest_kp(url: str | None = None) -> pd.DataFrame:
    """Fetch the latest estimated planetary Kp (rolling product).

    Returns
    -------
    pd.DataFrame
        UTC-indexed frame with a ``kp`` column (0-9, may be fractional).
    """
    payload = _get_json(url or SWPC_ENDPOINTS["kp"])
    frame = parse_products_json(payload)
    if frame.empty:
        return frame
    out = pd.DataFrame(index=frame.index)
    kp_col = next((c for c in ("kp_index", "kp", "estimated_kp", "Kp") if c in frame.columns), None)
    if kp_col is not None:
        out["kp"] = pd.to_numeric(frame[kp_col], errors="coerce").astype("float64")
    out.index.name = "time"
    return out.sort_index()


def fetch_alerts(url: str | None = None) -> list[dict]:
    """Fetch current SWPC alerts/watches/warnings (for the dashboard alert panel)."""
    payload = _get_json(url or SWPC_ENDPOINTS["alerts"])
    return list(payload) if isinstance(payload, list) else []


def fetch_merged_realtime() -> pd.DataFrame:
    """Fetch and merge electrons + solar wind + Kp into one schema-shaped frame.

    Convenience for the serving path: outer-joins :func:`fetch_latest_electrons`,
    :func:`fetch_latest_solar_wind`, and :func:`fetch_latest_kp` on the time index and
    returns the merged-schema columns that are available in real time.

    Returns
    -------
    pd.DataFrame
        UTC-indexed frame with whatever of ``flux_e2, vsw, density, bt, bz_gsm, kp`` the
        live feeds provide.
    """
    electrons = fetch_latest_electrons()
    solar_wind = fetch_latest_solar_wind()
    kp = fetch_latest_kp()

    flux = electrons[["flux_e2"]] if "flux_e2" in electrons.columns else electrons
    merged = flux.join(solar_wind, how="outer").join(kp, how="outer")
    # Kp is 3-hourly; forward-fill it onto the finer solar-wind cadence.
    if "kp" in merged.columns:
        merged["kp"] = merged["kp"].ffill()
    merged.index.name = "time"
    return merged.sort_index()


__all__ = [
    "SWPC_ENDPOINTS",
    "parse_products_json",
    "parse_records_json",
    "fetch_latest_electrons",
    "fetch_latest_solar_wind",
    "fetch_latest_kp",
    "fetch_alerts",
    "fetch_merged_realtime",
]
