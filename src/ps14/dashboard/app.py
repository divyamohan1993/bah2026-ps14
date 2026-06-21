"""Streamlit live dashboard skeleton (R4 §7, ARCHITECTURE.md (f.5)).

Panels:
  * live >2 MeV flux + solar-wind (Vsw, Bz, Kp) time series;
  * multi-horizon forecast (nowcast / 6 h / 12 h) with P10-P90 uncertainty bands;
  * alert status indicator (current/forecast P(flux >= 1000 pfu)).

Refresh live panels with ``st.fragment(run_every=...)`` (Streamlit >= 1.37) so a timer
reruns just a panel, not the whole script. Reads the FastAPI ``/forecast`` + ``/latest``
endpoints (or the hot cache directly). Run with ``streamlit run src/ps14/dashboard/app.py``
(``make dashboard``). Importable without Streamlit installed.
"""

from __future__ import annotations

from typing import Any

try:  # optional dependency (the [viz] extra)
    import streamlit as st  # type: ignore

    _HAS_STREAMLIT = True
except ImportError:  # pragma: no cover - exercised only without the extra
    st = None  # type: ignore
    _HAS_STREAMLIT = False


def fetch_latest(api_base: str = "http://localhost:8000") -> dict[str, Any]:
    """Fetch the latest observed values from the serving API (or hot cache)."""
    raise NotImplementedError("TODO: requests.get(f'{api_base}/latest').json().")


def fetch_forecast(api_base: str = "http://localhost:8000") -> dict[str, Any]:
    """Fetch the current multi-horizon forecast payload from the serving API."""
    raise NotImplementedError("TODO: requests.get(f'{api_base}/forecast').json().")


def render_flux_panel(history) -> None:
    """Render the live >2 MeV flux + solar-wind time-series panel (Plotly)."""
    raise NotImplementedError("TODO: Plotly time series of flux/Vsw/Bz/Kp; mark the 1000 pfu line.")


def render_forecast_panel(forecast: dict) -> None:
    """Render the multi-horizon forecast with P10-P90 uncertainty bands."""
    raise NotImplementedError("TODO: Plotly fan chart of P10/P50/P90 across nowcast/6h/12h.")


def render_alert_panel(forecast: dict) -> None:
    """Render the alert-status indicator from P(flux >= 1000 pfu) per horizon."""
    raise NotImplementedError("TODO: colored status badge per horizon from p_exceed_1000pfu.")


def main() -> None:
    """Streamlit entry point (the module body calls this when run by Streamlit)."""
    if not _HAS_STREAMLIT:
        raise RuntimeError(
            "Streamlit is not installed; install the '[viz]' extra to run the dashboard."
        )
    raise NotImplementedError(
        "TODO: st.set_page_config(...); st.title('PS-14 >2 MeV GEO Electron Forecast'); "
        "use st.fragment(run_every=...) panels calling render_* with "
        "fetch_latest/fetch_forecast (R4 §7)."
    )


if __name__ == "__main__":  # pragma: no cover - Streamlit invokes the module directly
    main()


__all__ = [
    "fetch_latest",
    "fetch_forecast",
    "render_flux_panel",
    "render_forecast_panel",
    "render_alert_panel",
    "main",
]
