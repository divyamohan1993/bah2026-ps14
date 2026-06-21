"""Streamlit live dashboard for the >2 MeV GEO electron forecast (R4 §7, ARCHITECTURE (f.5)).

Panels:
  * live >2 MeV flux + solar-wind (Vsw, Bz) time series (Plotly);
  * multi-horizon forecast (nowcast / 6 h / 12 h) with P10-P90 uncertainty bands;
  * the 1000-pfu alert-status indicator (per-horizon P(flux >= 1000 pfu));
  * a model / skill panel.

Live panels refresh with ``st.fragment(run_every=...)`` (Streamlit >= 1.37) so a timer
reruns just a panel, not the whole script. The dashboard reads the serving API
(``/latest`` + ``/forecast``) when available, otherwise drives a local
:class:`~ps14.serve.inference.Predictor` on a synthetic stream for a fully-offline demo.

``streamlit`` / ``plotly`` / ``requests`` are imported lazily so ``import ps14`` works with
only the core deps; the optional ``[viz]`` extra is needed only to actually run the app.
Run with ``streamlit run src/ps14/dashboard/app.py`` (``make dashboard``).
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ps14.constants import HARSH_PFU, HORIZON_LEAD_MINUTES, HORIZON_NAMES, LOG_HARSH
from ps14.serve.inference import ForecastPayload, OnlineFeatureState, Predictor
from ps14.serve.scheduler import synthetic_source

API_BASE_DEFAULT = "http://localhost:8000"
_HISTORY_LEN = 288  # one day at 5-min cadence


# ======================================================================================
# Data access (API or local demo)
# ======================================================================================
def fetch_latest(api_base: str = API_BASE_DEFAULT, *, timeout: float = 2.0) -> dict[str, Any]:
    """Fetch the latest observed values from the serving API (``GET /latest``)."""
    import requests  # lazy: optional [data] extra

    return requests.get(f"{api_base}/latest", timeout=timeout).json()


def fetch_forecast(api_base: str = API_BASE_DEFAULT, *, timeout: float = 2.0) -> dict[str, Any]:
    """Fetch the current multi-horizon forecast payload (``GET /forecast``)."""
    import requests  # lazy: optional [data] extra

    return requests.get(f"{api_base}/forecast", timeout=timeout).json()


class DemoFeed:
    """Local offline feed: a synthetic stream through a :class:`Predictor` (no network).

    Keeps a rolling history of flux / Vsw / Bz for the time-series panel and recomputes a
    forecast on each :meth:`step`, so the dashboard demo runs end-to-end without the API or
    any trained artifact.
    """

    def __init__(self, predictor: Predictor | None = None, *, seed: int = 1993) -> None:
        self.predictor = predictor if predictor is not None else Predictor()
        self.feature_state = OnlineFeatureState()
        self._source = synthetic_source(seed=seed)
        self.flux: deque[float] = deque(maxlen=_HISTORY_LEN)
        self.vsw: deque[float] = deque(maxlen=_HISTORY_LEN)
        self.bz: deque[float] = deque(maxlen=_HISTORY_LEN)
        self.step_idx: deque[int] = deque(maxlen=_HISTORY_LEN)
        self._i = 0
        self.latest_forecast: ForecastPayload | None = None
        # Seed a little history so the first frame is not empty.
        for _ in range(min(48, _HISTORY_LEN)):
            self.step()

    def step(self) -> ForecastPayload:
        """Advance one synthetic sample and recompute the forecast."""
        sample = self._source()
        self._i += 1
        self.flux.append(10.0 ** sample["log_flux_e2"])
        self.vsw.append(sample["vsw"])
        self.bz.append(sample["bz_gsm"])
        self.step_idx.append(self._i)
        features = self.feature_state.update(sample).copy()
        ctx = {k: float(sample[k]) for k in ("mlt", "kp", "vsw") if k in sample}
        self.latest_forecast = self.predictor.predict(features, context=ctx)
        return self.latest_forecast


# ======================================================================================
# Plotly figures (return figures so they are unit-testable without Streamlit)
# ======================================================================================
def build_flux_figure(
    steps: list[int], flux: list[float], vsw: list[float], bz: list[float]
) -> Any:
    """Build the live flux + solar-wind Plotly figure (flux log-y, Vsw/Bz on a 2nd axis)."""
    from plotly.subplots import make_subplots  # lazy: optional [viz] extra

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_scatter(x=steps, y=flux, name=">2 MeV flux (pfu)", line={"color": "#e45756"})
    fig.add_scatter(
        x=steps, y=vsw, name="Vsw (km/s)", line={"color": "#4c78a8"}, secondary_y=True
    )
    fig.add_scatter(
        x=steps, y=bz, name="Bz (nT)", line={"color": "#54a24b", "dash": "dot"}, secondary_y=True
    )
    # 1000-pfu alert line.
    if steps:
        fig.add_hline(
            y=HARSH_PFU, line={"color": "crimson", "dash": "dash"}, annotation_text="1000 pfu"
        )
    fig.update_yaxes(type="log", title_text="flux (pfu)", secondary_y=False)
    fig.update_yaxes(title_text="Vsw / Bz", secondary_y=True)
    fig.update_layout(
        title="Live >2 MeV flux + solar wind", height=360, margin={"t": 40, "b": 20}
    )
    return fig


def build_forecast_figure(forecast: dict[str, Any]) -> Any:
    """Build the multi-horizon fan chart (P10-P90 band + P50) in linear pfu."""
    import plotly.graph_objects as go  # lazy: optional [viz] extra

    horizons = forecast.get("horizons", {})
    leads, p10, p50, p90 = [], [], [], []
    for name in HORIZON_NAMES:
        h = horizons.get(name)
        if h is None:
            continue
        leads.append(h.get("lead_min", HORIZON_LEAD_MINUTES.get(name, 0)) / 60.0)
        p10.append(10.0 ** h["p10"])
        p50.append(10.0 ** h["p50"])
        p90.append(10.0 ** h["p90"])

    fig = go.Figure()
    if leads:
        fig.add_scatter(
            x=leads + leads[::-1],
            y=p90 + p10[::-1],
            fill="toself",
            fillcolor="rgba(76,120,168,0.25)",
            line={"color": "rgba(0,0,0,0)"},
            name="P10-P90",
        )
        fig.add_scatter(x=leads, y=p50, mode="lines+markers", name="P50", line={"color": "#4c78a8"})
        fig.add_hline(
            y=HARSH_PFU, line={"color": "crimson", "dash": "dash"}, annotation_text="1000 pfu"
        )
    fig.update_yaxes(type="log", title_text="flux (pfu)")
    fig.update_xaxes(title_text="lead time (hours)")
    fig.update_layout(
        title="Multi-horizon forecast (P10-P50-P90)", height=360, margin={"t": 40, "b": 20}
    )
    return fig


def alert_levels(forecast: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-horizon alert summary ``{horizon: {p_exceed, alert, color, label}}``.

    Pure data (no Streamlit) so it is unit-testable; the renderer maps it to badges.
    """
    out: dict[str, dict[str, Any]] = {}
    for name in HORIZON_NAMES:
        h = forecast.get("horizons", {}).get(name, {})
        pe = float(h.get("p_exceed_1000pfu", 0.0))
        alert = bool(h.get("alert", pe >= 0.5))
        if pe >= 0.5:
            color, label = "red", "ALERT"
        elif pe >= 0.2:
            color, label = "orange", "WATCH"
        else:
            color, label = "green", "NOMINAL"
        out[name] = {"p_exceed": pe, "alert": alert, "color": color, "label": label}
    return out


# ======================================================================================
# Streamlit renderers
# ======================================================================================
def render_flux_panel(feed: DemoFeed) -> None:
    """Render the live >2 MeV flux + solar-wind time-series panel (Plotly)."""
    import streamlit as st  # lazy: optional [viz] extra

    fig = build_flux_figure(
        list(feed.step_idx), list(feed.flux), list(feed.vsw), list(feed.bz)
    )
    st.plotly_chart(fig, use_container_width=True)


def render_forecast_panel(forecast: dict[str, Any]) -> None:
    """Render the multi-horizon forecast with P10-P90 uncertainty bands (Plotly)."""
    import streamlit as st  # lazy: optional [viz] extra

    st.plotly_chart(build_forecast_figure(forecast), use_container_width=True)


def render_alert_panel(forecast: dict[str, Any]) -> None:
    """Render the 1000-pfu alert-status indicator per horizon."""
    import streamlit as st  # lazy: optional [viz] extra

    levels = alert_levels(forecast)
    cols = st.columns(len(HORIZON_NAMES))
    for col, name in zip(cols, HORIZON_NAMES):
        lvl = levels[name]
        col.metric(
            label=f"{name} (+{HORIZON_LEAD_MINUTES[name]} min)",
            value=lvl["label"],
            delta=f"P(exceed)={lvl['p_exceed']:.0%}",
        )


def render_model_panel(forecast: dict[str, Any]) -> None:
    """Render the model / skill panel (model name, source, latency, threshold)."""
    import streamlit as st  # lazy: optional [viz] extra

    c1, c2, c3 = st.columns(3)
    c1.metric("Model", forecast.get("model", "—"))
    c2.metric("Source", forecast.get("source", "—"))
    c3.metric("Latency", f"{forecast.get('latency_ms', 0.0):.2f} ms")
    st.caption(
        f"Harsh alert threshold: {forecast.get('threshold_pfu', HARSH_PFU):.0f} pfu "
        f"(log10 = {LOG_HARSH:.1f}). Forecast issued {forecast.get('issued_utc', '—')}."
    )


def main(api_base: str = API_BASE_DEFAULT, *, refresh_s: float = 5.0) -> None:
    """Streamlit entry point.

    Tries the serving API first; on any failure it falls back to a local synthetic
    :class:`DemoFeed` so the dashboard always runs. Live panels refresh on a timer via
    ``st.fragment(run_every=...)`` when available (Streamlit >= 1.37).
    """
    try:
        import streamlit as st  # lazy: optional [viz] extra
    except ImportError as exc:
        raise RuntimeError(
            "Streamlit is not installed; install the '[viz]' extra to run the dashboard."
        ) from exc

    st.set_page_config(page_title="PS-14 >2 MeV GEO Electron Forecast", layout="wide")
    st.title("PS-14 — >2 MeV GEO Electron Radiation Forecast")
    st.caption(
        "Multi-horizon (nowcast / 6 h / 12 h) forecast with P10-P90 uncertainty and a "
        "1000-pfu alert. O(1) online features + cached inference (R4)."
    )

    if "feed" not in st.session_state:
        st.session_state.feed = DemoFeed()
    feed: DemoFeed = st.session_state.feed

    def _get_forecast() -> dict[str, Any]:
        try:
            return fetch_forecast(api_base)
        except Exception:  # noqa: BLE001 - offline demo fallback
            payload = feed.step()
            return payload.model_dump()

    has_fragment = hasattr(st, "fragment")

    def _live_body() -> None:
        forecast = _get_forecast()
        render_alert_panel(forecast)
        left, right = st.columns(2)
        with left:
            render_flux_panel(feed)
        with right:
            render_forecast_panel(forecast)
        render_model_panel(forecast)

    if has_fragment:

        @st.fragment(run_every=refresh_s)
        def _live() -> None:
            _live_body()

        _live()
    else:  # pragma: no cover - very old Streamlit
        _live_body()
        st.info("Upgrade to Streamlit >= 1.37 for auto-refreshing live panels.")


def _demo_dataframe(feed: DemoFeed) -> Any:
    """Return a small numpy view of the demo history (used by tests / notebooks)."""
    return np.column_stack(
        [list(feed.step_idx), list(feed.flux), list(feed.vsw), list(feed.bz)]
    )


if __name__ == "__main__":  # pragma: no cover - Streamlit invokes the module directly
    main()


__all__ = [
    "fetch_latest",
    "fetch_forecast",
    "DemoFeed",
    "build_flux_figure",
    "build_forecast_figure",
    "alert_levels",
    "render_flux_panel",
    "render_forecast_panel",
    "render_alert_panel",
    "render_model_panel",
    "main",
]
