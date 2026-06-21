"""APScheduler 60 s refresh job for the real-time forecast (R4 §6).

The job: poll SWPC (or the synthetic replay) -> fold new samples into the O(1) online
features -> run cached inference -> write latest + forecast to the hot cache -> push to
WebSocket subscribers. Each refresh is dominated by network I/O, not compute (R4 §6).
Requires the ``[serve]`` extra (apscheduler).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class RefreshState:
    """Holds the online-feature accumulators + the warm cached forecaster for the job.

    Bundles the ring buffers / Welford accumulators / monotonic deques (one per tracked
    channel) and the :class:`~ps14.serve.inference.CachedForecaster` so the scheduled job
    is a thin O(new) update.
    """

    def __init__(self, config: Any | None = None) -> None:
        self.config = config
        self.online: dict[str, Any] = {}  # channel -> online accumulator(s)
        self.forecaster: Any | None = None
        self.latest: dict[str, Any] = {}
        self.forecast: dict[str, Any] = {}

    def initialize(self) -> None:
        """Allocate online accumulators and load the cached forecaster + LUT."""
        raise NotImplementedError(
            "TODO: build RingBuffer/Welford/MonotonicDeque per channel; load CachedForecaster "
            "+ ClimatologyLUT; seed buffers from recent history."
        )


def refresh_once(state: RefreshState, *, source: str = "synthetic") -> dict:
    """Run one refresh cycle and return the new forecast payload.

    Parameters
    ----------
    state:
        The initialized :class:`RefreshState`.
    source:
        ``"swpc"`` (real-time JSON) or ``"synthetic"`` (offline replay).

    Returns
    -------
    dict
        The forecast payload (also written to ``state.forecast``).
    """
    raise NotImplementedError(
        "TODO: fetch new samples (swpc_realtime or synthetic.replay_stream); "
        "update online features (O(new)); build the feature vector; "
        "state.forecaster.forecast(...); update latest/forecast."
    )


def start_scheduler(state: RefreshState, on_update: Callable[[dict], None] | None = None) -> Any:
    """Start an APScheduler ``BackgroundScheduler`` running :func:`refresh_once` every 60 s.

    Parameters
    ----------
    state:
        Initialized refresh state.
    on_update:
        Optional callback invoked with each new forecast payload (e.g. WebSocket fan-out).

    Returns
    -------
    Any
        The started scheduler (caller keeps a reference / shuts it down on exit).
    """
    raise NotImplementedError(
        "TODO: BackgroundScheduler(); "
        "add_job(refresh_once, 'interval', seconds=cfg.serving.refresh_seconds); "
        "start(); call on_update(payload) after each refresh (R4 §6)."
    )


__all__ = ["RefreshState", "refresh_once", "start_scheduler"]
