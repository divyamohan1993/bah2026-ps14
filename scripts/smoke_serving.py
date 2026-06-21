#!/usr/bin/env python3
"""Serving / ONNX / dashboard smoke test (Milestone 4).

Exercises the real-time path end-to-end without launching a long-running server:

1. FastAPI app  : GET /health and POST /forecast via starlette TestClient (httpx backend).
2. ONNX export  : build a tiny torch quantile head ([N,L,F] -> [N, n_q, n_h]), export to
                  ONNX (serve.inference.export_to_onnx), load it back through onnxruntime in
                  the Predictor, and verify a schema-valid multi-horizon forecast.
3. Fallbacks    : confirm the O(1) ForecastCache returns the same payload on a repeat call,
                  and that the ClimatologyLUT fallback produces a forecast with no artifact.
4. Dashboard    : import the Streamlit dashboard module (no server) to confirm it loads.

Exit code 0 on success; prints a concise PASS/FAIL line per check.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ps14.constants import HORIZON_NAMES, QUANTILES
from ps14.datasets import schema

_N_FEATURES = len(schema.FEATURE_COLUMNS)
_N_H = len(HORIZON_NAMES)
_N_Q = len(QUANTILES)


def _feature_vec(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=_N_FEATURES).astype("float32")
    v[schema.FEATURE_COLUMNS.index(schema.TARGET)] = 2.6  # plausible log-flux
    return v


def check_fastapi() -> bool:
    from starlette.testclient import TestClient

    from ps14.serve.api import create_app

    client = TestClient(create_app())
    r = client.get("/health")
    assert r.status_code == 200, r.status_code
    body = r.json()
    assert body["status"] == "ok", body
    r = client.post(
        "/forecast",
        json={"features": _feature_vec(1).tolist(), "mlt": 12.0, "kp": 3.0, "vsw": 450.0},
    )
    assert r.status_code == 200, (r.status_code, r.text)
    fc = r.json()
    assert set(fc["horizons"]) == set(HORIZON_NAMES), fc["horizons"]
    print(f"  [PASS] FastAPI /health + POST /forecast (horizons={list(fc['horizons'])})")
    return True


class _TinyQuantileHead:
    """A torch module emitting [N, n_q, n_h] quantiles from a flattened encoder window."""


def check_onnx_predictor() -> bool:
    import torch

    from ps14.serve.inference import Predictor, export_to_onnx

    class TinyHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(_N_FEATURES, _N_Q * _N_H)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [N, L, F]; use the last encoder step, emit [N, n_q, n_h].
            last = x[:, -1, :]
            out = self.lin(last).reshape(-1, _N_Q, _N_H)
            # Make quantiles monotone + plausible (around log-flux ~2.6).
            base = 2.6 + 0.0 * out[:, :1, :]
            spread = torch.cumsum(torch.nn.functional.softplus(out), dim=1) * 0.2
            return base + spread - spread.mean(dim=1, keepdim=True)

    with tempfile.TemporaryDirectory() as d:
        onnx_path = Path(d) / "tiny.onnx"
        export_to_onnx(TinyHead(), onnx_path, lookback=1, n_features=_N_FEATURES)
        assert onnx_path.exists(), "ONNX file not written"

        pred = Predictor(onnx_path=onnx_path)
        assert pred.backend == "onnx", f"expected ONNX backend, got {pred.backend}"
        payload = pred.predict(_feature_vec(2))
        assert set(payload.horizons) == set(HORIZON_NAMES), payload.horizons
        # Cache: a repeat call returns the identical payload object (O(1) hit).
        payload2 = pred.predict(_feature_vec(2))
        assert payload2 is payload, "ForecastCache did not return the cached payload"
        print(
            f"  [PASS] ONNX export + onnxruntime Predictor "
            f"(backend={pred.backend}, p50_nowcast={payload.horizons['nowcast'].p50:.3f}, "
            f"cache hit OK)"
        )
    return True


def check_climatology_fallback() -> bool:
    from ps14.serve.inference import ClimatologyLUT, Predictor

    # No artifacts -> climatology fallback.
    pred = Predictor()
    assert pred.backend == "climatology", pred.backend
    payload = pred.predict(_feature_vec(3), context={"mlt": 12.0, "kp": 3.0, "vsw": 500.0})
    assert set(payload.horizons) == set(HORIZON_NAMES), payload.horizons
    # Direct LUT forecast is schema-valid too.
    lut = ClimatologyLUT()
    f_noon = lut.forecast(mlt=12.0, doy=80.0, kp=3.0, vsw=500.0)
    assert set(f_noon.horizons) == set(HORIZON_NAMES)
    print(f"  [PASS] ClimatologyLUT O(1) fallback (backend={pred.backend})")
    return True


def check_dashboard_import() -> bool:
    import importlib

    mod = importlib.import_module("ps14.dashboard.app")
    assert mod is not None
    print(f"  [PASS] Streamlit dashboard module imports ({mod.__name__})")
    return True


def main() -> int:
    print("PS-14 serving / ONNX / dashboard smoke test")
    ok = True
    for name, fn in [
        ("fastapi", check_fastapi),
        ("onnx_predictor", check_onnx_predictor),
        ("climatology_fallback", check_climatology_fallback),
        ("dashboard_import", check_dashboard_import),
    ]:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - report and continue
            ok = False
            print(f"  [FAIL] {name}: {type(exc).__name__}: {exc}")
    print("ALL SMOKE CHECKS PASSED" if ok else "SOME SMOKE CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
