# BAH-2026 PS-14 — Forecasting >2 MeV electron flux at GEO.
# Thin Make targets mirroring the `ps14` CLI. All commands run fully OFFLINE on
# synthetic data by default; real downloaders are opt-in (`make fetch-data`).

PY ?= python3
CONFIG ?= config/default.yaml
# Demo knobs (tractable CPU run); override e.g. `make demo DEMO_ARGS="--years 1 --no-tft"`.
DEMO_ARGS ?= --years 1.5 --lookback 288 --stride 4 --max-windows 6000 --tft-epochs 4

.PHONY: help setup demo synth-data fetch-data preprocess features train evaluate \
        baseline serve dashboard test lint format clean

help:
	@echo "PS-14 targets:"
	@echo "  setup       Install the package with all extras (editable)."
	@echo "  demo        Run the full end-to-end pipeline on synthetic data + save reports/."
	@echo "  synth-data  Generate physically-plausible synthetic CDFs + parquet (offline)."
	@echo "  fetch-data  Download real GOES/OMNI/Wind data (needs [data] extra + network)."
	@echo "  preprocess  Clean -> resample -> align -> transform into the merged 5-min frame."
	@echo "  features    Build the feature matrix and supervised-window tensors."
	@echo "  baseline    Fit + score the baseline tier (persistence/climatology/lightgbm/refm)."
	@echo "  train       Train the primary model (TFT dual-head) per config."
	@echo "  evaluate    Walk-forward evaluation, per-horizon metrics + report."
	@echo "  serve       Launch the FastAPI + APScheduler real-time service."
	@echo "  dashboard   Launch the Streamlit live dashboard."
	@echo "  test        Run pytest."
	@echo "  lint        ruff check + mypy."
	@echo "  format      ruff format."
	@echo "  clean       Remove derived data/interim/processed + caches."

setup:
	$(PY) -m pip install -e ".[dl,data,serve,viz,dev]"

demo:
	$(PY) scripts/run_demo.py $(DEMO_ARGS)

synth-data:
	$(PY) -m ps14.cli synth-data --config $(CONFIG)

fetch-data:
	$(PY) -m ps14.cli fetch-data --config $(CONFIG)

preprocess:
	$(PY) -m ps14.cli preprocess --config $(CONFIG)

features:
	$(PY) -m ps14.cli features --config $(CONFIG)

baseline:
	$(PY) -m ps14.cli baseline --config $(CONFIG)

train:
	$(PY) -m ps14.cli train --config $(CONFIG)

evaluate:
	$(PY) -m ps14.cli evaluate --config $(CONFIG)

serve:
	$(PY) -m ps14.cli serve --config $(CONFIG)

dashboard:
	streamlit run src/ps14/dashboard/app.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
	$(PY) -m mypy src

format:
	$(PY) -m ruff format src tests

clean:
	rm -rf data/interim/* data/processed/* reports/*
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache
