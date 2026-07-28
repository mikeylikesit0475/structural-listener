.PHONY: install lint typecheck test check crack friction impact noise replay heatmap demo-gif serve clean

install: ## Sync dev environment
	uv sync --all-extras --dev

lint: ## Ruff lint + format check
	uv run ruff check
	uv run ruff format --check

typecheck: ## mypy --strict
	uv run mypy

test: ## Run pytest
	uv run pytest

check: lint typecheck test ## All gates

# --- Sprint 0.6 CLI: inject a simulated event, write 4-channel .npz with ground truth ---
# Usage: make crack X=0.62 Y=0.31
# Output: data/out/<type>_<x>_<y>.npz with waveforms, sample rate, source coords, sensor layout.
crack friction impact noise:
	uv run python -m structural_listener.cli.inject $@ --x $(X) --y $(Y)

# --- Sprint 1.4: regenerate the localisation error heat map (docs/figures/error-heatmap.png) ---
heatmap:
	uv run python -m structural_listener.eval.plot_error_heatmap

replay:
# --- Sprint 2.7: replay real recorded audio through the DSP path (qualitative) ---
# Place wav files under data/replay/ (e.g. a MIMII slice) then: make replay
	uv run python -m structural_listener.model.replay

# --- Sprint 3.1: FastAPI + WebSocket service ---
serve:
	uv run uvicorn structural_listener.service.app:app --reload --port 8000

# --- Sprint 3.7: demo GIF (animated simulation of the live UI) ---
demo-gif:
	uv run python -m structural_listener.eval.plot_demo_gif

clean: ## Remove caches and generated artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +