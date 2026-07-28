.PHONY: install lint typecheck test check crack friction impact noise replay clean

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

replay:
	@echo "Target 'replay' not implemented yet (Sprint 2.7)."
	@exit 1

clean: ## Remove caches and generated artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +