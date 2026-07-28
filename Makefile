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

# --- Sprint 0.6 CLI (implementation lands in 0.6; stubs keep the Makefile honest) ---
# Each target writes a 4-channel .npz with ground-truth source coordinates to data/out/.
crack friction impact noise:
	@echo "Target '$@' not implemented yet (Sprint 0.6)."
	@exit 1

replay:
	@echo "Target 'replay' not implemented yet (Sprint 2.7)."
	@exit 1

clean: ## Remove caches and generated artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +