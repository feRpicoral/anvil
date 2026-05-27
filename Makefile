.PHONY: help install lint format format-check typecheck test check clean data-smoke

PYTHON := uv run python
DATA_SMOKE_DIR := data/smoke

help:
	@echo "Anvil — common tasks"
	@echo ""
	@echo "  install      Install dependencies via uv"
	@echo "  lint         Ruff lint check"
	@echo "  format       Ruff format (writes changes)"
	@echo "  format-check Ruff format check"
	@echo "  typecheck    mypy"
	@echo "  test         pytest"
	@echo "  check        lint + format-check + typecheck + test (mirrors CI)"
	@echo ""
	@echo "  data-smoke   50-sample fixture-replay dataset; zero API spend"
	@echo ""
	@echo "  clean        Remove caches and build artefacts"

install:
	uv sync --all-extras --all-groups

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint format-check typecheck test

data-smoke:
	$(PYTHON) -m scripts.synth --config configs/data-smoke.toml
	$(PYTHON) -m scripts.curate --input $(DATA_SMOKE_DIR)/raw_synthesis.jsonl --output $(DATA_SMOKE_DIR)/curated.jsonl --no-dedup
	$(PYTHON) -m scripts.split --input $(DATA_SMOKE_DIR)/curated.jsonl --output-dir $(DATA_SMOKE_DIR) --allow-overlap

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
