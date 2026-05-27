.PHONY: help install lint format format-check typecheck test check clean

PYTHON := uv run python

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
	@echo "  clean        Remove caches and build artefacts"
	@echo ""
	@echo "Domain targets (data, train, eval, cost, chart, publish, rehearse)"
	@echo "are added as Phases 3–9 land. See DECISIONS.md."

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

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
