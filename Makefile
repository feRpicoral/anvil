.PHONY: help install lint format format-check typecheck test check clean data-smoke data-full train-smoke train-full eval-smoke cost

PYTHON := uv run python
DATA_SMOKE_DIR := data/smoke
DATA_FULL_DIR := data/full

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
	@echo "  data-full    4000-sample paid dataset (requires CONFIRM_PAID=1 and OPENAI_API_KEY)"
	@echo "  train-smoke  generate smoke data and run TRL smoke training"
	@echo "  train-full   paid Unsloth + Llama 3.1 8B QLoRA run (requires CONFIRM_PAID=1, HF_TOKEN, GPU)"
	@echo "  eval-smoke   3-variant fixture eval; zero API spend; writes results/eval/smoke/"
	@echo "  cost         build the cost-comparison JSON (training + inference + breakeven)"
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

data-full:
ifneq ($(CONFIRM_PAID),1)
	@echo "data-full is a paid run. Re-invoke with CONFIRM_PAID=1 to proceed."
	@exit 1
endif
	$(PYTHON) -m scripts.synth --config configs/data-full.toml
	$(PYTHON) -m scripts.curate --input $(DATA_FULL_DIR)/raw_synthesis.jsonl --output $(DATA_FULL_DIR)/curated.jsonl
	$(PYTHON) -m scripts.split --input $(DATA_FULL_DIR)/curated.jsonl --output-dir $(DATA_FULL_DIR)

train-smoke: data-smoke
	$(PYTHON) -m scripts.train --config configs/train-smoke.toml

train-full:
ifneq ($(CONFIRM_PAID),1)
	@echo "train-full is a paid run. Re-invoke with CONFIRM_PAID=1 to proceed."
	@exit 1
endif
	$(PYTHON) -m scripts.train --config configs/train-full.toml

eval-smoke:
	$(PYTHON) -m scripts.eval --config configs/eval-smoke.toml

cost: eval-smoke
	$(PYTHON) -m scripts.cost --config configs/cost-smoke.toml

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
