"""Tests for the paid-path additions in `scripts.synth`."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from anvil.data.synthesis import GenerationResult
from scripts.synth import (
    BudgetExceededError,
    SynthConfig,
    build_generator,
    load_config,
    run,
    synthesize,
)


class _CountingGenerator:
    """Deterministic stub returning a constant per-call cost."""

    def __init__(self, cost_usd: float = 0.10) -> None:
        self.cost_usd = cost_usd
        self.calls = 0
        self.last_prompts: tuple[str, str] | None = None

    async def generate(
        self,
        contract_type: str,
        system_prompt: str,
        user_prompt: str,
        seed: int,
    ) -> GenerationResult:
        self.calls += 1
        self.last_prompts = (system_prompt, user_prompt)
        return GenerationResult(
            contract_type=contract_type,  # type: ignore[arg-type]
            contract_text=f"# Sample {self.calls}",
            extraction={},
            input_tokens=100,
            output_tokens=200,
            cost_usd=self.cost_usd,
            backend="stub",
        )


def _write_paid_config(tmp_path: Path, **overrides: Any) -> Path:
    body = {
        "backend": '"openai"',
        "model": '"gpt-4o-2024-08-06"',
        "num_samples": "5",
        "max_spend_usd": "10.0",
        "output_dir": f'"{tmp_path / "out"}"',
        "seed": "0",
    }
    body.update({k: f'"{v}"' if isinstance(v, str) else str(v) for k, v in overrides.items()})
    lines = "\n".join(f"{k} = {v}" for k, v in body.items())
    path = tmp_path / "config.toml"
    path.write_text(lines + "\n", encoding="utf-8")
    return path


def test_load_config_parses_model_and_spend_cap(tmp_path: Path) -> None:
    config_path = _write_paid_config(tmp_path)

    config = load_config(config_path)

    assert config.backend == "openai"
    assert config.model == "gpt-4o-2024-08-06"
    assert config.max_spend_usd == 10.0
    assert config.num_samples == 5


def test_load_config_rejects_non_positive_spend_cap(tmp_path: Path) -> None:
    config_path = _write_paid_config(tmp_path, max_spend_usd=0)

    with pytest.raises(ValueError, match="positive number"):
        load_config(config_path)


def test_load_config_allows_no_spend_cap(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'backend = "fixture"\n'
        f'fixtures_dir = "{tmp_path}"\n'
        f'output_dir = "{tmp_path / "out"}"\n'
        "num_samples = 3\n"
        "seed = 0\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.max_spend_usd is None


def test_build_generator_constructs_openai_generator() -> None:
    config = SynthConfig(
        backend="openai",
        fixtures_dir=None,
        num_samples=1,
        output_dir=Path("/tmp"),
        seed=0,
        model="gpt-4o-2024-08-06",
    )

    with patch("scripts.synth.OpenAIGenerator") as constructor:
        constructor.return_value = "openai-instance"
        build_generator(config)

    constructor.assert_called_once_with(model="gpt-4o-2024-08-06")


def test_build_generator_constructs_openai_with_default_model() -> None:
    config = SynthConfig(
        backend="openai",
        fixtures_dir=None,
        num_samples=1,
        output_dir=Path("/tmp"),
        seed=0,
    )

    with patch("scripts.synth.OpenAIGenerator") as constructor:
        constructor.return_value = "openai-instance"
        build_generator(config)

    constructor.assert_called_once_with()


def test_build_generator_constructs_anthropic_generator() -> None:
    config = SynthConfig(
        backend="anthropic",
        fixtures_dir=None,
        num_samples=1,
        output_dir=Path("/tmp"),
        seed=0,
        model="claude-sonnet-4-6",
    )

    with patch("scripts.synth.AnthropicGenerator") as constructor:
        constructor.return_value = "anthropic-instance"
        build_generator(config)

    constructor.assert_called_once_with(model="claude-sonnet-4-6")


def test_synthesize_renders_prompts_via_parameters_factory() -> None:
    generator = _CountingGenerator(cost_usd=0.0)

    asyncio.run(synthesize(generator, num_samples=3, base_seed=0))

    assert generator.calls == 3
    assert generator.last_prompts is not None
    system, user = generator.last_prompts
    assert "contract_text" in system
    assert "Draft" in user
    assert "{" not in user, "user prompt should be fully rendered"


def test_synthesize_aborts_when_spend_cap_crossed() -> None:
    generator = _CountingGenerator(cost_usd=2.5)

    with pytest.raises(BudgetExceededError, match="exceeds cap"):
        asyncio.run(synthesize(generator, num_samples=10, base_seed=0, max_spend_usd=5.0))

    # Two calls succeed (totaling $5.00, equal to cap), the third pushes over.
    assert generator.calls == 3


def test_synthesize_completes_when_under_cap() -> None:
    generator = _CountingGenerator(cost_usd=0.5)

    results = asyncio.run(synthesize(generator, num_samples=5, base_seed=0, max_spend_usd=10.0))

    assert len(results) == 5
    assert generator.calls == 5


def test_run_surfaces_budget_exceeded_error(tmp_path: Path) -> None:
    config_path = _write_paid_config(tmp_path, max_spend_usd=0.1, num_samples=5)
    args = argparse.Namespace(config=config_path)

    with patch("scripts.synth.build_generator") as build:
        build.return_value = _CountingGenerator(cost_usd=0.5)
        with pytest.raises(BudgetExceededError):
            run(args)
