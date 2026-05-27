"""Drive synthesis with a `StructuredGenerator` and write raw JSONL.

The smoke path uses `FixtureGenerator` so the command runs with zero API
spend. The paid path swaps in `OpenAIGenerator` or `AnthropicGenerator`
behind the same Protocol; the only knob is the `backend` key in the
config plus an optional `max_spend_usd` hard cap.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import sys
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from anvil.data.anthropic_generator import AnthropicGenerator
from anvil.data.openai_generator import OpenAIGenerator
from anvil.data.parameters import generate_parameters
from anvil.data.prompts import (
    CONTRACT_TYPES,
    ContractType,
    PromptParameters,
    render_user_prompt,
    system_prompt,
)
from anvil.data.synthesis import FixtureGenerator, GenerationResult, StructuredGenerator

_RAW_SYNTHESIS_FILENAME = "raw_synthesis.jsonl"


class BudgetExceededError(RuntimeError):
    """Raised when running spend crosses `max_spend_usd` mid-synthesis."""

    def __init__(self, message: str, partial_results: Sequence[GenerationResult]) -> None:
        super().__init__(message)
        self.partial_results = tuple(partial_results)


@dataclasses.dataclass(frozen=True)
class SynthConfig:
    backend: str
    fixtures_dir: Path | None
    num_samples: int
    output_dir: Path
    seed: int
    model: str | None = None
    max_spend_usd: float | None = None


def load_config(path: Path) -> SynthConfig:
    """Parse a TOML config file into a `SynthConfig`."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    try:
        backend = str(raw["backend"])
        num_samples = _positive_int(raw["num_samples"], "num_samples")
        output_dir = Path(str(raw["output_dir"]))
        seed = int(raw.get("seed", 0))
    except KeyError as exc:
        raise ValueError(f"{path}: missing required key {exc}") from exc
    fixtures_dir = Path(str(raw["fixtures_dir"])) if "fixtures_dir" in raw else None
    model = str(raw["model"]) if "model" in raw else None
    max_spend_usd = _positive_float(raw["max_spend_usd"]) if "max_spend_usd" in raw else None
    return SynthConfig(
        backend=backend,
        fixtures_dir=fixtures_dir,
        num_samples=num_samples,
        output_dir=output_dir,
        seed=seed,
        model=model,
        max_spend_usd=max_spend_usd,
    )


def build_generator(config: SynthConfig) -> StructuredGenerator:
    """Construct the generator named in the config."""
    if config.backend == "fixture":
        if config.fixtures_dir is None:
            raise ValueError("fixture backend requires fixtures_dir")
        return FixtureGenerator(config.fixtures_dir)
    if config.backend == "openai":
        if config.model is not None:
            return OpenAIGenerator(model=config.model)
        return OpenAIGenerator()
    if config.backend == "anthropic":
        if config.model is not None:
            return AnthropicGenerator(model=config.model)
        return AnthropicGenerator()
    raise ValueError(f"unsupported backend: {config.backend!r}")


def _positive_int(value: object, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _positive_float(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("max_spend_usd must be a positive number")
    if isinstance(value, int | float):
        result = float(value)
        if result <= 0 or not math.isfinite(result):
            raise ValueError("max_spend_usd must be a positive number")
        return result
    raise ValueError("max_spend_usd must be a positive number")


async def synthesize(
    generator: StructuredGenerator,
    num_samples: int,
    base_seed: int,
    max_spend_usd: float | None = None,
    parameter_factory: Callable[[ContractType, int, int], PromptParameters] | None = None,
) -> list[GenerationResult]:
    """Produce `num_samples` records by cycling contract types.

    Contract type rotates `nda → msa → license → nda → ...` so the dataset
    stays balanced across classes for any N. Each call gets a fresh
    `PromptParameters` from `parameter_factory` (default: `generate_parameters`),
    rendered as a user prompt alongside the shared system prompt.

    `max_spend_usd` is enforced after each call. The first call that would
    cross the cap raises `BudgetExceededError`; the partial result list is
    still saved on disk so an operator can inspect what landed.
    """
    factory = parameter_factory if parameter_factory is not None else generate_parameters
    sys_prompt = system_prompt()
    results: list[GenerationResult] = []
    total_cost = 0.0
    for index in range(num_samples):
        contract_type: ContractType = CONTRACT_TYPES[index % len(CONTRACT_TYPES)]
        parameters = factory(contract_type, index, base_seed)
        user_prompt = render_user_prompt(parameters)
        result = await generator.generate(
            contract_type=contract_type,
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            seed=base_seed + index,
        )
        results.append(result)
        total_cost += result.cost_usd
        if max_spend_usd is not None and total_cost > max_spend_usd:
            raise BudgetExceededError(
                f"spend ${total_cost:.4f} exceeds cap ${max_spend_usd:.4f} "
                f"after {len(results)} samples",
                results,
            )
    return results


def write_raw_jsonl(results: Sequence[GenerationResult], path: Path) -> None:
    """Serialize `results` as JSONL of `dataclasses.asdict` rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(dataclasses.asdict(result), ensure_ascii=False))
            fh.write("\n")


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    generator = build_generator(config)
    output_path = config.output_dir / _RAW_SYNTHESIS_FILENAME
    try:
        results = asyncio.run(
            synthesize(
                generator,
                config.num_samples,
                config.seed,
                max_spend_usd=config.max_spend_usd,
            )
        )
    except BudgetExceededError as exc:
        write_raw_jsonl(exc.partial_results, output_path)
        print(f"synth: ABORT — {exc}", file=sys.stderr)
        raise
    write_raw_jsonl(results, output_path)
    total_cost = sum(result.cost_usd for result in results)
    print(
        f"synth: wrote {len(results)} records to {output_path} "
        f"(backend={config.backend}, cost=${total_cost:.4f})",
        file=sys.stderr,
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic contract data.")
    parser.add_argument("--config", type=Path, required=True, help="TOML synthesis config.")
    return parser.parse_args(argv)


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())


def load_results(path: Path) -> list[GenerationResult]:
    """Reverse of `write_raw_jsonl`. Used by the next step in the pipeline."""
    out: list[GenerationResult] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            out.append(GenerationResult(**row))
    return out
