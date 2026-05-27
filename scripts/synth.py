"""Drive synthesis with a `StructuredGenerator` and write raw JSONL.

The smoke path uses `FixtureGenerator` so the command runs with zero API
spend. Real OpenAI/Anthropic backends slot into the same Protocol in a
follow-up PR; this script's only knob is the `backend` key in the config.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from anvil.data.prompts import CONTRACT_TYPES, ContractType
from anvil.data.synthesis import FixtureGenerator, GenerationResult, StructuredGenerator

_RAW_SYNTHESIS_FILENAME = "raw_synthesis.jsonl"


@dataclasses.dataclass(frozen=True)
class SynthConfig:
    backend: str
    fixtures_dir: Path | None
    num_samples: int
    output_dir: Path
    seed: int


def load_config(path: Path) -> SynthConfig:
    """Parse a TOML config file into a `SynthConfig`."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    try:
        backend = str(raw["backend"])
        num_samples = int(raw["num_samples"])
        output_dir = Path(str(raw["output_dir"]))
        seed = int(raw.get("seed", 0))
    except KeyError as exc:
        raise ValueError(f"{path}: missing required key {exc}") from exc
    fixtures_dir = Path(str(raw["fixtures_dir"])) if "fixtures_dir" in raw else None
    return SynthConfig(
        backend=backend,
        fixtures_dir=fixtures_dir,
        num_samples=num_samples,
        output_dir=output_dir,
        seed=seed,
    )


def build_generator(config: SynthConfig) -> StructuredGenerator:
    """Construct the generator named in the config."""
    if config.backend == "fixture":
        if config.fixtures_dir is None:
            raise ValueError("fixture backend requires fixtures_dir")
        return FixtureGenerator(config.fixtures_dir)
    raise ValueError(f"unsupported backend: {config.backend!r}")


async def synthesize(
    generator: StructuredGenerator,
    num_samples: int,
    base_seed: int,
) -> list[GenerationResult]:
    """Produce `num_samples` records by cycling contract types and seeds.

    The contract type rotates `nda → msa → license → nda → ...` so the
    smoke dataset stays balanced across classes even for small N.
    """
    results: list[GenerationResult] = []
    for index in range(num_samples):
        contract_type: ContractType = CONTRACT_TYPES[index % len(CONTRACT_TYPES)]
        result = await generator.generate(
            contract_type=contract_type,
            system_prompt="",
            user_prompt="",
            seed=base_seed + index,
        )
        results.append(result)
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
    results = asyncio.run(synthesize(generator, config.num_samples, config.seed))
    output_path = config.output_dir / _RAW_SYNTHESIS_FILENAME
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
