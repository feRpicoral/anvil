"""Run the three-way eval (base / fine-tuned / GPT-4o) end-to-end.

Reads a TOML config, loads messages-format test cases, builds a predictor
per variant (fixture / openai / local), runs each over the cases, and
writes per-variant + comparison JSON to `output_dir`.

The smoke path uses only `fixture` predictors so `make eval-smoke` runs
in CI without an API key or a model load. Real predictors require the
corresponding install (openai SDK is part of base deps; local requires
`constraints/train.txt`).
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

from anvil.data.schema import ContractExtraction
from anvil.eval.runner import (
    EvalCase,
    ExtractionPredictor,
    FixturePredictor,
    Prediction,
    VariantSummary,
    run_variant,
    summarize_variant,
)

_KNOWN_PREDICTORS = ("fixture", "openai", "local")


@dataclasses.dataclass(frozen=True)
class VariantConfig:
    name: str
    predictor: str
    fixtures_path: Path | None = None
    model: str | None = None
    base_model: str | None = None
    adapter_path: Path | None = None


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    test_jsonl: Path
    output_dir: Path
    variants: tuple[VariantConfig, ...]


def load_eval_config(path: Path) -> EvalConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    try:
        test_jsonl = Path(str(raw["test_jsonl"]))
        output_dir = Path(str(raw["output_dir"]))
        variants_raw = raw["variant"]
    except KeyError as exc:
        raise ValueError(f"{path}: missing required key {exc}") from exc
    if not isinstance(variants_raw, list) or not variants_raw:
        raise ValueError(f"{path}: 'variant' must be a non-empty array of tables")
    seen_names: set[str] = set()
    variants: list[VariantConfig] = []
    for entry in variants_raw:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: variant entries must be tables")
        name = str(entry["name"])
        if name in seen_names:
            raise ValueError(f"{path}: duplicate variant name {name!r}")
        seen_names.add(name)
        predictor = str(entry["predictor"])
        if predictor not in _KNOWN_PREDICTORS:
            raise ValueError(
                f"{path}: variant {name!r} predictor must be one of {_KNOWN_PREDICTORS}"
            )
        variants.append(
            VariantConfig(
                name=name,
                predictor=predictor,
                fixtures_path=Path(str(entry["fixtures_path"]))
                if "fixtures_path" in entry
                else None,
                model=str(entry["model"]) if "model" in entry else None,
                base_model=str(entry["base_model"]) if "base_model" in entry else None,
                adapter_path=Path(str(entry["adapter_path"])) if "adapter_path" in entry else None,
            )
        )
    return EvalConfig(test_jsonl=test_jsonl, output_dir=output_dir, variants=tuple(variants))


def load_test_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open(encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            messages = row["messages"]
            user_msg = next(m for m in messages if m["role"] == "user")
            assistant_msg = next(m for m in messages if m["role"] == "assistant")
            gold = ContractExtraction.model_validate_json(assistant_msg["content"])
            case_id = row.get("case_id") or f"case-{index:04d}"
            cases.append(
                EvalCase(
                    case_id=str(case_id),
                    contract_text=user_msg["content"],
                    gold_extraction=gold,
                )
            )
    if not cases:
        raise ValueError(f"{path}: no test cases found")
    return cases


def load_fixture_predictions(path: Path) -> dict[str, Prediction]:
    by_id: dict[str, Prediction] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            by_id[str(row["case_id"])] = Prediction(
                raw_output=str(row["raw_output"]),
                input_tokens=int(row.get("input_tokens", 0)),
                output_tokens=int(row.get("output_tokens", 0)),
                cost_usd=float(row.get("cost_usd", 0.0)),
            )
    if not by_id:
        raise ValueError(f"{path}: no fixture predictions found")
    return by_id


def build_predictor(config: VariantConfig) -> ExtractionPredictor:
    if config.predictor == "fixture":
        if config.fixtures_path is None:
            raise ValueError(f"variant {config.name!r}: fixture predictor requires fixtures_path")
        return FixturePredictor(load_fixture_predictions(config.fixtures_path))
    if config.predictor == "openai":
        from anvil.eval.openai_predictor import OpenAIExtractionPredictor

        if config.model is not None:
            return OpenAIExtractionPredictor(model=config.model)
        return OpenAIExtractionPredictor()
    if config.predictor == "local":
        from anvil.eval.local_predictor import LocalExtractionPredictor

        if not config.base_model:
            raise ValueError(f"variant {config.name!r}: local predictor requires base_model")
        return LocalExtractionPredictor(
            base_model=config.base_model,
            adapter_path=config.adapter_path,
        )
    raise ValueError(f"unknown predictor type: {config.predictor!r}")


def write_summary(output_dir: Path, summary: VariantSummary) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{summary.variant}.json"
    path.write_text(json.dumps(dataclasses.asdict(summary), indent=2) + "\n", encoding="utf-8")
    return path


def write_comparison(output_dir: Path, summaries: Sequence[VariantSummary]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "comparison.json"
    payload: dict[str, Any] = {
        "n_cases": summaries[0].n_cases if summaries else 0,
        "variants": [dataclasses.asdict(s) for s in summaries],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> int:
    config = load_eval_config(args.config)
    cases = load_test_cases(config.test_jsonl)
    summaries: list[VariantSummary] = []
    for variant_config in config.variants:
        predictor = build_predictor(variant_config)
        outputs = asyncio.run(run_variant(predictor, cases, variant_config.name))
        summary = summarize_variant(outputs, cases)
        path = write_summary(config.output_dir, summary)
        print(
            f"eval: {variant_config.name} validity={summary.json_validity_rate:.2f} "
            f"cost=${summary.total_cost_usd:.4f} → {path}",
            file=sys.stderr,
        )
        summaries.append(summary)
    comparison_path = write_comparison(config.output_dir, summaries)
    print(f"eval: wrote comparison → {comparison_path}", file=sys.stderr)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the three-way extraction eval.")
    parser.add_argument("--config", type=Path, required=True, help="TOML eval config.")
    return parser.parse_args(argv)


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
