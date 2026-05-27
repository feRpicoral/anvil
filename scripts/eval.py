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
import re
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
_VARIANT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


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
        test_jsonl = _required_path(raw, "test_jsonl", path)
        output_dir = _required_path(raw, "output_dir", path)
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
        name = _required_str(entry, "name", path)
        if _VARIANT_NAME_RE.fullmatch(name) is None:
            raise ValueError(f"{path}: invalid variant name {name!r}")
        if name in seen_names:
            raise ValueError(f"{path}: duplicate variant name {name!r}")
        seen_names.add(name)
        predictor = _required_str(entry, "predictor", path)
        if predictor not in _KNOWN_PREDICTORS:
            raise ValueError(
                f"{path}: variant {name!r} predictor must be one of {_KNOWN_PREDICTORS}"
            )
        variants.append(
            VariantConfig(
                name=name,
                predictor=predictor,
                fixtures_path=_optional_path(entry, "fixtures_path", path),
                model=_optional_str(entry, "model", path),
                base_model=_optional_str(entry, "base_model", path),
                adapter_path=_optional_path(entry, "adapter_path", path),
            )
        )
    return EvalConfig(test_jsonl=test_jsonl, output_dir=output_dir, variants=tuple(variants))


def load_test_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
                messages = row["messages"]
                if not isinstance(messages, list):
                    raise ValueError("messages must be a list")
                user_msg = _single_message(messages, "user")
                assistant_msg = _single_message(messages, "assistant")
                contract_text = _required_str(user_msg, "content", path)
                assistant_content = _required_str(assistant_msg, "content", path)
                gold = ContractExtraction.model_validate_json(assistant_content)
                case_id = _case_id(row, len(cases))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid test case: {exc}") from exc
            if case_id in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate case_id {case_id!r}")
            seen_ids.add(case_id)
            cases.append(
                EvalCase(
                    case_id=case_id,
                    contract_text=contract_text,
                    gold_extraction=gold,
                )
            )
    if not cases:
        raise ValueError(f"{path}: no test cases found")
    return cases


def load_fixture_predictions(path: Path) -> dict[str, Prediction]:
    by_id: dict[str, Prediction] = {}
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
                if not isinstance(row, dict):
                    raise ValueError("row must be an object")
                case_id = _required_str(row, "case_id", path)
                if case_id in by_id:
                    raise ValueError(f"duplicate case_id {case_id!r}")
                by_id[case_id] = Prediction(
                    raw_output=_required_str(row, "raw_output", path),
                    input_tokens=_non_negative_int(row.get("input_tokens", 0), "input_tokens"),
                    output_tokens=_non_negative_int(row.get("output_tokens", 0), "output_tokens"),
                    cost_usd=_non_negative_float(row.get("cost_usd", 0.0), "cost_usd"),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid fixture prediction: {exc}"
                ) from exc
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
            f"cost=${summary.total_cost_usd:.4f} -> {path}",
            file=sys.stderr,
        )
        summaries.append(summary)
    comparison_path = write_comparison(config.output_dir, summaries)
    print(f"eval: wrote comparison -> {comparison_path}", file=sys.stderr)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the three-way extraction eval.")
    parser.add_argument("--config", type=Path, required=True, help="TOML eval config.")
    return parser.parse_args(argv)


def main() -> int:
    return run(parse_args())


def _required_str(mapping: dict[str, Any], key: str, path: Path) -> str:
    value = mapping[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}: {key} must be a non-empty string")
    return value


def _optional_str(mapping: dict[str, Any], key: str, path: Path) -> str | None:
    if key not in mapping:
        return None
    return _required_str(mapping, key, path)


def _required_path(mapping: dict[str, Any], key: str, path: Path) -> Path:
    return Path(_required_str(mapping, key, path))


def _optional_path(mapping: dict[str, Any], key: str, path: Path) -> Path | None:
    value = _optional_str(mapping, key, path)
    return Path(value) if value is not None else None


def _single_message(messages: list[Any], role: str) -> dict[str, Any]:
    matches = [m for m in messages if isinstance(m, dict) and m.get("role") == role]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {role!r} message")
    return matches[0]


def _case_id(row: dict[str, Any], index: int) -> str:
    if "case_id" not in row or row["case_id"] is None:
        return f"case-{index:04d}"
    if not isinstance(row["case_id"], str) or not row["case_id"]:
        raise ValueError("case_id must be a non-empty string")
    return row["case_id"]


def _non_negative_int(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _non_negative_float(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise ValueError(f"{key} must be a non-negative number")
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
