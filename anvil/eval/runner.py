"""Evaluation runner shared by local, fine-tuned, and hosted predictors."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from anvil.data.schema import ContractExtraction
from anvil.eval.metrics import (
    aggregate_scores,
    score_extraction,
    validate_json_output,
)


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    contract_text: str
    gold_extraction: ContractExtraction


@dataclass(frozen=True)
class Prediction:
    raw_output: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True)
class VariantOutput:
    case_id: str
    variant: str
    raw_output: str
    parsed: ContractExtraction | None
    parse_reason: str | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


@runtime_checkable
class ExtractionPredictor(Protocol):
    async def predict(self, contract_text: str) -> Prediction: ...


@dataclass(frozen=True)
class VariantSummary:
    variant: str
    n_cases: int
    json_validity_rate: float
    field_scores: dict[str, float]
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    mean_latency_ms: float


class FixturePredictor:
    """Replay canned predictions keyed by case_id; zero API cost.

    Lets the eval runner be exercised in CI without spinning a model or
    hitting an API. Raises `KeyError` for unknown case_ids so a missing
    fixture fails the run instead of silently producing zeros.
    """

    def __init__(self, predictions_by_case_id: dict[str, Prediction]) -> None:
        if not predictions_by_case_id:
            raise ValueError("FixturePredictor needs at least one prediction")
        self._by_id = dict(predictions_by_case_id)
        self._lookup: dict[str, str] = {}

    def attach_cases(self, cases: Iterable[EvalCase]) -> None:
        lookup: dict[str, str] = {}
        for case in cases:
            if case.contract_text in lookup:
                raise ValueError(f"duplicate contract_text for case_id={case.case_id!r}")
            lookup[case.contract_text] = case.case_id
        self._lookup = lookup

    async def predict(self, contract_text: str) -> Prediction:
        case_id = self._lookup.get(contract_text)
        if case_id is None:
            raise KeyError(
                "FixturePredictor has no case_id mapped for the supplied contract_text; "
                "call attach_cases() first"
            )
        if case_id not in self._by_id:
            raise KeyError(f"FixturePredictor has no prediction for case_id={case_id!r}")
        return self._by_id[case_id]


async def run_variant(
    predictor: ExtractionPredictor,
    cases: Iterable[EvalCase],
    variant: str,
) -> list[VariantOutput]:
    case_list = list(cases)
    if isinstance(predictor, FixturePredictor):
        predictor.attach_cases(case_list)
    outputs: list[VariantOutput] = []
    for case in case_list:
        start = time.perf_counter()
        prediction = await predictor.predict(case.contract_text)
        elapsed_ms = (time.perf_counter() - start) * 1000
        validity = validate_json_output(prediction.raw_output)
        parsed: ContractExtraction | None = None
        if validity.valid:
            parsed = ContractExtraction.model_validate(json.loads(prediction.raw_output))
        outputs.append(
            VariantOutput(
                case_id=case.case_id,
                variant=variant,
                raw_output=prediction.raw_output,
                parsed=parsed,
                parse_reason=validity.reason,
                input_tokens=prediction.input_tokens,
                output_tokens=prediction.output_tokens,
                cost_usd=prediction.cost_usd,
                latency_ms=elapsed_ms,
            )
        )
    return outputs


def summarize_variant(outputs: list[VariantOutput], cases: list[EvalCase]) -> VariantSummary:
    """Aggregate outputs for one variant.

    JSON validity is measured against the full case set; missing outputs
    count as invalid. Field scores only use parsed outputs.
    """
    gold_by_case_id: dict[str, ContractExtraction] = {}
    for case in cases:
        if case.case_id in gold_by_case_id:
            raise ValueError(f"duplicate case_id={case.case_id!r}")
        gold_by_case_id[case.case_id] = case.gold_extraction

    variants = {output.variant for output in outputs}
    if len(variants) > 1:
        raise ValueError(f"cannot summarize mixed variants: {sorted(variants)}")

    output_case_ids: set[str] = set()
    for output in outputs:
        if output.case_id in output_case_ids:
            raise ValueError(f"duplicate output for case_id={output.case_id!r}")
        output_case_ids.add(output.case_id)
        if output.case_id not in gold_by_case_id:
            raise KeyError(f"no gold extraction for case_id={output.case_id!r}")

    n_cases = len(cases)
    if not outputs:
        return VariantSummary(
            variant="",
            n_cases=n_cases,
            json_validity_rate=0.0,
            field_scores={},
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost_usd=0.0,
            mean_latency_ms=0.0,
        )
    variant = outputs[0].variant
    valid_outputs = [o for o in outputs if o.parsed is not None]
    validity_rate = len(valid_outputs) / n_cases if n_cases > 0 else 0.0
    per_case_scores: list[dict[str, float]] = []
    for output in valid_outputs:
        assert output.parsed is not None
        per_case_scores.append(score_extraction(output.parsed, gold_by_case_id[output.case_id]))
    field_scores = aggregate_scores(per_case_scores)
    return VariantSummary(
        variant=variant,
        n_cases=n_cases,
        json_validity_rate=validity_rate,
        field_scores=field_scores,
        total_input_tokens=sum(o.input_tokens for o in outputs),
        total_output_tokens=sum(o.output_tokens for o in outputs),
        total_cost_usd=sum(o.cost_usd for o in outputs),
        mean_latency_ms=sum(o.latency_ms for o in outputs) / len(outputs),
    )
