from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from anvil.data.schema import ContractExtraction
from anvil.eval.runner import (
    EvalCase,
    ExtractionPredictor,
    FixturePredictor,
    Prediction,
    run_variant,
    summarize_variant,
)


def _gold_payload() -> dict[str, Any]:
    return {
        "parties": [
            {"name": "Acme Corp.", "role": "disclosing_party"},
            {"name": "Globex Industries LLC", "role": "receiving_party"},
        ],
        "effective_date": "2026-02-15",
        "term": {
            "duration_months": 24,
            "is_perpetual": False,
            "auto_renew": False,
            "renewal_notice_days": None,
        },
        "governing_law": "Delaware",
        "jurisdiction": None,
        "confidentiality": None,
        "termination": {
            "triggers": ["material breach"],
            "notice_days": 30,
            "cure_period_days": 15,
        },
        "indemnification": None,
        "dispute_resolution": {
            "forum": "litigation",
            "venue": "Wilmington, Delaware",
            "governing_rules": None,
        },
    }


def _case(case_id: str, contract_text: str | None = None) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        contract_text=contract_text or f"Contract body for {case_id}",
        gold_extraction=ContractExtraction.model_validate(_gold_payload()),
    )


def _prediction(
    payload: dict[str, Any] | None = None, raw_override: str | None = None
) -> Prediction:
    if raw_override is not None:
        return Prediction(
            raw_output=raw_override, input_tokens=100, output_tokens=200, cost_usd=0.05
        )
    return Prediction(
        raw_output=json.dumps(payload or _gold_payload()),
        input_tokens=100,
        output_tokens=200,
        cost_usd=0.05,
    )


def test_fixture_predictor_requires_at_least_one_entry() -> None:
    with pytest.raises(ValueError, match="at least one"):
        FixturePredictor({})


def test_fixture_predictor_satisfies_protocol() -> None:
    predictor = FixturePredictor({"c0": _prediction()})

    assert isinstance(predictor, ExtractionPredictor)


def test_fixture_predictor_raises_for_unmapped_text() -> None:
    predictor = FixturePredictor({"c0": _prediction()})

    with pytest.raises(KeyError, match="attach_cases"):
        asyncio.run(predictor.predict("unmapped contract text"))


def test_fixture_predictor_raises_for_missing_prediction() -> None:
    case = _case("missing")
    predictor = FixturePredictor({"c0": _prediction()})
    predictor.attach_cases([case])

    with pytest.raises(KeyError, match="missing"):
        asyncio.run(predictor.predict(case.contract_text))


def test_run_variant_returns_one_output_per_case() -> None:
    cases = [_case("c0"), _case("c1"), _case("c2")]
    predictor = FixturePredictor(
        {
            "c0": _prediction(),
            "c1": _prediction(),
            "c2": _prediction(),
        }
    )

    outputs = asyncio.run(run_variant(predictor, cases, variant="finetuned"))

    assert len(outputs) == 3
    assert {o.case_id for o in outputs} == {"c0", "c1", "c2"}
    assert {o.variant for o in outputs} == {"finetuned"}
    for output in outputs:
        assert output.parsed is not None
        assert output.parse_reason is None
        assert output.latency_ms >= 0


def test_run_variant_parses_valid_payload() -> None:
    cases = [_case("c0")]
    predictor = FixturePredictor({"c0": _prediction()})

    outputs = asyncio.run(run_variant(predictor, cases, variant="base"))

    assert outputs[0].parsed is not None
    assert len(outputs[0].parsed.parties) == 2


def test_run_variant_records_invalid_json() -> None:
    cases = [_case("c0")]
    predictor = FixturePredictor({"c0": _prediction(raw_override="{not json")})

    outputs = asyncio.run(run_variant(predictor, cases, variant="base"))

    assert outputs[0].parsed is None
    assert outputs[0].parse_reason is not None
    assert "json_decode" in outputs[0].parse_reason


def test_run_variant_carries_accounting_fields() -> None:
    cases = [_case("c0")]
    predictor = FixturePredictor({"c0": _prediction()})

    outputs = asyncio.run(run_variant(predictor, cases, variant="gpt-4o"))

    assert outputs[0].input_tokens == 100
    assert outputs[0].output_tokens == 200
    assert outputs[0].cost_usd == 0.05


def test_summarize_variant_empty_outputs_returns_zeros() -> None:
    summary = summarize_variant([], [])

    assert summary.n_cases == 0
    assert summary.json_validity_rate == 0.0
    assert summary.field_scores == {}


def test_summarize_variant_perfect_predictions_score_one() -> None:
    cases = [_case(f"c{i}") for i in range(3)]
    predictor = FixturePredictor({c.case_id: _prediction() for c in cases})

    outputs = asyncio.run(run_variant(predictor, cases, variant="finetuned"))
    summary = summarize_variant(outputs, cases)

    assert summary.n_cases == 3
    assert summary.json_validity_rate == 1.0
    assert all(score == 1.0 for score in summary.field_scores.values())
    assert summary.total_cost_usd == pytest.approx(0.15)


def test_summarize_variant_partial_invalidity_lowers_rate() -> None:
    cases = [_case(f"c{i}") for i in range(4)]
    predictor = FixturePredictor(
        {
            "c0": _prediction(),
            "c1": _prediction(raw_override="{bad"),
            "c2": _prediction(),
            "c3": _prediction(raw_override="not json"),
        }
    )

    outputs = asyncio.run(run_variant(predictor, cases, variant="base"))
    summary = summarize_variant(outputs, cases)

    assert summary.json_validity_rate == 0.5
    assert summary.field_scores  # only computed over valid outputs


def test_summarize_variant_raises_when_gold_missing_for_a_valid_output() -> None:
    cases = [_case("c0")]
    predictor = FixturePredictor({"c0": _prediction()})

    outputs = asyncio.run(run_variant(predictor, cases, variant="base"))
    # Strip the case from the lookup the summarizer relies on.
    with pytest.raises(KeyError, match="gold extraction"):
        summarize_variant(outputs, cases=[])


def test_summarize_variant_aggregates_costs_and_latency() -> None:
    cases = [_case(f"c{i}") for i in range(3)]
    predictor = FixturePredictor({c.case_id: _prediction() for c in cases})

    outputs = asyncio.run(run_variant(predictor, cases, variant="gpt-4o"))
    summary = summarize_variant(outputs, cases)

    assert summary.total_input_tokens == 300
    assert summary.total_output_tokens == 600
    assert summary.total_cost_usd == pytest.approx(0.15)
    assert summary.mean_latency_ms >= 0


def test_summarize_variant_variant_label_propagates() -> None:
    cases = [_case("c0")]
    predictor = FixturePredictor({"c0": _prediction()})

    outputs = asyncio.run(run_variant(predictor, cases, variant="gpt-4o-baseline"))
    summary = summarize_variant(outputs, cases)

    assert summary.variant == "gpt-4o-baseline"
