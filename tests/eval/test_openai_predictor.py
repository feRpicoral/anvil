from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from anvil.eval.openai_predictor import (
    OpenAIExtractionPredictor,
    extraction_response_schema,
)
from anvil.eval.runner import ExtractionPredictor


def _fake_response(content: str, prompt_tokens: int = 200, completion_tokens: int = 300) -> Any:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.choices[0].message.refusal = None
    response.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return response


def _extraction_payload() -> dict[str, Any]:
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


def _make_predictor(
    content: str,
    *,
    seed: int | None = 0,
    **usage: int,
) -> tuple[OpenAIExtractionPredictor, AsyncMock]:
    client = MagicMock()
    create = AsyncMock(return_value=_fake_response(content, **usage))
    client.chat.completions.create = create
    return OpenAIExtractionPredictor(client=client, seed=seed), create


def test_predictor_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="unknown OpenAI model"):
        OpenAIExtractionPredictor(model="not-a-real-model", client=MagicMock())


def test_predictor_satisfies_protocol() -> None:
    predictor, _ = _make_predictor(json.dumps(_extraction_payload()))

    assert isinstance(predictor, ExtractionPredictor)


def test_extraction_response_schema_is_strict() -> None:
    schema = extraction_response_schema()

    assert schema["name"] == "contract_extraction"
    assert schema["strict"] is True
    assert schema["schema"]["type"] == "object"
    assert "parties" in schema["schema"]["properties"]
    assert "dispute_resolution" in schema["schema"]["properties"]


def test_predict_returns_raw_json_in_prediction() -> None:
    payload = _extraction_payload()
    predictor, _ = _make_predictor(json.dumps(payload))

    prediction = asyncio.run(predictor.predict("This Mutual NDA is entered into..."))

    assert json.loads(prediction.raw_output) == payload


def test_predict_computes_cost_from_pricing() -> None:
    predictor, _ = _make_predictor(
        json.dumps(_extraction_payload()),
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )

    prediction = asyncio.run(predictor.predict("contract text"))

    assert prediction.cost_usd == pytest.approx(2.50 + 10.00)


def test_predict_passes_seed_and_strict_response_format_to_api() -> None:
    predictor, create = _make_predictor(json.dumps(_extraction_payload()))

    asyncio.run(predictor.predict("This Mutual NDA..."))

    assert create.await_args is not None
    kwargs = create.await_args.kwargs
    assert kwargs["model"] == "gpt-4o-2024-08-06"
    assert kwargs["seed"] == 0
    assert kwargs["messages"][0]["role"] == "system"
    assert "JSON object" in kwargs["messages"][0]["content"]
    assert kwargs["messages"][1]["role"] == "user"
    assert kwargs["messages"][1]["content"] == "This Mutual NDA..."
    assert kwargs["response_format"]["type"] == "json_schema"
    schema = kwargs["response_format"]["json_schema"]
    assert schema["name"] == "contract_extraction"
    assert schema["strict"] is True
    assert "parties" in schema["schema"]["properties"]


def test_predict_can_disable_seed() -> None:
    predictor, create = _make_predictor(json.dumps(_extraction_payload()), seed=None)

    asyncio.run(predictor.predict("This Mutual NDA..."))

    assert create.await_args is not None
    assert "seed" not in create.await_args.kwargs


def test_predict_propagates_token_counts() -> None:
    predictor, _ = _make_predictor(
        json.dumps(_extraction_payload()),
        prompt_tokens=2_000,
        completion_tokens=400,
    )

    prediction = asyncio.run(predictor.predict("contract"))

    assert prediction.input_tokens == 2_000
    assert prediction.output_tokens == 400


def test_predict_raises_on_refusal() -> None:
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.refusal = "Cannot process for policy reasons"
    response.choices[0].message.content = None
    client.chat.completions.create = AsyncMock(return_value=response)
    predictor = OpenAIExtractionPredictor(client=client)

    with pytest.raises(RuntimeError, match="refused"):
        asyncio.run(predictor.predict("contract"))


def test_predict_handles_missing_usage() -> None:
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = json.dumps(_extraction_payload())
    response.choices[0].message.refusal = None
    response.usage = None
    client.chat.completions.create = AsyncMock(return_value=response)
    predictor = OpenAIExtractionPredictor(client=client)

    prediction = asyncio.run(predictor.predict("contract"))

    assert prediction.input_tokens == 0
    assert prediction.output_tokens == 0
    assert prediction.cost_usd == 0.0


def test_model_property_exposes_choice() -> None:
    predictor, _ = _make_predictor(json.dumps(_extraction_payload()))

    assert predictor.model == "gpt-4o-2024-08-06"
