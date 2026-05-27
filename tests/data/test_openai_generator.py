from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from anvil.data.openai_generator import OpenAIGenerator


def _fake_response(content: str, prompt_tokens: int = 100, completion_tokens: int = 200) -> Any:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    response.choices[0].message.refusal = None
    response.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return response


def _valid_payload() -> dict[str, Any]:
    return {
        "contract_text": "# NDA between Acme and Globex...",
        "extraction": {
            "parties": [
                {"name": "Acme", "role": "disclosing_party"},
                {"name": "Globex", "role": "receiving_party"},
            ],
            "effective_date": "2026-02-15",
            "term": {
                "duration_months": 12,
                "is_perpetual": False,
                "auto_renew": False,
                "renewal_notice_days": None,
            },
            "governing_law": "Delaware",
            "jurisdiction": None,
            "confidentiality": None,
            "termination": {"triggers": [], "notice_days": None, "cure_period_days": None},
            "indemnification": None,
            "dispute_resolution": {"forum": "litigation", "venue": None, "governing_rules": None},
        },
    }


def _make_generator(content: str, **usage: int) -> tuple[OpenAIGenerator, AsyncMock]:
    client = MagicMock()
    create = AsyncMock(return_value=_fake_response(content, **usage))
    client.chat.completions.create = create
    return OpenAIGenerator(client=client), create


def test_generator_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="unknown OpenAI model"):
        OpenAIGenerator(model="totally-not-real", client=MagicMock())


def test_generate_returns_generation_result() -> None:
    payload = _valid_payload()
    generator, _ = _make_generator(json.dumps(payload))

    result = asyncio.run(generator.generate("nda", "sys", "user", seed=42))

    assert result.contract_type == "nda"
    assert result.contract_text == payload["contract_text"]
    assert result.extraction == payload["extraction"]
    assert result.input_tokens == 100
    assert result.output_tokens == 200
    assert result.backend == "openai:gpt-4o-2024-08-06"


def test_generate_computes_cost_from_pricing() -> None:
    generator, _ = _make_generator(
        json.dumps(_valid_payload()),
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )

    result = asyncio.run(generator.generate("nda", "sys", "user", seed=0))

    assert result.cost_usd == pytest.approx(2.50 + 10.00)


def test_generate_passes_seed_and_response_format_to_api() -> None:
    generator, create = _make_generator(json.dumps(_valid_payload()))

    asyncio.run(generator.generate("msa", "system here", "user here", seed=99))

    assert create.await_args is not None
    kwargs = create.await_args.kwargs
    assert kwargs["seed"] == 99
    assert kwargs["model"] == "gpt-4o-2024-08-06"
    assert kwargs["messages"][0]["role"] == "system"
    assert kwargs["messages"][0]["content"] == "system here"
    assert kwargs["messages"][1]["role"] == "user"
    assert kwargs["response_format"]["type"] == "json_schema"
    json_schema = kwargs["response_format"]["json_schema"]
    assert json_schema["strict"] is True
    assert json_schema["name"] == "contract_synthesis"
    assert "contract_text" in json_schema["schema"]["properties"]


def test_generate_raises_on_refusal() -> None:
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.refusal = "refused for policy reasons"
    response.choices[0].message.content = None
    create = AsyncMock(return_value=response)
    client.chat.completions.create = create
    generator = OpenAIGenerator(client=client)

    with pytest.raises(RuntimeError, match="refused"):
        asyncio.run(generator.generate("nda", "sys", "user", seed=0))


def test_generate_raises_on_non_object_response() -> None:
    generator, _ = _make_generator(json.dumps([1, 2, 3]))

    with pytest.raises(RuntimeError, match="not a JSON object"):
        asyncio.run(generator.generate("nda", "sys", "user", seed=0))


def test_generate_raises_on_missing_keys() -> None:
    generator, _ = _make_generator(json.dumps({"contract_text": "only this"}))

    with pytest.raises(RuntimeError, match="missing contract_text or extraction"):
        asyncio.run(generator.generate("nda", "sys", "user", seed=0))


def test_generate_raises_on_wrong_field_types() -> None:
    generator, _ = _make_generator(json.dumps({"contract_text": 5, "extraction": {}}))

    with pytest.raises(RuntimeError, match="contract_text is not a string"):
        asyncio.run(generator.generate("nda", "sys", "user", seed=0))


def test_generate_raises_when_extraction_not_object() -> None:
    generator, _ = _make_generator(json.dumps({"contract_text": "ok", "extraction": []}))

    with pytest.raises(RuntimeError, match="extraction is not an object"):
        asyncio.run(generator.generate("nda", "sys", "user", seed=0))
