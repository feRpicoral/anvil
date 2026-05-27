from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from anvil.data.anthropic_generator import AnthropicGenerator


def _text_block(text: str) -> Any:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _fake_response(text: str, input_tokens: int = 100, output_tokens: int = 200) -> Any:
    response = MagicMock()
    response.content = [_text_block(text)]
    response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
    return response


def _valid_payload() -> dict[str, Any]:
    return {
        "contract_text": "# Mutual NDA",
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


def _make_generator(text: str, **usage: int) -> tuple[AnthropicGenerator, AsyncMock]:
    client = MagicMock()
    create = AsyncMock(return_value=_fake_response(text, **usage))
    client.messages.create = create
    return AnthropicGenerator(client=client), create


def test_generator_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="unknown Anthropic model"):
        AnthropicGenerator(model="not-a-real-claude", client=MagicMock())


def test_generator_rejects_zero_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens must be positive"):
        AnthropicGenerator(client=MagicMock(), max_tokens=0)


def test_generator_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError, match="temperature must be between 0 and 1"):
        AnthropicGenerator(client=MagicMock(), temperature=1.1)


def test_generate_returns_generation_result() -> None:
    payload = _valid_payload()
    generator, _ = _make_generator(json.dumps(payload))

    result = asyncio.run(generator.generate("nda", "sys", "user", seed=42))

    assert result.contract_type == "nda"
    assert result.contract_text == payload["contract_text"]
    assert result.extraction == payload["extraction"]
    assert result.input_tokens == 100
    assert result.output_tokens == 200
    assert result.backend == "anthropic:claude-sonnet-4-6"


def test_generate_computes_cost_from_pricing() -> None:
    generator, _ = _make_generator(
        json.dumps(_valid_payload()),
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )

    result = asyncio.run(generator.generate("nda", "sys", "user", seed=0))

    assert result.cost_usd == pytest.approx(3.0 + 15.0)


def test_generate_handles_code_fenced_response() -> None:
    payload = _valid_payload()
    fenced = "```json\n" + json.dumps(payload) + "\n```"
    generator, _ = _make_generator(fenced)

    result = asyncio.run(generator.generate("nda", "sys", "user", seed=0))

    assert result.extraction == payload["extraction"]


def test_generate_handles_code_fenced_without_lang_tag() -> None:
    payload = _valid_payload()
    fenced = "```\n" + json.dumps(payload) + "\n```"
    generator, _ = _make_generator(fenced)

    result = asyncio.run(generator.generate("nda", "sys", "user", seed=0))

    assert result.contract_text == payload["contract_text"]


def test_generate_raises_on_non_object() -> None:
    generator, _ = _make_generator(json.dumps([1, 2]))

    with pytest.raises(RuntimeError, match="not a JSON object"):
        asyncio.run(generator.generate("nda", "sys", "user", seed=0))


def test_generate_raises_on_missing_keys() -> None:
    generator, _ = _make_generator(json.dumps({"contract_text": "ok"}))

    with pytest.raises(RuntimeError, match="missing contract_text or extraction"):
        asyncio.run(generator.generate("nda", "sys", "user", seed=0))


def test_generate_raises_when_extraction_fails_schema() -> None:
    generator, _ = _make_generator(json.dumps({"contract_text": "ok", "extraction": {}}))

    with pytest.raises(RuntimeError, match="extraction does not match schema"):
        asyncio.run(generator.generate("nda", "sys", "user", seed=0))


def test_generate_raises_when_no_text_blocks() -> None:
    client = MagicMock()
    response = MagicMock()
    response.content = []
    response.usage = MagicMock(input_tokens=0, output_tokens=0)
    client.messages.create = AsyncMock(return_value=response)
    generator = AnthropicGenerator(client=client)

    with pytest.raises(RuntimeError, match="no text blocks"):
        asyncio.run(generator.generate("nda", "sys", "user", seed=0))


def test_generate_passes_system_and_user_to_api() -> None:
    generator, create = _make_generator(json.dumps(_valid_payload()))

    asyncio.run(generator.generate("msa", "system text", "user text", seed=0))

    assert create.await_args is not None
    kwargs = create.await_args.kwargs
    assert kwargs["system"].startswith("system text")
    assert "Use this JSON Schema" in kwargs["system"]
    assert "contract_text" in kwargs["system"]
    assert "extraction" in kwargs["system"]
    assert kwargs["messages"][0]["content"] == "user text"
    assert kwargs["max_tokens"] == 8192
    assert kwargs["temperature"] == 0.0
    assert kwargs["model"] == "claude-sonnet-4-6"
