"""Anthropic Claude backend for `StructuredGenerator`.

Claude doesn't expose OpenAI-style strict structured outputs, so we steer
the model with the system prompt and validate the JSON object post-hoc.
The wrapper handles code-fenced responses (``` ... ```) and rejects
anything that doesn't decode into our expected `{contract_text, extraction}`
shape.
"""

from __future__ import annotations

import json
from typing import Any, cast

from anthropic import AsyncAnthropic

from anvil.data.pricing import ANTHROPIC_PRICES, compute_cost_usd
from anvil.data.prompts import ContractType
from anvil.data.synthesis import (
    GenerationResult,
    synthesis_response_schema,
    validate_generation_payload,
)

_DEFAULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_MAX_TOKENS = 8192
_DEFAULT_TEMPERATURE = 0.0


class AnthropicGenerator:
    """Async Claude synthesis backend with post-hoc JSON validation."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = _DEFAULT_TEMPERATURE,
        client: AsyncAnthropic | None = None,
    ) -> None:
        if model not in ANTHROPIC_PRICES:
            raise ValueError(f"unknown Anthropic model for pricing: {model!r}")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if not 0 <= temperature <= 1:
            raise ValueError("temperature must be between 0 and 1")
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = client if client is not None else AsyncAnthropic(api_key=api_key)

    @property
    def model(self) -> str:
        return self._model

    async def generate(
        self,
        contract_type: ContractType,
        system_prompt: str,
        user_prompt: str,
        seed: int,
    ) -> GenerationResult:
        # Claude does not expose a seed parameter.
        del seed
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=_system_prompt_with_schema(system_prompt),
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = _join_text_blocks(response.content)
        payload = _parse_payload(text)
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = compute_cost_usd(ANTHROPIC_PRICES[self._model], input_tokens, output_tokens)
        return GenerationResult(
            contract_type=contract_type,
            contract_text=payload["contract_text"],
            extraction=payload["extraction"],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            backend=f"anthropic:{self._model}",
        )


def _join_text_blocks(blocks: list[Any]) -> str:
    """Concatenate the textual content blocks returned by Claude."""
    parts: list[str] = []
    for block in blocks:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            parts.append(cast(str, getattr(block, "text", "")))
    if not parts:
        raise RuntimeError("Anthropic response contained no text blocks")
    return "".join(parts)


def _parse_payload(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of `text`, stripping code fences."""
    cleaned = _strip_code_fence(text.strip())
    parsed: Any = json.loads(cleaned)
    return validate_generation_payload(parsed, "Anthropic")


def _system_prompt_with_schema(system_prompt: str) -> str:
    schema = json.dumps(synthesis_response_schema()["schema"], ensure_ascii=False, sort_keys=True)
    schema_instruction = f"Use this JSON Schema for the full response object:\n{schema}"
    stripped = system_prompt.rstrip()
    if not stripped:
        return schema_instruction
    return f"{stripped}\n\n{schema_instruction}"


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    # Drop the opening fence (with optional language tag), then trim the trailing fence.
    without_open = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if without_open.rstrip().endswith("```"):
        return without_open.rstrip()[:-3].rstrip()
    return without_open
