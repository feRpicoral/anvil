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
from anvil.data.synthesis import GenerationResult

_DEFAULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_MAX_TOKENS = 4096


class AnthropicGenerator:
    """Async Claude synthesis backend with post-hoc JSON validation."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        client: AsyncAnthropic | None = None,
    ) -> None:
        if model not in ANTHROPIC_PRICES:
            raise ValueError(f"unknown Anthropic model for pricing: {model!r}")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self._model = model
        self._max_tokens = max_tokens
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
        # Claude doesn't expose a seed; the kwarg stays in the signature for
        # Protocol parity. Determinism on Claude is achieved by temperature=0
        # plus stable prompts, both of which apply by default here.
        del seed
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
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
    if not isinstance(parsed, dict):
        raise RuntimeError("Anthropic response is not a JSON object")
    if "contract_text" not in parsed or "extraction" not in parsed:
        raise RuntimeError("Anthropic response missing contract_text or extraction")
    if not isinstance(parsed["contract_text"], str):
        raise RuntimeError("Anthropic response contract_text is not a string")
    if not isinstance(parsed["extraction"], dict):
        raise RuntimeError("Anthropic response extraction is not an object")
    return parsed


def _strip_code_fence(text: str) -> str:
    if not text.startswith("```"):
        return text
    # Drop the opening fence (with optional language tag), then trim the trailing fence.
    without_open = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if without_open.rstrip().endswith("```"):
        return without_open.rstrip()[:-3].rstrip()
    return without_open
