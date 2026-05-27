"""OpenAI Structured Outputs backend for `StructuredGenerator`.

Uses `response_format={"type": "json_schema", ..., "strict": true}` so the
model is constrained to emit a payload that matches our wrapper schema.
A schema-conformant response always carries both `contract_text` and
`extraction`; we still validate to fail loudly if a future API change
relaxes the guarantee.
"""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params import ResponseFormatJSONSchema
from openai.types.shared_params.response_format_json_schema import JSONSchema

from anvil.data.pricing import OPENAI_PRICES, compute_cost_usd
from anvil.data.prompts import ContractType
from anvil.data.synthesis import (
    GenerationResult,
    synthesis_response_schema,
    validate_generation_payload,
)

_DEFAULT_MODEL = "gpt-4o-2024-08-06"


class OpenAIGenerator:
    """Async GPT-4o synthesis backend with strict structured outputs."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = _DEFAULT_MODEL,
        client: AsyncOpenAI | None = None,
    ) -> None:
        if model not in OPENAI_PRICES:
            raise ValueError(f"unknown OpenAI model for pricing: {model!r}")
        self._model = model
        self._client = client if client is not None else AsyncOpenAI(api_key=api_key)

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
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(role="system", content=system_prompt),
            ChatCompletionUserMessageParam(role="user", content=user_prompt),
        ]
        schema = synthesis_response_schema()
        json_schema: JSONSchema = {
            "name": schema["name"],
            "strict": schema["strict"],
            "schema": schema["schema"],
        }
        response_format = ResponseFormatJSONSchema(type="json_schema", json_schema=json_schema)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            response_format=response_format,
            seed=seed,
        )
        choice = response.choices[0]
        refusal = getattr(choice.message, "refusal", None)
        if refusal:
            raise RuntimeError(f"OpenAI refused generation: {refusal}")
        content = choice.message.content or ""
        payload = _parse_payload(content)
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cost = compute_cost_usd(OPENAI_PRICES[self._model], input_tokens, output_tokens)
        return GenerationResult(
            contract_type=contract_type,
            contract_text=payload["contract_text"],
            extraction=payload["extraction"],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            backend=f"openai:{self._model}",
        )


def _parse_payload(content: str) -> dict[str, Any]:
    parsed: Any = json.loads(content)
    return validate_generation_payload(parsed, "OpenAI")
