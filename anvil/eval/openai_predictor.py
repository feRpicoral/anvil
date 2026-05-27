"""OpenAI GPT-4o extraction predictor for the three-way eval.

Implements the `ExtractionPredictor` Protocol against the OpenAI API with
strict JSON Schema enforcement. The system prompt asks for the structured
extraction only (no contract text echo); the user prompt carries the
contract text. `response_format` constrains the model to emit JSON that
conforms to `ContractExtraction`'s schema.

This is the baseline the fine-tuned model is compared against. Lives in
`anvil/eval/` rather than `anvil/data/` because extraction is the eval
task, not synthesis.
"""

from __future__ import annotations

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
from anvil.data.schema import contract_extraction_json_schema
from anvil.eval.runner import Prediction

_DEFAULT_MODEL = "gpt-4o-2024-08-06"
_RESPONSE_SCHEMA_NAME = "contract_extraction"

_SYSTEM_PROMPT = (
    "You extract structured contract fields from legal documents. "
    "Return a single JSON object that conforms exactly to the provided schema; "
    "use null where the contract does not specify a value. "
    "Do not include the contract text, explanations, or any prose outside the JSON object."
)


def extraction_response_schema() -> dict[str, Any]:
    """Wrap the strict `ContractExtraction` schema for OpenAI structured outputs."""
    schema = contract_extraction_json_schema()
    return {
        "name": _RESPONSE_SCHEMA_NAME,
        "strict": True,
        "schema": schema,
    }


class OpenAIExtractionPredictor:
    """Async GPT-4o predictor with strict JSON-schema-constrained outputs."""

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

    async def predict(self, contract_text: str) -> Prediction:
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(role="system", content=_SYSTEM_PROMPT),
            ChatCompletionUserMessageParam(role="user", content=contract_text),
        ]
        schema = extraction_response_schema()
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
        )
        choice = response.choices[0]
        refusal = getattr(choice.message, "refusal", None)
        if refusal:
            raise RuntimeError(f"OpenAI refused extraction: {refusal}")
        raw_output = choice.message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cost = compute_cost_usd(OPENAI_PRICES[self._model], input_tokens, output_tokens)
        return Prediction(
            raw_output=raw_output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
