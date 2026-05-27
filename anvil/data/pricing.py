"""API pricing constants for synthesis and eval baselines.

Prices are quoted as `(input_usd_per_million, output_usd_per_million)` tuples
in May 2026 USD. Edit here when OpenAI/Anthropic publish new rates; do not
duplicate the numbers in generator code.
"""

from __future__ import annotations

from typing import Final

PricePair = tuple[float, float]

OPENAI_PRICES: Final[dict[str, PricePair]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4.1": (2.00, 8.00),
}

ANTHROPIC_PRICES: Final[dict[str, PricePair]] = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def compute_cost_usd(prices: PricePair, input_tokens: int, output_tokens: int) -> float:
    """USD cost for a call given `(input_per_M, output_per_M)` and token counts."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be non-negative")
    input_per_m, output_per_m = prices
    return (input_tokens / 1_000_000) * input_per_m + (output_tokens / 1_000_000) * output_per_m
