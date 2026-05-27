from __future__ import annotations

import pytest

from anvil.data.pricing import ANTHROPIC_PRICES, OPENAI_PRICES, compute_cost_usd


def test_compute_cost_usd_zero_tokens_is_zero() -> None:
    assert compute_cost_usd((2.5, 10.0), 0, 0) == 0.0


def test_compute_cost_usd_input_only() -> None:
    assert compute_cost_usd((2.5, 10.0), 1_000_000, 0) == pytest.approx(2.5)


def test_compute_cost_usd_output_only() -> None:
    assert compute_cost_usd((2.5, 10.0), 0, 1_000_000) == pytest.approx(10.0)


def test_compute_cost_usd_combined() -> None:
    cost = compute_cost_usd((2.5, 10.0), 500_000, 500_000)

    assert cost == pytest.approx(2.5 * 0.5 + 10.0 * 0.5)


def test_compute_cost_usd_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        compute_cost_usd((2.5, 10.0), -1, 0)
    with pytest.raises(ValueError, match="non-negative"):
        compute_cost_usd((2.5, 10.0), 0, -1)


def test_openai_pricing_table_has_gpt4o_snapshot() -> None:
    assert OPENAI_PRICES["gpt-4o-2024-08-06"] == (2.50, 10.00)
    assert OPENAI_PRICES["gpt-4o-mini"] == (0.15, 0.60)


def test_anthropic_pricing_table_has_sonnet_and_haiku() -> None:
    assert ANTHROPIC_PRICES["claude-sonnet-4-6"] == (3.00, 15.00)
    assert ANTHROPIC_PRICES["claude-haiku-4-5"] == (1.00, 5.00)
