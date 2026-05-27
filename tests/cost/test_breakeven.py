from __future__ import annotations

import math
from typing import Any

import pytest

from anvil.cost.breakeven import (
    BreakevenPoint,
    cumulative_cost_curve,
    monthly_breakeven_m_tokens,
)


def test_breakeven_known_value() -> None:
    volume = monthly_breakeven_m_tokens(
        training_cost_usd=120.0,
        fine_tuned_usd_per_1m=0.10,
        api_usd_per_1m=6.0,
        months_horizon=12,
    )

    assert volume == pytest.approx(10.0 / 5.9, abs=1e-6)


def test_breakeven_scales_linearly_with_training_cost() -> None:
    half = monthly_breakeven_m_tokens(
        training_cost_usd=60.0, fine_tuned_usd_per_1m=0.10, api_usd_per_1m=6.0
    )
    full = monthly_breakeven_m_tokens(
        training_cost_usd=120.0, fine_tuned_usd_per_1m=0.10, api_usd_per_1m=6.0
    )

    assert full == pytest.approx(2.0 * half)


def test_breakeven_scales_inversely_with_horizon() -> None:
    short = monthly_breakeven_m_tokens(
        training_cost_usd=120.0,
        fine_tuned_usd_per_1m=0.10,
        api_usd_per_1m=6.0,
        months_horizon=6,
    )
    long = monthly_breakeven_m_tokens(
        training_cost_usd=120.0,
        fine_tuned_usd_per_1m=0.10,
        api_usd_per_1m=6.0,
        months_horizon=12,
    )

    assert short == pytest.approx(2.0 * long)


def test_breakeven_widens_with_smaller_api_premium() -> None:
    big_premium = monthly_breakeven_m_tokens(
        training_cost_usd=120.0, fine_tuned_usd_per_1m=0.10, api_usd_per_1m=10.0
    )
    small_premium = monthly_breakeven_m_tokens(
        training_cost_usd=120.0, fine_tuned_usd_per_1m=0.10, api_usd_per_1m=1.0
    )

    assert small_premium > big_premium


def test_breakeven_rejects_api_cost_not_above_fine_tuned() -> None:
    with pytest.raises(ValueError, match="strictly less"):
        monthly_breakeven_m_tokens(
            training_cost_usd=120.0,
            fine_tuned_usd_per_1m=10.0,
            api_usd_per_1m=10.0,
        )


def test_breakeven_rejects_zero_horizon() -> None:
    with pytest.raises(ValueError, match="months_horizon"):
        monthly_breakeven_m_tokens(
            training_cost_usd=120.0,
            fine_tuned_usd_per_1m=0.10,
            api_usd_per_1m=6.0,
            months_horizon=0,
        )


def test_breakeven_rejects_non_integer_horizon() -> None:
    months_horizon: Any = 1.5

    with pytest.raises(ValueError, match="months_horizon"):
        monthly_breakeven_m_tokens(
            training_cost_usd=120.0,
            fine_tuned_usd_per_1m=0.10,
            api_usd_per_1m=6.0,
            months_horizon=months_horizon,
        )


def test_breakeven_rejects_negative_training_cost() -> None:
    with pytest.raises(ValueError, match="training_cost_usd"):
        monthly_breakeven_m_tokens(
            training_cost_usd=-1.0, fine_tuned_usd_per_1m=0.10, api_usd_per_1m=6.0
        )


def test_breakeven_rejects_negative_fine_tuned_cost() -> None:
    with pytest.raises(ValueError, match="fine_tuned_usd_per_1m"):
        monthly_breakeven_m_tokens(
            training_cost_usd=120.0, fine_tuned_usd_per_1m=-0.01, api_usd_per_1m=6.0
        )


def test_breakeven_rejects_non_finite_api_cost() -> None:
    with pytest.raises(ValueError, match="api_usd_per_1m"):
        monthly_breakeven_m_tokens(
            training_cost_usd=120.0, fine_tuned_usd_per_1m=0.10, api_usd_per_1m=math.inf
        )


def test_breakeven_rejects_nan_training_cost() -> None:
    with pytest.raises(ValueError, match="training_cost_usd"):
        monthly_breakeven_m_tokens(
            training_cost_usd=math.nan, fine_tuned_usd_per_1m=0.10, api_usd_per_1m=6.0
        )


def test_breakeven_rejects_wrong_numeric_type() -> None:
    training_cost_usd: Any = "120"

    with pytest.raises(ValueError, match="training_cost_usd"):
        monthly_breakeven_m_tokens(
            training_cost_usd=training_cost_usd,
            fine_tuned_usd_per_1m=0.10,
            api_usd_per_1m=6.0,
        )


def test_cumulative_cost_curve_starts_at_training_cost_for_finetuned() -> None:
    curve = cumulative_cost_curve(
        training_cost_usd=120.0,
        fine_tuned_usd_per_1m=0.10,
        api_usd_per_1m=6.0,
        monthly_volume_m_tokens=2.0,
        months_horizon=6,
    )

    assert curve[0] == BreakevenPoint(
        month=0,
        cumulative_finetuned_usd=120.0,
        cumulative_api_usd=0.0,
    )


def test_cumulative_cost_curve_length_matches_horizon_plus_one() -> None:
    curve = cumulative_cost_curve(
        training_cost_usd=120.0,
        fine_tuned_usd_per_1m=0.10,
        api_usd_per_1m=6.0,
        monthly_volume_m_tokens=2.0,
        months_horizon=6,
    )

    assert [p.month for p in curve] == [0, 1, 2, 3, 4, 5, 6]


def test_cumulative_cost_curve_at_breakeven_volume_crosses_at_horizon() -> None:
    volume = monthly_breakeven_m_tokens(
        training_cost_usd=120.0,
        fine_tuned_usd_per_1m=0.10,
        api_usd_per_1m=6.0,
        months_horizon=6,
    )
    curve = cumulative_cost_curve(
        training_cost_usd=120.0,
        fine_tuned_usd_per_1m=0.10,
        api_usd_per_1m=6.0,
        monthly_volume_m_tokens=volume,
        months_horizon=6,
    )
    final = curve[-1]

    assert final.cumulative_finetuned_usd == pytest.approx(final.cumulative_api_usd)


def test_cumulative_cost_curve_below_breakeven_stays_finetuned_more_expensive() -> None:
    volume = monthly_breakeven_m_tokens(
        training_cost_usd=120.0,
        fine_tuned_usd_per_1m=0.10,
        api_usd_per_1m=6.0,
        months_horizon=6,
    )
    curve = cumulative_cost_curve(
        training_cost_usd=120.0,
        fine_tuned_usd_per_1m=0.10,
        api_usd_per_1m=6.0,
        monthly_volume_m_tokens=volume / 2,
        months_horizon=6,
    )
    final = curve[-1]

    assert final.cumulative_finetuned_usd > final.cumulative_api_usd


def test_cumulative_cost_curve_rejects_negative_volume() -> None:
    with pytest.raises(ValueError, match="monthly_volume_m_tokens"):
        cumulative_cost_curve(
            training_cost_usd=120.0,
            fine_tuned_usd_per_1m=0.10,
            api_usd_per_1m=6.0,
            monthly_volume_m_tokens=-1.0,
        )


def test_cumulative_cost_curve_rejects_wrong_volume_type() -> None:
    monthly_volume_m_tokens: Any = "2"

    with pytest.raises(ValueError, match="monthly_volume_m_tokens"):
        cumulative_cost_curve(
            training_cost_usd=120.0,
            fine_tuned_usd_per_1m=0.10,
            api_usd_per_1m=6.0,
            monthly_volume_m_tokens=monthly_volume_m_tokens,
        )
