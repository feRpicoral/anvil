from __future__ import annotations

import math
from typing import Any

import pytest

from anvil.cost.training_cost import TrainingCost


def test_total_usd_sums_components() -> None:
    cost = TrainingCost(
        gpu_hours=2.0,
        gpu_hourly_usd=0.69,
        synthesis_api_cost_usd=54.0,
        eval_api_cost_usd=6.0,
    )

    assert cost.gpu_cost_usd == pytest.approx(1.38)
    assert cost.total_usd == pytest.approx(1.38 + 54.0 + 6.0)


def test_zero_components_are_legal() -> None:
    cost = TrainingCost(
        gpu_hours=0.0,
        gpu_hourly_usd=0.0,
        synthesis_api_cost_usd=0.0,
        eval_api_cost_usd=0.0,
    )

    assert cost.total_usd == 0.0


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("gpu_hours", -0.1),
        ("gpu_hourly_usd", -0.1),
        ("synthesis_api_cost_usd", -0.01),
        ("eval_api_cost_usd", -0.01),
        ("gpu_hours", math.inf),
        ("synthesis_api_cost_usd", math.nan),
    ],
)
def test_rejects_invalid_field_values(field_name: str, bad_value: float) -> None:
    kwargs: dict[str, float] = {
        "gpu_hours": 1.0,
        "gpu_hourly_usd": 0.69,
        "synthesis_api_cost_usd": 10.0,
        "eval_api_cost_usd": 5.0,
    }
    kwargs[field_name] = bad_value

    with pytest.raises(ValueError, match=field_name):
        TrainingCost(**kwargs)


@pytest.mark.parametrize("bad_value", ["4", True])
def test_rejects_wrong_field_types(bad_value: Any) -> None:
    kwargs: dict[str, Any] = {
        "gpu_hours": bad_value,
        "gpu_hourly_usd": 0.69,
        "synthesis_api_cost_usd": 10.0,
        "eval_api_cost_usd": 5.0,
    }

    with pytest.raises(ValueError, match="gpu_hours"):
        TrainingCost(**kwargs)


def test_to_dict_contains_components_and_derived_totals() -> None:
    cost = TrainingCost(
        gpu_hours=4.0,
        gpu_hourly_usd=0.69,
        synthesis_api_cost_usd=54.0,
        eval_api_cost_usd=6.0,
    )

    payload = cost.to_dict()

    assert payload["gpu_hours"] == 4.0
    assert payload["gpu_cost_usd"] == pytest.approx(2.76)
    assert payload["total_usd"] == pytest.approx(2.76 + 60.0)
