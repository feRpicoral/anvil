"""Training-cost amortization vs. ongoing API spend.

`monthly_breakeven_tokens` answers the headline question: how many tokens
per month does the fine-tuned model need to process before its one-time
training cost is paid back relative to continuing to call a commercial
API? The math is intentionally trivial; the value is making the volume
visible alongside the README's cost chart.

`cumulative_cost_curve` returns the running totals (fine-tuned with
amortized training vs. API) for a list of monthly volumes — the data
the breakeven chart plots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BreakevenPoint:
    """One sample on the cumulative-cost curve."""

    month: int
    cumulative_finetuned_usd: float
    cumulative_api_usd: float


def monthly_breakeven_tokens(
    *,
    training_cost_usd: float,
    fine_tuned_usd_per_1m: float,
    api_usd_per_1m: float,
    months_horizon: int = 12,
) -> float:
    """Monthly token volume (millions) at which fine-tuning breaks even.

    Spreads `training_cost_usd` over `months_horizon` months and solves
    for the volume where amortized training + fine-tuned per-1M equals
    the API's per-1M.

        amortized_per_month = training_cost_usd / months_horizon
        amortized_per_month + X * fine_tuned_per_1m = X * api_per_1m
        X = amortized_per_month / (api_per_1m - fine_tuned_per_1m)
    """
    _validate_inputs(
        training_cost_usd=training_cost_usd,
        fine_tuned_usd_per_1m=fine_tuned_usd_per_1m,
        api_usd_per_1m=api_usd_per_1m,
        months_horizon=months_horizon,
    )
    if api_usd_per_1m <= fine_tuned_usd_per_1m:
        raise ValueError(
            "fine-tuned per-1M cost must be strictly less than api per-1M cost; "
            f"got fine_tuned={fine_tuned_usd_per_1m}, api={api_usd_per_1m}"
        )
    amortized = training_cost_usd / months_horizon
    return amortized / (api_usd_per_1m - fine_tuned_usd_per_1m)


def cumulative_cost_curve(
    *,
    training_cost_usd: float,
    fine_tuned_usd_per_1m: float,
    api_usd_per_1m: float,
    monthly_volume_m_tokens: float,
    months_horizon: int = 12,
) -> list[BreakevenPoint]:
    """Cumulative cost month-by-month for fine-tuned (incl. training) vs. API.

    Returns `months_horizon + 1` points covering month 0 (training paid up
    front, zero API spend) through month `months_horizon`. The chart code
    reads `cumulative_finetuned_usd` and `cumulative_api_usd` directly.
    """
    _validate_inputs(
        training_cost_usd=training_cost_usd,
        fine_tuned_usd_per_1m=fine_tuned_usd_per_1m,
        api_usd_per_1m=api_usd_per_1m,
        months_horizon=months_horizon,
    )
    if monthly_volume_m_tokens < 0 or not math.isfinite(monthly_volume_m_tokens):
        raise ValueError(
            f"monthly_volume_m_tokens must be a non-negative finite number, "
            f"got {monthly_volume_m_tokens}"
        )
    points: list[BreakevenPoint] = []
    for month in range(months_horizon + 1):
        finetuned = training_cost_usd + month * monthly_volume_m_tokens * fine_tuned_usd_per_1m
        api = month * monthly_volume_m_tokens * api_usd_per_1m
        points.append(
            BreakevenPoint(
                month=month,
                cumulative_finetuned_usd=finetuned,
                cumulative_api_usd=api,
            )
        )
    return points


def _validate_inputs(
    *,
    training_cost_usd: float,
    fine_tuned_usd_per_1m: float,
    api_usd_per_1m: float,
    months_horizon: int,
) -> None:
    if training_cost_usd < 0 or not math.isfinite(training_cost_usd):
        raise ValueError(
            f"training_cost_usd must be a non-negative finite number, got {training_cost_usd}"
        )
    if fine_tuned_usd_per_1m < 0 or not math.isfinite(fine_tuned_usd_per_1m):
        raise ValueError(
            f"fine_tuned_usd_per_1m must be a non-negative finite number, "
            f"got {fine_tuned_usd_per_1m}"
        )
    if api_usd_per_1m < 0 or not math.isfinite(api_usd_per_1m):
        raise ValueError(
            f"api_usd_per_1m must be a non-negative finite number, got {api_usd_per_1m}"
        )
    if months_horizon < 1:
        raise ValueError(f"months_horizon must be >= 1, got {months_horizon}")
