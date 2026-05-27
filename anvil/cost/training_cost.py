"""One-time training cost: GPU rental + API spend on synthesis + eval.

`TrainingCost` is a pure data record with the three independent components
the run actually pays for, plus convenience properties. The Makefile-driven
runs (`make data-full`, `make train-full`, `make eval-smoke`/`make eval-full`)
emit numbers that feed this directly; nothing here loads or parses those
JSON files.

Pair with `anvil/cost/breakeven.py` to compute the monthly token volume at
which fine-tuning pays back vs. continuing to call a commercial API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingCost:
    """One-time costs for producing the fine-tuned model.

    Components stay separate so the breakdown can be surfaced in charts and
    the budget envelope can be audited against the actual spend.
    """

    gpu_hours: float
    gpu_hourly_usd: float
    synthesis_api_cost_usd: float
    eval_api_cost_usd: float

    def __post_init__(self) -> None:
        for name, value in (
            ("gpu_hours", self.gpu_hours),
            ("gpu_hourly_usd", self.gpu_hourly_usd),
            ("synthesis_api_cost_usd", self.synthesis_api_cost_usd),
            ("eval_api_cost_usd", self.eval_api_cost_usd),
        ):
            _validate_non_negative_finite(name, value)

    @property
    def gpu_cost_usd(self) -> float:
        return self.gpu_hours * self.gpu_hourly_usd

    @property
    def total_usd(self) -> float:
        return self.gpu_cost_usd + self.synthesis_api_cost_usd + self.eval_api_cost_usd

    def to_dict(self) -> dict[str, float]:
        return {
            "gpu_hours": self.gpu_hours,
            "gpu_hourly_usd": self.gpu_hourly_usd,
            "gpu_cost_usd": self.gpu_cost_usd,
            "synthesis_api_cost_usd": self.synthesis_api_cost_usd,
            "eval_api_cost_usd": self.eval_api_cost_usd,
            "total_usd": self.total_usd,
        }


def _validate_non_negative_finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a non-negative finite number, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative finite number, got {value}")
