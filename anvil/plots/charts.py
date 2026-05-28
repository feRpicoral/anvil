"""Five canonical charts that drive the impact-first README.

Each function takes typed inputs and returns a `matplotlib.figure.Figure`.
Callers save it (`fig.savefig(...)`); the chart pipeline (`scripts/chart.py`,
arriving next) wires them to JSON inputs and writes PNGs under `results/charts/`.

The functions don't apply the project stylesheet — call
`anvil.plots.style.apply_style()` once at the top of your script.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from anvil.cost.breakeven import BreakevenPoint
from anvil.cost.inference_cost import CostComparison
from anvil.plots.style import palette


@dataclass(frozen=True)
class LossSample:
    """One step of the training-loss curve."""

    step: int
    train_loss: float
    val_loss: float | None = None


def training_loss_curve(history: Sequence[LossSample]) -> Figure:
    """Train loss across steps with optional val-loss overlay."""
    if not history:
        raise ValueError("history must contain at least one sample")
    colors = palette()
    fig, ax = plt.subplots()
    steps = [sample.step for sample in history]
    ax.plot(
        steps,
        [sample.train_loss for sample in history],
        label="train",
        color=colors["train_loss"],
    )
    val_pairs: list[tuple[int, float]] = [
        (sample.step, sample.val_loss) for sample in history if sample.val_loss is not None
    ]
    if val_pairs:
        ax.plot(
            [step for step, _ in val_pairs],
            [loss for _, loss in val_pairs],
            label="val",
            color=colors["val_loss"],
            marker="o",
            linestyle="--",
        )
    ax.set_title("Training loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    return fig


def task_metric_comparison(
    per_variant_field_scores: dict[str, dict[str, float]],
    *,
    fields: Sequence[str] | None = None,
) -> Figure:
    """Grouped bar chart: per-field score for each variant."""
    if not per_variant_field_scores:
        raise ValueError("per_variant_field_scores must be non-empty")
    variants = list(per_variant_field_scores.keys())
    if fields is None:
        # Stable order: union of fields across variants, sorted.
        field_set: set[str] = set()
        for variant_scores in per_variant_field_scores.values():
            field_set.update(variant_scores.keys())
        fields = sorted(field_set)
    if not fields:
        raise ValueError("at least one field score is required")

    colors = palette()
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    width = 0.8 / len(variants)
    positions = list(range(len(fields)))
    for index, variant in enumerate(variants):
        bar_heights = [per_variant_field_scores[variant].get(field, 0.0) for field in fields]
        offset = (index - (len(variants) - 1) / 2) * width
        ax.bar(
            [p + offset for p in positions],
            bar_heights,
            width=width,
            label=variant,
            color=_variant_color(variant, colors),
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(list(fields), rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Per-field extraction score")
    ax.set_ylabel("Score")
    ax.legend()
    return fig


def json_validity_rate(per_variant_rate: dict[str, float]) -> Figure:
    """Single-row bar chart of JSON-validity rate per variant."""
    if not per_variant_rate:
        raise ValueError("per_variant_rate must be non-empty")
    colors = palette()
    variants = list(per_variant_rate.keys())
    rates = [per_variant_rate[v] for v in variants]
    fig, ax = plt.subplots()
    bar_colors = [_variant_color(v, colors) for v in variants]
    bars = ax.bar(variants, rates, color=bar_colors)
    for bar, rate in zip(bars, rates, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(rate + 0.02, 1.02),
            f"{rate:.0%}",
            ha="center",
            va="bottom",
        )
    ax.set_ylim(0.0, 1.05)
    ax.set_title("JSON-validity rate by variant")
    ax.set_ylabel("Rate")
    return fig


def cost_per_1m_tokens(comparison: CostComparison) -> Figure:
    """Horizontal bar chart of USD-per-1M-tokens for each scenario in `comparison`."""
    rows = [*comparison.self_hosted, *comparison.api]
    if not rows:
        raise ValueError("comparison must contain at least one scenario")
    colors = palette()
    labels = [row.label for row in rows]
    values = [row.usd_per_1m_tokens for row in rows]
    bar_colors = [
        colors["self_hosted"] if index < len(comparison.self_hosted) else colors["api_blended"]
        for index in range(len(rows))
    ]
    fig, ax = plt.subplots()
    bars = ax.barh(labels, values, color=bar_colors)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"  ${value:.4f}",
            va="center",
        )
    ax.invert_yaxis()
    ax.set_title("Cost per 1M tokens")
    ax.set_xlabel("USD per 1M tokens")
    return fig


def breakeven_curve(
    points: Sequence[BreakevenPoint],
    *,
    monthly_volume_m_tokens: float,
    primary_api_label: str = "API",
) -> Figure:
    """Two lines: cumulative cost month-by-month for fine-tuned vs. API."""
    if not points:
        raise ValueError("points must contain at least one BreakevenPoint")
    if monthly_volume_m_tokens < 0:
        raise ValueError("monthly_volume_m_tokens must be non-negative")
    colors = palette()
    months = [p.month for p in points]
    fig, ax = plt.subplots()
    ax.plot(
        months,
        [p.cumulative_finetuned_usd for p in points],
        label="fine-tuned (incl. training)",
        color=colors["self_hosted"],
        marker="o",
    )
    ax.plot(
        months,
        [p.cumulative_api_usd for p in points],
        label=primary_api_label,
        color=colors["api_blended"],
        marker="s",
        linestyle="--",
    )
    crossover = _first_crossover(points)
    if crossover is not None:
        ax.axvline(crossover, color=colors["muted"], linestyle=":", alpha=0.7)
        ax.text(
            crossover,
            max(p.cumulative_finetuned_usd for p in points) * 0.95,
            f"  breakeven @ month {crossover}",
            color=colors["muted"],
            va="top",
        )
    ax.set_title(f"Cumulative cost at {monthly_volume_m_tokens:.2f} M tokens/month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Cumulative USD")
    ax.legend()
    return fig


def _variant_color(variant: str, colors: dict[str, str]) -> str:
    key = variant.replace("-", "_").lower()
    if key in colors:
        return colors[key]
    if key.startswith("gpt"):
        return colors["gpt_4o"]
    if key.startswith("claude"):
        return colors["claude"]
    return colors["muted"]


def _first_crossover(points: Sequence[BreakevenPoint]) -> int | None:
    """Return the first month where api cumulative >= finetuned cumulative."""
    for point in points:
        if point.cumulative_api_usd >= point.cumulative_finetuned_usd:
            return point.month
    return None
