from __future__ import annotations

from collections.abc import Callable, Iterator

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from anvil.cost.breakeven import BreakevenPoint, cumulative_cost_curve
from anvil.cost.inference_cost import CostComparison, CostScenario, build_self_hosted, compare
from anvil.plots.charts import (
    LossSample,
    breakeven_curve,
    cost_per_1m_tokens,
    json_validity_rate,
    task_metric_comparison,
    training_loss_curve,
)


@pytest.fixture(autouse=True)
def _restore_rcparams() -> Iterator[None]:
    with mpl.rc_context():
        yield


@pytest.fixture(autouse=True)
def _close_figs() -> Iterator[None]:
    yield
    plt.close("all")


def test_training_loss_curve_returns_figure_with_one_train_line() -> None:
    history = [
        LossSample(step=0, train_loss=2.0),
        LossSample(step=10, train_loss=1.5),
        LossSample(step=20, train_loss=1.2),
    ]

    fig = training_loss_curve(history)

    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    labels = [line.get_label() for line in ax.get_lines()]
    assert "train" in labels
    assert "val" not in labels


def test_training_loss_curve_adds_val_line_when_present() -> None:
    history = [
        LossSample(step=0, train_loss=2.0, val_loss=2.1),
        LossSample(step=10, train_loss=1.5, val_loss=1.6),
        LossSample(step=20, train_loss=1.2, val_loss=1.3),
    ]

    fig = training_loss_curve(history)

    labels = [line.get_label() for line in fig.axes[0].get_lines()]
    assert "train" in labels
    assert "val" in labels


def test_training_loss_curve_skips_val_when_only_some_samples_have_it() -> None:
    import numpy as np

    history = [
        LossSample(step=0, train_loss=2.0, val_loss=2.1),
        LossSample(step=10, train_loss=1.5),
        LossSample(step=20, train_loss=1.2, val_loss=1.3),
    ]

    fig = training_loss_curve(history)

    val_line = next(line for line in fig.axes[0].get_lines() if line.get_label() == "val")
    xs = np.asarray(val_line.get_xdata()).tolist()
    assert xs == [0.0, 20.0]


def test_training_loss_curve_rejects_empty_history() -> None:
    with pytest.raises(ValueError, match="history"):
        training_loss_curve([])


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: LossSample(step=-1, train_loss=1.0), "step"),
        (lambda: LossSample(step=True, train_loss=1.0), "step"),
        (lambda: LossSample(step=1, train_loss=-0.1), "train_loss"),
        (lambda: LossSample(step=1, train_loss=float("nan")), "train_loss"),
        (lambda: LossSample(step=1, train_loss=1.0, val_loss=float("inf")), "val_loss"),
    ],
)
def test_loss_sample_rejects_invalid_values(
    build: Callable[[], LossSample],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        build()


def test_task_metric_comparison_returns_one_bar_group_per_field() -> None:
    scores = {
        "base": {"parties": 0.3, "term": 0.5},
        "finetuned": {"parties": 0.85, "term": 0.8},
        "gpt-4o": {"parties": 0.9, "term": 0.85},
    }

    fig = task_metric_comparison(scores)

    ax = fig.axes[0]
    bars = ax.containers
    assert len(bars) == 3
    assert all(len(group) == 2 for group in bars)


def test_task_metric_comparison_uses_explicit_field_order_when_provided() -> None:
    scores = {
        "base": {"parties": 0.3, "term": 0.5, "governing_law": 0.4},
    }

    fig = task_metric_comparison(scores, fields=("term", "parties"))

    ax = fig.axes[0]
    labels = [label.get_text() for label in ax.get_xticklabels()]
    assert labels == ["term", "parties"]


def test_task_metric_comparison_handles_missing_field_as_zero() -> None:
    scores = {
        "base": {"parties": 0.5},
        "finetuned": {"parties": 0.9, "term": 0.7},
    }

    fig = task_metric_comparison(scores, fields=("parties", "term"))

    ax = fig.axes[0]
    base_group = next(c for c in ax.containers if c.get_label() == "base")
    term_height = base_group[1].get_height()
    assert term_height == 0.0


def test_task_metric_comparison_rejects_empty_variants() -> None:
    with pytest.raises(ValueError, match="per_variant_field_scores"):
        task_metric_comparison({})


def test_task_metric_comparison_rejects_no_fields_after_filter() -> None:
    with pytest.raises(ValueError, match="field score"):
        task_metric_comparison({"base": {}})


@pytest.mark.parametrize(
    ("scores", "match"),
    [
        ({"": {"parties": 0.5}}, "variant"),
        ({"base": {"": 0.5}}, "field"),
        ({"base": {"parties": -0.1}}, "base.parties"),
        ({"base": {"parties": 1.1}}, "base.parties"),
        ({"base": {"parties": float("nan")}}, "base.parties"),
    ],
)
def test_task_metric_comparison_rejects_invalid_scores(
    scores: dict[str, dict[str, float]],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        task_metric_comparison(scores)


def test_json_validity_rate_renders_one_bar_per_variant() -> None:
    rates = {"base": 0.3, "finetuned": 0.99, "gpt-4o": 1.0}

    fig = json_validity_rate(rates)

    ax = fig.axes[0]
    assert len(ax.containers) == 1
    bars = ax.containers[0]
    assert len(bars) == 3


def test_json_validity_rate_y_axis_capped_at_one() -> None:
    fig = json_validity_rate({"finetuned": 1.0})

    ax = fig.axes[0]
    assert ax.get_ylim()[1] >= 1.0
    assert ax.get_ylim()[0] == 0.0


def test_json_validity_rate_rejects_empty() -> None:
    with pytest.raises(ValueError, match="per_variant_rate"):
        json_validity_rate({})


@pytest.mark.parametrize(
    ("rates", "match"),
    [
        ({"": 0.5}, "variant"),
        ({"base": -0.1}, "base"),
        ({"base": 1.1}, "base"),
        ({"base": float("inf")}, "base"),
    ],
)
def test_json_validity_rate_rejects_invalid_rates(
    rates: dict[str, float],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        json_validity_rate(rates)


def test_cost_per_1m_tokens_renders_one_bar_per_scenario() -> None:
    sh = build_self_hosted(
        label="Llama 3.1 8B on A5000",
        gpu_tier_key="runpod-rtx-a5000-community",
        sustained_throughput_tps=2100.0,
    )
    comparison = compare([sh], ["gpt-4o", "claude-sonnet-4-6"])

    fig = cost_per_1m_tokens(comparison)

    ax = fig.axes[0]
    bars = ax.containers[0]
    assert len(bars) == 3


def test_cost_per_1m_tokens_rejects_empty() -> None:
    with pytest.raises(ValueError, match="comparison"):
        cost_per_1m_tokens(CostComparison(self_hosted=[], api=[], input_share=0.5))


@pytest.mark.parametrize(
    ("comparison", "match"),
    [
        (
            CostComparison(
                self_hosted=[CostScenario(label="", usd_per_1m_tokens=0.1)],
                api=[],
                input_share=0.5,
            ),
            "scenario",
        ),
        (
            CostComparison(
                self_hosted=[CostScenario(label="local", usd_per_1m_tokens=-0.1)],
                api=[],
                input_share=0.5,
            ),
            "local",
        ),
        (
            CostComparison(
                self_hosted=[CostScenario(label="local", usd_per_1m_tokens=float("nan"))],
                api=[],
                input_share=0.5,
            ),
            "local",
        ),
    ],
)
def test_cost_per_1m_tokens_rejects_invalid_rows(
    comparison: CostComparison,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        cost_per_1m_tokens(comparison)


def test_breakeven_curve_renders_two_lines() -> None:
    points = cumulative_cost_curve(
        training_cost_usd=120.0,
        fine_tuned_usd_per_1m=0.10,
        api_usd_per_1m=6.0,
        monthly_volume_m_tokens=2.0,
        months_horizon=6,
    )

    fig = breakeven_curve(points, monthly_volume_m_tokens=2.0, primary_api_label="GPT-4o")

    ax = fig.axes[0]
    labels = [line.get_label() for line in ax.get_lines() if line.get_label() != "_nolegend_"]
    assert "fine-tuned (incl. training)" in labels
    assert "GPT-4o" in labels


def test_breakeven_curve_draws_crossover_marker_when_curves_cross() -> None:
    points = cumulative_cost_curve(
        training_cost_usd=10.0,
        fine_tuned_usd_per_1m=0.05,
        api_usd_per_1m=10.0,
        monthly_volume_m_tokens=2.0,
        months_horizon=6,
    )

    fig = breakeven_curve(points, monthly_volume_m_tokens=2.0)

    ax = fig.axes[0]
    vlines = [line for line in ax.get_lines() if line.get_linestyle() == ":"]
    assert vlines, "expected a dotted vertical breakeven line"


def test_breakeven_curve_rejects_empty_points() -> None:
    with pytest.raises(ValueError, match="points"):
        breakeven_curve([], monthly_volume_m_tokens=1.0)


def test_breakeven_curve_rejects_negative_volume() -> None:
    points = [BreakevenPoint(month=0, cumulative_finetuned_usd=10.0, cumulative_api_usd=0.0)]

    with pytest.raises(ValueError, match="monthly_volume_m_tokens"):
        breakeven_curve(points, monthly_volume_m_tokens=-1.0)


def test_breakeven_curve_rejects_non_finite_volume() -> None:
    points = [BreakevenPoint(month=0, cumulative_finetuned_usd=10.0, cumulative_api_usd=0.0)]

    with pytest.raises(ValueError, match="monthly_volume_m_tokens"):
        breakeven_curve(points, monthly_volume_m_tokens=float("nan"))


def test_breakeven_curve_rejects_empty_primary_api_label() -> None:
    points = [BreakevenPoint(month=0, cumulative_finetuned_usd=10.0, cumulative_api_usd=0.0)]

    with pytest.raises(ValueError, match="primary_api_label"):
        breakeven_curve(points, monthly_volume_m_tokens=1.0, primary_api_label="")


@pytest.mark.parametrize(
    ("points", "match"),
    [
        (
            [BreakevenPoint(month=-1, cumulative_finetuned_usd=10.0, cumulative_api_usd=0.0)],
            "month",
        ),
        (
            [BreakevenPoint(month=1, cumulative_finetuned_usd=-0.1, cumulative_api_usd=0.0)],
            "finetuned",
        ),
        (
            [
                BreakevenPoint(
                    month=1, cumulative_finetuned_usd=1.0, cumulative_api_usd=float("inf")
                )
            ],
            "api",
        ),
        (
            [
                BreakevenPoint(month=1, cumulative_finetuned_usd=10.0, cumulative_api_usd=1.0),
                BreakevenPoint(month=1, cumulative_finetuned_usd=11.0, cumulative_api_usd=2.0),
            ],
            "strictly increasing",
        ),
    ],
)
def test_breakeven_curve_rejects_invalid_points(
    points: list[BreakevenPoint],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        breakeven_curve(points, monthly_volume_m_tokens=1.0)
