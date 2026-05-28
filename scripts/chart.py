"""Regenerate the five canonical Anvil charts from JSON results.

Reads:
  - eval comparison JSON (per-variant validity + field scores)
  - cost report JSON (training cost + inference comparison + breakeven curve)
  - optional training loss history JSON (a list of {step, train_loss, val_loss})

Writes five PNGs into `--output-dir`: training-loss, task-metric-comparison,
json-validity, cost-per-1m, breakeven. Skips any chart whose source JSON is
missing so a partial run still produces what's available.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from anvil.cost.breakeven import BreakevenPoint
from anvil.cost.inference_cost import CostComparison, CostScenario
from anvil.plots.charts import (
    LossSample,
    breakeven_curve,
    cost_per_1m_tokens,
    json_validity_rate,
    task_metric_comparison,
    training_loss_curve,
)
from anvil.plots.style import apply_style

_TRAINING_LOSS_PNG = "training-loss.png"
_TASK_METRIC_PNG = "task-metric-comparison.png"
_VALIDITY_PNG = "json-validity.png"
_COST_PER_1M_PNG = "cost-per-1m.png"
_BREAKEVEN_PNG = "breakeven.png"


def load_loss_history(path: Path) -> list[LossSample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path}: loss history must be a JSON array")
    return [
        LossSample(
            step=int(row["step"]),
            train_loss=float(row["train_loss"]),
            val_loss=float(row["val_loss"]) if row.get("val_loss") is not None else None,
        )
        for row in payload
    ]


def load_validity_and_field_scores(
    path: Path,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    variants = payload.get("variants", [])
    validity: dict[str, float] = {}
    field_scores: dict[str, dict[str, float]] = {}
    for variant in variants:
        name = str(variant["variant"])
        validity[name] = float(variant.get("json_validity_rate", 0.0))
        field_scores[name] = {
            str(k): float(v) for k, v in (variant.get("field_scores", {}) or {}).items()
        }
    return validity, field_scores


def load_cost_payload(
    path: Path,
) -> tuple[CostComparison, list[BreakevenPoint], float, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    inference = payload.get("inference_cost", {})
    self_hosted = [
        CostScenario(
            label=str(row["label"]),
            usd_per_1m_tokens=float(row["usd_per_1m_tokens"]),
            notes=str(row.get("notes", "")),
        )
        for row in inference.get("self_hosted", [])
    ]
    api = [
        CostScenario(
            label=str(row["label"]),
            usd_per_1m_tokens=float(row["usd_per_1m_tokens"]),
            notes=str(row.get("notes", "")),
        )
        for row in inference.get("api", [])
    ]
    comparison = CostComparison(
        self_hosted=self_hosted,
        api=api,
        input_share=float(inference.get("input_share", 0.5)),
        notes=[str(n) for n in inference.get("notes", [])],
    )
    breakeven = payload.get("breakeven", {})
    curve = [
        BreakevenPoint(
            month=int(p["month"]),
            cumulative_finetuned_usd=float(p["cumulative_finetuned_usd"]),
            cumulative_api_usd=float(p["cumulative_api_usd"]),
        )
        for p in breakeven.get("curve", [])
    ]
    monthly_volume = float(breakeven.get("monthly_volume_m_tokens", 0.0))
    primary_api_label = str(breakeven.get("primary_api_label", "API"))
    return comparison, curve, monthly_volume, primary_api_label


def save_figure(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def run(args: argparse.Namespace) -> int:
    apply_style()
    written: list[Path] = []
    output_dir = args.output_dir

    if args.eval_comparison is not None and args.eval_comparison.exists():
        validity, field_scores = load_validity_and_field_scores(args.eval_comparison)
        if validity:
            written.append(save_figure(json_validity_rate(validity), output_dir / _VALIDITY_PNG))
        if any(field_scores.values()):
            non_empty = {k: v for k, v in field_scores.items() if v}
            written.append(
                save_figure(task_metric_comparison(non_empty), output_dir / _TASK_METRIC_PNG)
            )

    if args.cost_report is not None and args.cost_report.exists():
        comparison, curve, monthly_volume, primary_api_label = load_cost_payload(args.cost_report)
        if comparison.self_hosted or comparison.api:
            written.append(
                save_figure(cost_per_1m_tokens(comparison), output_dir / _COST_PER_1M_PNG)
            )
        if curve:
            written.append(
                save_figure(
                    breakeven_curve(
                        curve,
                        monthly_volume_m_tokens=monthly_volume,
                        primary_api_label=primary_api_label,
                    ),
                    output_dir / _BREAKEVEN_PNG,
                )
            )

    if args.loss_history is not None and args.loss_history.exists():
        history = load_loss_history(args.loss_history)
        if history:
            written.append(
                save_figure(training_loss_curve(history), output_dir / _TRAINING_LOSS_PNG)
            )

    if not written:
        print("chart: nothing to draw (no inputs found)", file=sys.stderr)
        return 0
    rendered = ", ".join(p.name for p in written)
    print(f"chart: wrote {len(written)} chart(s) ({rendered}) → {output_dir}", file=sys.stderr)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate the canonical Anvil charts.")
    parser.add_argument(
        "--eval-comparison",
        type=Path,
        default=Path("results/eval/smoke/comparison.json"),
        help="Eval comparison JSON (per-variant validity + field scores).",
    )
    parser.add_argument(
        "--cost-report",
        type=Path,
        default=Path("results/cost/smoke.json"),
        help="Cost report JSON (inference comparison + breakeven).",
    )
    parser.add_argument(
        "--loss-history",
        type=Path,
        default=None,
        help="Optional training loss history JSON ([{step, train_loss, val_loss?}, ...]).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/charts"),
        help="Where to write the PNGs.",
    )
    return parser.parse_args(argv)


def main() -> int:
    return run(parse_args())


_ = (BreakevenPoint, CostScenario, Any)  # re-export anchors for the type checker


if __name__ == "__main__":
    raise SystemExit(main())
