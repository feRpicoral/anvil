"""Chain training, inference, and breakeven into one cost report.

Reads a TOML config that pins the training-run metadata (GPU hours +
synthesis/eval API spend), the fine-tuned model's throughput assumption
(with a cited published source so the number isn't pulled from thin air),
and the API baselines to compare against. Optionally folds in eval
spend reported in an `eval/comparison.json` so the cost numbers match
the exact variants the README shows.

Output: `results/cost/comparison.json` — the single payload the chart
pipeline consumes.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from anvil.cost.breakeven import cumulative_cost_curve, monthly_breakeven_m_tokens
from anvil.cost.inference_cost import (
    API_PRICING,
    GPU_TIERS,
    build_self_hosted,
    compare,
)
from anvil.cost.training_cost import TrainingCost


@dataclasses.dataclass(frozen=True)
class CostConfig:
    output_path: Path
    training_gpu_hours: float
    training_gpu_tier_key: str
    synthesis_api_cost_usd: float
    eval_api_cost_usd: float
    fine_tuned_label: str
    fine_tuned_gpu_tier_key: str
    fine_tuned_throughput_tps: float
    fine_tuned_throughput_source: str
    fine_tuned_utilization: float
    api_keys: tuple[str, ...]
    input_share: float
    months_horizon: int
    eval_comparison_path: Path | None


def load_cost_config(path: Path) -> CostConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    try:
        output_path = Path(str(raw["output_path"]))
        training = raw["training"]
        fine_tuned = raw["fine_tuned"]
        api_keys = tuple(str(k) for k in raw["api_keys"])
    except KeyError as exc:
        raise ValueError(f"{path}: missing required key {exc}") from exc

    if not isinstance(training, dict) or not isinstance(fine_tuned, dict):
        raise ValueError(f"{path}: 'training' and 'fine_tuned' must be tables")
    if not api_keys:
        raise ValueError(f"{path}: 'api_keys' must be a non-empty list")

    training_tier_key = str(training["gpu_tier_key"])
    if training_tier_key not in GPU_TIERS:
        raise ValueError(f"{path}: unknown training gpu_tier_key {training_tier_key!r}")

    fine_tuned_tier_key = str(fine_tuned["gpu_tier_key"])
    if fine_tuned_tier_key not in GPU_TIERS:
        raise ValueError(f"{path}: unknown fine_tuned gpu_tier_key {fine_tuned_tier_key!r}")

    for key in api_keys:
        if key not in API_PRICING:
            raise ValueError(f"{path}: unknown API pricing key {key!r}")

    return CostConfig(
        output_path=output_path,
        training_gpu_hours=float(training["gpu_hours"]),
        training_gpu_tier_key=training_tier_key,
        synthesis_api_cost_usd=float(training.get("synthesis_api_cost_usd", 0.0)),
        eval_api_cost_usd=float(training.get("eval_api_cost_usd", 0.0)),
        fine_tuned_label=str(fine_tuned["label"]),
        fine_tuned_gpu_tier_key=fine_tuned_tier_key,
        fine_tuned_throughput_tps=float(fine_tuned["throughput_tps"]),
        fine_tuned_throughput_source=str(fine_tuned["throughput_source"]),
        fine_tuned_utilization=float(fine_tuned.get("utilization", 1.0)),
        api_keys=api_keys,
        input_share=float(raw.get("input_share", 0.5)),
        months_horizon=int(raw.get("months_horizon", 12)),
        eval_comparison_path=Path(str(raw["eval_comparison_path"]))
        if "eval_comparison_path" in raw
        else None,
    )


def fold_eval_api_cost(eval_comparison_path: Path) -> float:
    """Sum `total_cost_usd` across the variants in an eval comparison JSON.

    Lets the cost report match the EXACT spend the eval reported, instead of
    relying on the operator to copy the number into the TOML.
    """
    payload = json.loads(eval_comparison_path.read_text(encoding="utf-8"))
    variants = payload.get("variants", [])
    return sum(float(v.get("total_cost_usd", 0.0)) for v in variants)


def build_report(config: CostConfig) -> dict[str, Any]:
    eval_api_cost = config.eval_api_cost_usd
    if config.eval_comparison_path is not None:
        eval_api_cost += fold_eval_api_cost(config.eval_comparison_path)

    training_tier = GPU_TIERS[config.training_gpu_tier_key]
    training = TrainingCost(
        gpu_hours=config.training_gpu_hours,
        gpu_hourly_usd=training_tier.hourly_usd,
        synthesis_api_cost_usd=config.synthesis_api_cost_usd,
        eval_api_cost_usd=eval_api_cost,
    )

    self_hosted = build_self_hosted(
        label=config.fine_tuned_label,
        gpu_tier_key=config.fine_tuned_gpu_tier_key,
        sustained_throughput_tps=config.fine_tuned_throughput_tps,
        utilization=config.fine_tuned_utilization,
    )
    comparison = compare([self_hosted], list(config.api_keys), input_share=config.input_share)

    primary_api = comparison.api[0]
    breakeven_volume = monthly_breakeven_m_tokens(
        training_cost_usd=training.total_usd,
        fine_tuned_usd_per_1m=self_hosted.usd_per_1m_tokens,
        api_usd_per_1m=primary_api.usd_per_1m_tokens,
        months_horizon=config.months_horizon,
    )
    curve = cumulative_cost_curve(
        training_cost_usd=training.total_usd,
        fine_tuned_usd_per_1m=self_hosted.usd_per_1m_tokens,
        api_usd_per_1m=primary_api.usd_per_1m_tokens,
        monthly_volume_m_tokens=breakeven_volume,
        months_horizon=config.months_horizon,
    )

    return {
        "training_cost": training.to_dict(),
        "inference_cost": comparison.to_dict(),
        "fine_tuned": {
            "label": self_hosted.label,
            "gpu_tier_key": self_hosted.gpu_tier_key,
            "sustained_throughput_tps": self_hosted.sustained_throughput_tps,
            "throughput_source": config.fine_tuned_throughput_source,
            "utilization": self_hosted.utilization,
            "usd_per_1m_tokens": self_hosted.usd_per_1m_tokens,
        },
        "breakeven": {
            "primary_api_label": primary_api.label,
            "primary_api_usd_per_1m": primary_api.usd_per_1m_tokens,
            "months_horizon": config.months_horizon,
            "monthly_volume_m_tokens": breakeven_volume,
            "curve": [dataclasses.asdict(p) for p in curve],
        },
    }


def write_report(output_path: Path, report: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    config = load_cost_config(args.config)
    report = build_report(config)
    write_report(config.output_path, report)
    print(
        f"cost: total_training=${report['training_cost']['total_usd']:.2f} "
        f"self_hosted_per_1m=${report['fine_tuned']['usd_per_1m_tokens']:.4f} "
        f"breakeven={report['breakeven']['monthly_volume_m_tokens']:.2f}M tokens/month "
        f"→ {config.output_path}",
        file=sys.stderr,
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the cost-comparison JSON for the README.")
    parser.add_argument("--config", type=Path, required=True, help="TOML cost config.")
    return parser.parse_args(argv)


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
