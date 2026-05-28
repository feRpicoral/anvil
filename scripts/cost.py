"""Chain training, inference, and breakeven into one cost report.

Reads a TOML config that pins the training-run metadata (GPU hours +
synthesis/eval API spend), the fine-tuned model's throughput assumption
(with a cited published source so the number isn't pulled from thin air),
and the API baselines to compare against. Optionally folds in eval
spend reported in an `eval/comparison.json` so the cost numbers match
the exact variants the README shows.

Output: `results/cost/comparison.json`, the single payload the chart
pipeline consumes.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
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
        output_path = _required_path(raw, "output_path", path)
        training = _required_table(raw, "training", path)
        fine_tuned = _required_table(raw, "fine_tuned", path)
        api_keys = _required_str_tuple(raw, "api_keys", path)
    except KeyError as exc:
        raise ValueError(f"{path}: missing required key {exc}") from exc

    if not api_keys:
        raise ValueError(f"{path}: 'api_keys' must be a non-empty list")

    training_tier_key = _required_str(training, "gpu_tier_key", path, "training.gpu_tier_key")
    if training_tier_key not in GPU_TIERS:
        raise ValueError(f"{path}: unknown training gpu_tier_key {training_tier_key!r}")

    fine_tuned_tier_key = _required_str(fine_tuned, "gpu_tier_key", path, "fine_tuned.gpu_tier_key")
    if fine_tuned_tier_key not in GPU_TIERS:
        raise ValueError(f"{path}: unknown fine_tuned gpu_tier_key {fine_tuned_tier_key!r}")

    for key in api_keys:
        if key not in API_PRICING:
            raise ValueError(f"{path}: unknown API pricing key {key!r}")

    return CostConfig(
        output_path=output_path,
        training_gpu_hours=_required_number(training, "gpu_hours", path, "training.gpu_hours"),
        training_gpu_tier_key=training_tier_key,
        synthesis_api_cost_usd=_optional_number(
            training, "synthesis_api_cost_usd", path, "training.synthesis_api_cost_usd", 0.0
        ),
        eval_api_cost_usd=_optional_number(
            training, "eval_api_cost_usd", path, "training.eval_api_cost_usd", 0.0
        ),
        fine_tuned_label=_required_str(fine_tuned, "label", path, "fine_tuned.label"),
        fine_tuned_gpu_tier_key=fine_tuned_tier_key,
        fine_tuned_throughput_tps=_required_number(
            fine_tuned, "throughput_tps", path, "fine_tuned.throughput_tps"
        ),
        fine_tuned_throughput_source=_required_str(
            fine_tuned, "throughput_source", path, "fine_tuned.throughput_source"
        ),
        fine_tuned_utilization=_optional_number(
            fine_tuned, "utilization", path, "fine_tuned.utilization", 1.0
        ),
        api_keys=api_keys,
        input_share=_optional_number(raw, "input_share", path, "input_share", 0.5),
        months_horizon=_optional_int(raw, "months_horizon", path, "months_horizon", 12),
        eval_comparison_path=_optional_path(raw, "eval_comparison_path", path),
    )


def fold_eval_api_cost(eval_comparison_path: Path) -> float:
    payload = json.loads(eval_comparison_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{eval_comparison_path}: comparison payload must be an object")
    variants = payload.get("variants", [])
    if not isinstance(variants, list):
        raise ValueError(f"{eval_comparison_path}: variants must be a list")

    total = 0.0
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError(f"{eval_comparison_path}: variants[{index}] must be an object")
        total += _non_negative_finite(
            variant.get("total_cost_usd", 0.0),
            f"{eval_comparison_path}: variants[{index}].total_cost_usd",
        )
    return total


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
        f"-> {config.output_path}",
        file=sys.stderr,
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the cost-comparison JSON for the README.")
    parser.add_argument("--config", type=Path, required=True, help="TOML cost config.")
    return parser.parse_args(argv)


def main() -> int:
    return run(parse_args())


def _required_table(mapping: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    if key not in mapping:
        raise ValueError(f"{path}: missing required key {key}")
    value = mapping[key]
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {key!r} must be a table")
    return value


def _required_str(
    mapping: dict[str, Any], key: str, path: Path, display_key: str | None = None
) -> str:
    label = display_key or key
    if key not in mapping:
        raise ValueError(f"{path}: missing required key {label}")
    value = mapping[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}: {label} must be a non-empty string")
    return value


def _required_str_tuple(mapping: dict[str, Any], key: str, path: Path) -> tuple[str, ...]:
    if key not in mapping:
        raise ValueError(f"{path}: missing required key {key}")
    value = mapping[key]
    if not isinstance(value, list):
        raise ValueError(f"{path}: {key!r} must be a list of non-empty strings")
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ValueError(f"{path}: {key}[{index}] must be a non-empty string")
        items.append(item)
    return tuple(items)


def _required_path(mapping: dict[str, Any], key: str, path: Path) -> Path:
    return Path(_required_str(mapping, key, path))


def _optional_path(mapping: dict[str, Any], key: str, path: Path) -> Path | None:
    if key not in mapping:
        return None
    return _required_path(mapping, key, path)


def _required_number(
    mapping: dict[str, Any], key: str, path: Path, display_key: str | None = None
) -> float:
    if key not in mapping:
        raise ValueError(f"{path}: missing required key {display_key or key}")
    return _non_negative_finite(mapping[key], f"{path}: {display_key or key}")


def _optional_number(
    mapping: dict[str, Any], key: str, path: Path, display_key: str, default: float
) -> float:
    if key not in mapping:
        return default
    return _required_number(mapping, key, path, display_key)


def _optional_int(
    mapping: dict[str, Any], key: str, path: Path, display_key: str, default: int
) -> int:
    if key not in mapping:
        return default
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}: {display_key} must be an integer")
    if value < 1:
        raise ValueError(f"{path}: {display_key} must be an integer >= 1")
    return value


def _non_negative_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a non-negative finite number")
    if value < 0:
        raise ValueError(f"{label} must be a non-negative finite number")
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
