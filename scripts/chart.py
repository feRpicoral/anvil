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
import math
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from anvil.cost.breakeven import BreakevenPoint
from anvil.cost.inference_cost import CostComparison, CostScenario

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from anvil.plots.charts import LossSample

_TRAINING_LOSS_PNG = "training-loss.png"
_TASK_METRIC_PNG = "task-metric-comparison.png"
_VALIDITY_PNG = "json-validity.png"
_COST_PER_1M_PNG = "cost-per-1m.png"
_BREAKEVEN_PNG = "breakeven.png"


def _configure_matplotlib() -> None:
    cache_root = Path(tempfile.gettempdir()) / "anvil-matplotlib"
    mplconfigdir = cache_root / "mpl"
    xdg_cache_home = cache_root / "xdg"
    for cache_dir in (mplconfigdir, xdg_cache_home):
        cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mplconfigdir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_home))

    import matplotlib

    matplotlib.use("Agg")


def load_loss_history(path: Path) -> list[LossSample]:
    _configure_matplotlib()
    from anvil.plots.charts import LossSample

    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"{path}: loss history must be a JSON array")
    history: list[LossSample] = []
    for index, item in enumerate(payload):
        row = _object(item, f"{path}: loss_history[{index}]")
        history.append(
            LossSample(
                step=_required_non_negative_int(row, "step", f"{path}: loss_history[{index}]"),
                train_loss=_required_non_negative_finite(
                    row, "train_loss", f"{path}: loss_history[{index}]"
                ),
                val_loss=_optional_non_negative_finite(
                    row, "val_loss", f"{path}: loss_history[{index}]"
                ),
            )
        )
    return history


def load_validity_and_field_scores(
    path: Path,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    payload = _object(_read_json(path), f"{path}: comparison payload")
    variants = _optional_array(payload, "variants", f"{path}: comparison payload")
    validity: dict[str, float] = {}
    field_scores: dict[str, dict[str, float]] = {}
    for index, item in enumerate(variants):
        variant = _object(item, f"{path}: variants[{index}]")
        name = _required_non_empty_str(variant, "variant", f"{path}: variants[{index}]")
        if name in validity:
            raise ValueError(f"{path}: duplicate variant {name!r}")
        validity[name] = _required_rate(variant, "json_validity_rate", f"{path}: variants[{index}]")
        scores = _optional_object(variant, "field_scores", f"{path}: variants[{index}]")
        field_scores[name] = {
            _non_empty_str(key, f"{path}: variants[{index}].field_scores key"): _rate(
                value, f"{path}: variants[{index}].field_scores[{key!r}]"
            )
            for key, value in scores.items()
        }
    return validity, field_scores


def load_cost_payload(
    path: Path,
) -> tuple[CostComparison, list[BreakevenPoint], float, str]:
    payload = _object(_read_json(path), f"{path}: cost payload")
    inference = _optional_object(payload, "inference_cost", f"{path}: cost payload")
    self_hosted = _cost_scenarios(inference, "self_hosted", f"{path}: inference_cost")
    api = _cost_scenarios(inference, "api", f"{path}: inference_cost")
    comparison = CostComparison(
        self_hosted=self_hosted,
        api=api,
        input_share=_optional_rate(inference, "input_share", f"{path}: inference_cost", 0.5),
        notes=_optional_str_list(inference, "notes", f"{path}: inference_cost"),
    )
    breakeven = _optional_object(payload, "breakeven", f"{path}: cost payload")
    curve_rows = _optional_array(breakeven, "curve", f"{path}: breakeven")
    curve: list[BreakevenPoint] = []
    for index, item in enumerate(curve_rows):
        point = _object(item, f"{path}: breakeven.curve[{index}]")
        curve.append(
            BreakevenPoint(
                month=_required_non_negative_int(
                    point, "month", f"{path}: breakeven.curve[{index}]"
                ),
                cumulative_finetuned_usd=_required_non_negative_finite(
                    point, "cumulative_finetuned_usd", f"{path}: breakeven.curve[{index}]"
                ),
                cumulative_api_usd=_required_non_negative_finite(
                    point, "cumulative_api_usd", f"{path}: breakeven.curve[{index}]"
                ),
            )
        )
    monthly_volume = (
        _required_non_negative_finite(breakeven, "monthly_volume_m_tokens", f"{path}: breakeven")
        if curve
        else 0.0
    )
    primary_api_label = _optional_non_empty_str(
        breakeven, "primary_api_label", f"{path}: breakeven", "API"
    )
    return comparison, curve, monthly_volume, primary_api_label


def save_figure(fig: Figure, path: Path) -> Path:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def run(args: argparse.Namespace) -> int:
    _configure_matplotlib()
    from anvil.plots.charts import (
        breakeven_curve,
        cost_per_1m_tokens,
        json_validity_rate,
        task_metric_comparison,
        training_loss_curve,
    )
    from anvil.plots.style import apply_style

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
    print(f"chart: wrote {len(written)} chart(s) ({rendered}) -> {output_dir}", file=sys.stderr)
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


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _optional_object(mapping: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    if key not in mapping:
        return {}
    return _object(mapping[key], f"{label}.{key}")


def _optional_array(mapping: dict[str, Any], key: str, label: str) -> list[Any]:
    if key not in mapping:
        return []
    value = mapping[key]
    if not isinstance(value, list):
        raise ValueError(f"{label}.{key} must be a JSON array")
    return value


def _required(mapping: dict[str, Any], key: str, label: str) -> Any:
    try:
        return mapping[key]
    except KeyError as exc:
        raise ValueError(f"{label}.{key} is required") from exc


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _non_empty_str(value: Any, label: str) -> str:
    text = _string(value, label)
    if not text:
        raise ValueError(f"{label} must be a non-empty string")
    return text


def _required_non_empty_str(mapping: dict[str, Any], key: str, label: str) -> str:
    return _non_empty_str(_required(mapping, key, label), f"{label}.{key}")


def _optional_non_empty_str(
    mapping: dict[str, Any],
    key: str,
    label: str,
    default: str,
) -> str:
    if key not in mapping:
        return default
    return _non_empty_str(mapping[key], f"{label}.{key}")


def _optional_str_list(mapping: dict[str, Any], key: str, label: str) -> list[str]:
    if key not in mapping:
        return []
    values = mapping[key]
    if not isinstance(values, list):
        raise ValueError(f"{label}.{key} must be a JSON array")
    return [_string(item, f"{label}.{key}[{index}]") for index, item in enumerate(values)]


def _required_non_negative_int(mapping: dict[str, Any], key: str, label: str) -> int:
    value = _required(mapping, key, label)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label}.{key} must be a non-negative integer")
    return value


def _non_negative_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a non-negative finite number")
    if value < 0:
        raise ValueError(f"{label} must be a non-negative finite number")
    return float(value)


def _required_non_negative_finite(mapping: dict[str, Any], key: str, label: str) -> float:
    return _non_negative_finite(_required(mapping, key, label), f"{label}.{key}")


def _optional_non_negative_finite(
    mapping: dict[str, Any],
    key: str,
    label: str,
) -> float | None:
    if key not in mapping or mapping[key] is None:
        return None
    return _non_negative_finite(mapping[key], f"{label}.{key}")


def _rate(value: Any, label: str) -> float:
    rate = _non_negative_finite(value, label)
    if rate > 1.0:
        raise ValueError(f"{label} must be between 0 and 1")
    return rate


def _required_rate(mapping: dict[str, Any], key: str, label: str) -> float:
    return _rate(_required(mapping, key, label), f"{label}.{key}")


def _optional_rate(mapping: dict[str, Any], key: str, label: str, default: float) -> float:
    if key not in mapping:
        return default
    return _rate(mapping[key], f"{label}.{key}")


def _cost_scenarios(mapping: dict[str, Any], key: str, label: str) -> list[CostScenario]:
    scenarios: list[CostScenario] = []
    for index, item in enumerate(_optional_array(mapping, key, label)):
        row = _object(item, f"{label}.{key}[{index}]")
        scenarios.append(
            CostScenario(
                label=_required_non_empty_str(row, "label", f"{label}.{key}[{index}]"),
                usd_per_1m_tokens=_required_non_negative_finite(
                    row, "usd_per_1m_tokens", f"{label}.{key}[{index}]"
                ),
                notes=_string(row.get("notes", ""), f"{label}.{key}[{index}].notes"),
            )
        )
    return scenarios


if __name__ == "__main__":
    raise SystemExit(main())
