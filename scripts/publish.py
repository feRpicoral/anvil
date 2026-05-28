"""Render a model card, write README.md into the adapter dir, push to HF Hub.

Reads run metadata from a TOML config that points at the training config,
eval comparison JSON, and cost report JSON. Renders the model card via
`anvil.publish.model_card.render` and writes it as `README.md` next to
the adapter. With `--confirm`, also pushes the folder to the configured
repo via `anvil.publish.upload.upload_adapter`. Without `--confirm`, the
script stops after writing the README so the operator can review it.
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
from typing import Any, cast

from anvil.publish.model_card import (
    CostSummary,
    EvalSummaryRow,
    ModelCardData,
    render,
)
from anvil.publish.upload import upload_adapter
from anvil.training.qlora import load_config as load_training_config


@dataclasses.dataclass(frozen=True)
class PublishConfig:
    adapter_dir: Path
    repo_id: str
    model_name: str
    license: str
    language: str
    task_name: str
    task_description: str
    training_data_description: str
    training_framework: str
    training_config_path: Path
    eval_comparison_path: Path | None
    cost_report_path: Path | None
    sources: tuple[str, ...]
    tags: tuple[str, ...]
    private: bool


def load_publish_config(path: Path) -> PublishConfig:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return PublishConfig(
        adapter_dir=_required_path(raw, "adapter_dir", path),
        repo_id=_required_str(raw, "repo_id", path),
        model_name=_required_str(raw, "model_name", path),
        license=_required_str(raw, "license", path),
        language=_optional_str(raw, "language", path, "en"),
        task_name=_required_str(raw, "task_name", path),
        task_description=_required_str(raw, "task_description", path),
        training_data_description=_required_str(raw, "training_data_description", path),
        training_framework=_required_str(raw, "training_framework", path),
        training_config_path=_required_path(raw, "training_config_path", path),
        eval_comparison_path=_optional_path(raw, "eval_comparison_path", path),
        cost_report_path=_optional_path(raw, "cost_report_path", path),
        sources=_optional_str_tuple(raw, "sources", path),
        tags=_optional_str_tuple(raw, "tags", path),
        private=_optional_bool(raw, "private", path, False),
    )


def build_card_data(config: PublishConfig) -> ModelCardData:
    training = load_training_config(config.training_config_path)
    eval_summary = (
        _load_eval_summary(config.eval_comparison_path)
        if config.eval_comparison_path is not None
        else ()
    )
    cost = (
        _load_cost_summary(config.cost_report_path) if config.cost_report_path is not None else None
    )
    extra_kwargs: dict[str, object] = {}
    if config.tags:
        extra_kwargs["tags"] = config.tags
    return ModelCardData(
        model_name=config.model_name,
        base_model=training.base_model,
        license=config.license,
        language=config.language,
        task_name=config.task_name,
        task_description=config.task_description,
        training_data_description=config.training_data_description,
        training_framework=config.training_framework,
        quantization=training.quantization,
        lora_rank=training.rank,
        lora_alpha=training.alpha,
        epochs=training.epochs,
        learning_rate=training.learning_rate,
        max_seq_len=training.max_seq_len,
        eval_summary=eval_summary,
        cost=cost,
        sources=config.sources,
        **extra_kwargs,  # type: ignore[arg-type]
    )


def _load_eval_summary(path: Path) -> tuple[EvalSummaryRow, ...]:
    payload = _load_json_object(path, "comparison payload")
    variants = payload.get("variants", [])
    if not isinstance(variants, list):
        raise ValueError(f"{path}: variants must be a list")

    rows: list[EvalSummaryRow] = []
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError(f"{path}: variants[{index}] must be an object")
        field_scores = variant.get("field_scores", {})
        if not isinstance(field_scores, dict):
            raise ValueError(f"{path}: variants[{index}].field_scores must be an object")
        macro = (
            sum(
                _required_rate(
                    field_scores,
                    field,
                    path,
                    f"variants[{index}].field_scores.{field}",
                )
                for field in field_scores
            )
            / len(field_scores)
            if field_scores
            else 0.0
        )
        rows.append(
            EvalSummaryRow(
                variant=_required_str(variant, "variant", path, f"variants[{index}].variant"),
                json_validity_rate=_required_rate(
                    variant,
                    "json_validity_rate",
                    path,
                    f"variants[{index}].json_validity_rate",
                ),
                macro_f1=macro,
            )
        )
    return tuple(rows)


def _load_cost_summary(path: Path) -> CostSummary:
    payload = _load_json_object(path, "cost report payload")
    training = _required_object(payload, "training_cost", path)
    fine_tuned = _required_object(payload, "fine_tuned", path)
    breakeven = _required_object(payload, "breakeven", path)
    return CostSummary(
        training_total_usd=_required_number(training, "total_usd", path, "training_cost.total_usd"),
        self_hosted_per_1m_tokens=_required_number(
            fine_tuned,
            "usd_per_1m_tokens",
            path,
            "fine_tuned.usd_per_1m_tokens",
        ),
        primary_api_label=_required_str(
            breakeven,
            "primary_api_label",
            path,
            "breakeven.primary_api_label",
        ),
        primary_api_per_1m_tokens=_required_number(
            breakeven,
            "primary_api_usd_per_1m",
            path,
            "breakeven.primary_api_usd_per_1m",
        ),
        breakeven_monthly_m_tokens=_required_number(
            breakeven,
            "monthly_volume_m_tokens",
            path,
            "breakeven.monthly_volume_m_tokens",
        ),
        breakeven_months_horizon=_required_positive_int(
            breakeven,
            "months_horizon",
            path,
            "breakeven.months_horizon",
        ),
    )


def write_readme(adapter_dir: Path, card_markdown: str) -> Path:
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"adapter_dir not found: {adapter_dir}")
    path = adapter_dir / "README.md"
    path.write_text(card_markdown, encoding="utf-8")
    return path


def run(args: argparse.Namespace) -> int:
    config = load_publish_config(args.config)
    card = render(build_card_data(config))
    readme_path = write_readme(config.adapter_dir, card)
    print(f"publish: wrote {readme_path}", file=sys.stderr)
    if not args.confirm:
        print(
            f"publish: skipping upload. Re-invoke with --confirm to push to {config.repo_id}.",
            file=sys.stderr,
        )
        return 0
    url = upload_adapter(
        adapter_dir=config.adapter_dir,
        repo_id=config.repo_id,
        private=config.private,
    )
    print(f"publish: uploaded -> {url}", file=sys.stderr)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a LoRA adapter + model card to HF Hub.")
    parser.add_argument("--config", type=Path, required=True, help="TOML publish config.")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually push to HF Hub. Without this flag the script stops after writing README.md.",
    )
    return parser.parse_args(argv)


def main() -> int:
    return run(parse_args())


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: {label} must be an object")
    return cast(dict[str, Any], payload)


def _required_object(mapping: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    if key not in mapping:
        raise ValueError(f"{path}: missing required key {key}")
    value = mapping[key]
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {key} must be an object")
    return cast(dict[str, Any], value)


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


def _optional_str(mapping: dict[str, Any], key: str, path: Path, default: str) -> str:
    if key not in mapping:
        return default
    return _required_str(mapping, key, path)


def _required_path(mapping: dict[str, Any], key: str, path: Path) -> Path:
    return Path(_required_str(mapping, key, path))


def _optional_path(mapping: dict[str, Any], key: str, path: Path) -> Path | None:
    if key not in mapping:
        return None
    return _required_path(mapping, key, path)


def _optional_str_tuple(mapping: dict[str, Any], key: str, path: Path) -> tuple[str, ...]:
    if key not in mapping:
        return ()
    value = mapping[key]
    if not isinstance(value, list):
        raise ValueError(f"{path}: {key} must be a list of non-empty strings")
    items: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise ValueError(f"{path}: {key}[{index}] must be a non-empty string")
        items.append(item)
    return tuple(items)


def _optional_bool(mapping: dict[str, Any], key: str, path: Path, default: bool) -> bool:
    if key not in mapping:
        return default
    value = mapping[key]
    if not isinstance(value, bool):
        raise ValueError(f"{path}: {key} must be a boolean")
    return value


def _required_number(
    mapping: dict[str, Any], key: str, path: Path, display_key: str | None = None
) -> float:
    if key not in mapping:
        raise ValueError(f"{path}: missing required key {display_key or key}")
    return _non_negative_finite(mapping[key], f"{path}: {display_key or key}")


def _required_rate(
    mapping: dict[str, Any], key: str, path: Path, display_key: str | None = None
) -> float:
    label = f"{path}: {display_key or key}"
    value = _required_number(mapping, key, path, display_key)
    if value > 1.0:
        raise ValueError(f"{label} must be between 0 and 1")
    return value


def _required_positive_int(
    mapping: dict[str, Any], key: str, path: Path, display_key: str | None = None
) -> int:
    label = display_key or key
    if key not in mapping:
        raise ValueError(f"{path}: missing required key {label}")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}: {label} must be an integer")
    if value < 1:
        raise ValueError(f"{path}: {label} must be an integer >= 1")
    return value


def _non_negative_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a non-negative finite number")
    if value < 0:
        raise ValueError(f"{label} must be a non-negative finite number")
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
