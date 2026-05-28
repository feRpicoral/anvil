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
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

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
    try:
        adapter_dir = Path(str(raw["adapter_dir"]))
        repo_id = str(raw["repo_id"])
        model_name = str(raw["model_name"])
        license_name = str(raw["license"])
        language = str(raw.get("language", "en"))
        task_name = str(raw["task_name"])
        task_description = str(raw["task_description"])
        training_data_description = str(raw["training_data_description"])
        training_framework = str(raw["training_framework"])
        training_config_path = Path(str(raw["training_config_path"]))
    except KeyError as exc:
        raise ValueError(f"{path}: missing required key {exc}") from exc

    eval_comparison_path = (
        Path(str(raw["eval_comparison_path"])) if "eval_comparison_path" in raw else None
    )
    cost_report_path = Path(str(raw["cost_report_path"])) if "cost_report_path" in raw else None
    sources = tuple(str(s) for s in raw.get("sources", []))
    tags = tuple(str(t) for t in raw.get("tags", []))
    private = bool(raw.get("private", False))
    return PublishConfig(
        adapter_dir=adapter_dir,
        repo_id=repo_id,
        model_name=model_name,
        license=license_name,
        language=language,
        task_name=task_name,
        task_description=task_description,
        training_data_description=training_data_description,
        training_framework=training_framework,
        training_config_path=training_config_path,
        eval_comparison_path=eval_comparison_path,
        cost_report_path=cost_report_path,
        sources=sources,
        tags=tags,
        private=private,
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    variants = payload.get("variants", [])
    rows: list[EvalSummaryRow] = []
    for variant in variants:
        field_scores = variant.get("field_scores", {}) or {}
        macro = (
            sum(float(v) for v in field_scores.values()) / len(field_scores)
            if field_scores
            else 0.0
        )
        rows.append(
            EvalSummaryRow(
                variant=str(variant.get("variant", "")),
                json_validity_rate=float(variant.get("json_validity_rate", 0.0)),
                macro_f1=macro,
            )
        )
    return tuple(rows)


def _load_cost_summary(path: Path) -> CostSummary:
    payload = json.loads(path.read_text(encoding="utf-8"))
    training = payload.get("training_cost", {})
    fine_tuned = payload.get("fine_tuned", {})
    breakeven = payload.get("breakeven", {})
    return CostSummary(
        training_total_usd=float(training.get("total_usd", 0.0)),
        self_hosted_per_1m_tokens=float(fine_tuned.get("usd_per_1m_tokens", 0.0)),
        primary_api_label=str(breakeven.get("primary_api_label", "")),
        primary_api_per_1m_tokens=float(breakeven.get("primary_api_usd_per_1m", 0.0)),
        breakeven_monthly_m_tokens=float(breakeven.get("monthly_volume_m_tokens", 0.0)),
        breakeven_months_horizon=int(breakeven.get("months_horizon", 0)),
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
    print(f"publish: uploaded → {url}", file=sys.stderr)
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


if __name__ == "__main__":
    raise SystemExit(main())
