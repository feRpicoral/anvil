from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from scripts.publish import (
    build_card_data,
    load_publish_config,
    run,
    write_readme,
)


def _write_training_config(tmp_path: Path) -> Path:
    path = tmp_path / "train.toml"
    path.write_text(
        """
        base_model = "meta-llama/Llama-3.1-8B-Instruct"
        backend = "unsloth"
        output_dir = "outputs/full"
        train_jsonl = "data/full/train.jsonl"
        rank = 16
        alpha = 32
        epochs = 3
        learning_rate = 2.0e-4
        max_seq_len = 2048
        quantization = "nf4"
        """,
        encoding="utf-8",
    )
    return path


def _toml_value(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return str(value)


def _write_publish_config(tmp_path: Path, **overrides: Any) -> Path:
    train_path = _write_training_config(tmp_path)
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    defaults: dict[str, Any] = {
        "adapter_dir": str(adapter_dir),
        "repo_id": "acme/contracts",
        "model_name": "anvil-llama31-8b",
        "license": "llama3.1",
        "language": "en",
        "task_name": "Contract field extraction",
        "task_description": "Extracts the eight critical fields.",
        "training_data_description": "Synthetic NDAs/MSAs + CUAD slice.",
        "training_framework": "Unsloth + TRL",
        "training_config_path": str(train_path),
    }
    defaults.update(overrides)

    lines = [f"{key} = {_toml_value(value)}" for key, value in defaults.items()]
    path = tmp_path / "publish.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_eval_comparison(tmp_path: Path, *, finetuned_validity: float = 1.0) -> Path:
    payload = {
        "n_cases": 3,
        "variants": [
            {
                "variant": "base",
                "n_cases": 3,
                "json_validity_rate": 0.33,
                "field_scores": {"parties": 0.5, "term": 0.5},
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost_usd": 0.0,
                "mean_latency_ms": 0.0,
            },
            {
                "variant": "finetuned",
                "n_cases": 3,
                "json_validity_rate": finetuned_validity,
                "field_scores": {"parties": 0.9, "term": 0.85, "governing_law": 0.95},
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_cost_usd": 0.0,
                "mean_latency_ms": 0.0,
            },
        ],
    }
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_cost_report(tmp_path: Path) -> Path:
    payload = {
        "training_cost": {"total_usd": 56.76, "gpu_cost_usd": 2.76},
        "fine_tuned": {
            "label": "Llama 3.1 8B QLoRA",
            "usd_per_1m_tokens": 0.22,
        },
        "breakeven": {
            "primary_api_label": "GPT-4o",
            "primary_api_usd_per_1m": 6.25,
            "months_horizon": 12,
            "monthly_volume_m_tokens": 1.6,
        },
    }
    path = tmp_path / "cost.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_json_payload(tmp_path: Path, name: str, payload: object) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_publish_config_parses_required(tmp_path: Path) -> None:
    config_path = _write_publish_config(tmp_path)

    config = load_publish_config(config_path)

    assert config.repo_id == "acme/contracts"
    assert config.model_name == "anvil-llama31-8b"
    assert config.eval_comparison_path is None
    assert config.cost_report_path is None
    assert config.private is False


def test_load_publish_config_parses_optional_lists_and_private_flag(tmp_path: Path) -> None:
    config_path = _write_publish_config(
        tmp_path,
        sources=["https://example.test/source"],
        tags=["legal", "qlora"],
        private=True,
    )

    config = load_publish_config(config_path)

    assert config.sources == ("https://example.test/source",)
    assert config.tags == ("legal", "qlora")
    assert config.private is True


def test_load_publish_config_rejects_missing_required(tmp_path: Path) -> None:
    path = tmp_path / "publish.toml"
    path.write_text('repo_id = "acme/x"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="missing required key"):
        load_publish_config(path)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"adapter_dir": 123}, "adapter_dir"),
        ({"private": "false"}, "private"),
        ({"sources": "https://example.test/source"}, "sources"),
        ({"sources": [123]}, r"sources\[0\]"),
        ({"tags": "legal"}, "tags"),
        ({"tags": [""]}, r"tags\[0\]"),
    ],
)
def test_load_publish_config_rejects_malformed_fields(
    tmp_path: Path,
    overrides: dict[str, Any],
    match: str,
) -> None:
    config_path = _write_publish_config(tmp_path, **overrides)

    with pytest.raises(ValueError, match=match):
        load_publish_config(config_path)


def test_build_card_data_without_eval_or_cost(tmp_path: Path) -> None:
    config_path = _write_publish_config(tmp_path)
    config = load_publish_config(config_path)

    card = build_card_data(config)

    assert card.eval_summary == ()
    assert card.cost is None
    assert card.lora_rank == 16


def test_build_card_data_with_eval_includes_finetuned_row(tmp_path: Path) -> None:
    eval_path = _write_eval_comparison(tmp_path, finetuned_validity=0.99)
    config_path = _write_publish_config(tmp_path, eval_comparison_path=str(eval_path))
    config = load_publish_config(config_path)

    card = build_card_data(config)

    variants = {row.variant: row for row in card.eval_summary}
    assert "finetuned" in variants
    assert variants["finetuned"].json_validity_rate == pytest.approx(0.99)
    assert variants["finetuned"].macro_f1 == pytest.approx((0.9 + 0.85 + 0.95) / 3)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ([], "comparison payload must be an object"),
        ({"variants": {}}, "variants must be a list"),
        ({"variants": [1]}, r"variants\[0\] must be an object"),
        (
            {"variants": [{"variant": "base", "json_validity_rate": 1.0, "field_scores": []}]},
            r"variants\[0\]\.field_scores",
        ),
        (
            {"variants": [{"variant": "base", "json_validity_rate": "1.0", "field_scores": {}}]},
            r"variants\[0\]\.json_validity_rate",
        ),
        (
            {
                "variants": [
                    {
                        "variant": "base",
                        "json_validity_rate": 1.0,
                        "field_scores": {"parties": 1.2},
                    }
                ]
            },
            r"variants\[0\]\.field_scores\.parties",
        ),
    ],
)
def test_build_card_data_rejects_malformed_eval_summary(
    tmp_path: Path,
    payload: object,
    match: str,
) -> None:
    eval_path = _write_json_payload(tmp_path, "comparison.json", payload)
    config_path = _write_publish_config(tmp_path, eval_comparison_path=str(eval_path))
    config = load_publish_config(config_path)

    with pytest.raises(ValueError, match=match):
        build_card_data(config)


def test_build_card_data_with_cost(tmp_path: Path) -> None:
    cost_path = _write_cost_report(tmp_path)
    config_path = _write_publish_config(tmp_path, cost_report_path=str(cost_path))
    config = load_publish_config(config_path)

    card = build_card_data(config)

    assert card.cost is not None
    assert card.cost.training_total_usd == 56.76
    assert card.cost.primary_api_label == "GPT-4o"


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ([], "cost report payload must be an object"),
        ({"fine_tuned": {}, "breakeven": {}}, "training_cost"),
        ({"training_cost": [], "fine_tuned": {}, "breakeven": {}}, "training_cost"),
        (
            {
                "training_cost": {"total_usd": "56"},
                "fine_tuned": {"usd_per_1m_tokens": 0.22},
                "breakeven": {
                    "primary_api_label": "GPT-4o",
                    "primary_api_usd_per_1m": 6.25,
                    "months_horizon": 12,
                    "monthly_volume_m_tokens": 1.6,
                },
            },
            r"training_cost\.total_usd",
        ),
        (
            {
                "training_cost": {"total_usd": 56.76},
                "fine_tuned": {"usd_per_1m_tokens": 0.22},
                "breakeven": {
                    "primary_api_label": "GPT-4o",
                    "primary_api_usd_per_1m": 6.25,
                    "months_horizon": 0,
                    "monthly_volume_m_tokens": 1.6,
                },
            },
            r"breakeven\.months_horizon.*>= 1",
        ),
    ],
)
def test_build_card_data_rejects_malformed_cost_summary(
    tmp_path: Path,
    payload: object,
    match: str,
) -> None:
    cost_path = _write_json_payload(tmp_path, "cost.json", payload)
    config_path = _write_publish_config(tmp_path, cost_report_path=str(cost_path))
    config = load_publish_config(config_path)

    with pytest.raises(ValueError, match=match):
        build_card_data(config)


def test_write_readme_creates_file_in_adapter_dir(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()

    path = write_readme(adapter_dir, "# Hello")

    assert path == adapter_dir / "README.md"
    assert path.read_text(encoding="utf-8") == "# Hello"


def test_write_readme_rejects_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="adapter_dir"):
        write_readme(tmp_path / "missing", "# Hello")


def test_run_without_confirm_writes_readme_but_does_not_upload(tmp_path: Path) -> None:
    config_path = _write_publish_config(tmp_path)
    args = argparse.Namespace(config=config_path, confirm=False)

    with patch("scripts.publish.upload_adapter") as upload:
        rc = run(args)

    assert rc == 0
    assert (tmp_path / "adapter" / "README.md").exists()
    upload.assert_not_called()


def test_run_with_confirm_uploads(tmp_path: Path) -> None:
    config_path = _write_publish_config(tmp_path)
    args = argparse.Namespace(config=config_path, confirm=True)

    with patch("scripts.publish.upload_adapter") as upload:
        upload.return_value = "https://huggingface.co/acme/contracts/commit/abc"
        rc = run(args)

    assert rc == 0
    upload.assert_called_once()
    call_kwargs = upload.call_args.kwargs
    assert call_kwargs["repo_id"] == "acme/contracts"
    assert call_kwargs["adapter_dir"] == tmp_path / "adapter"
