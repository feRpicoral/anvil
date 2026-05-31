"""Tests for the pure helpers in `anvil.training.trl_trainer`.

The lazy-imported \"real\" trainer path is only exercised by `make train-smoke`
after `constraints/train.txt` is installed, so CI verifies the helpers and
the import-error message — not the trainer invocation itself.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from anvil.training.qlora import TrainingConfig
from anvil.training.trl_trainer import (
    build_chat_formatting_func,
    build_lora_kwargs,
    build_sft_kwargs,
    build_wandb_env,
    ensure_wandb_available,
    load_messages_jsonl,
)


class _Tokenizer:
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        return "\n".join(f"{m['role']}: {m['content']}" for m in conversation)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def _config(**overrides: object) -> TrainingConfig:
    defaults: dict[str, object] = {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "backend": "trl",
        "output_dir": Path("outputs/smoke"),
        "train_jsonl": Path("data/smoke/train.jsonl"),
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)  # type: ignore[arg-type]


def test_load_messages_jsonl_reads_rows(tmp_path: Path) -> None:
    rows = [
        {"messages": [{"role": "user", "content": "a"}]},
        {"messages": [{"role": "user", "content": "b"}]},
    ]
    path = _write_jsonl(tmp_path / "train.jsonl", rows)

    loaded = load_messages_jsonl(path)

    assert loaded == rows


def test_load_messages_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "train.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"messages": [{"role": "user", "content": "a"}]}\n\n'
        '{"messages": [{"role": "user", "content": "b"}]}\n',
        encoding="utf-8",
    )

    loaded = load_messages_jsonl(path)

    assert len(loaded) == 2


def test_load_messages_jsonl_rejects_non_object_row(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[1, 2, 3]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not a JSON object"):
        load_messages_jsonl(path)


def test_load_messages_jsonl_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"messages": [}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid JSON"):
        load_messages_jsonl(path)


def test_load_messages_jsonl_rejects_missing_messages(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "bad.jsonl", [{"content": "no messages key"}])

    with pytest.raises(ValueError, match="messages"):
        load_messages_jsonl(path)


def test_load_messages_jsonl_rejects_invalid_message_shape(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "bad.jsonl", [{"messages": [{"role": "user"}]}])

    with pytest.raises(ValueError, match="content"):
        load_messages_jsonl(path)


def test_load_messages_jsonl_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no records"):
        load_messages_jsonl(path)


def test_build_lora_kwargs_maps_core_fields() -> None:
    config = _config(rank=8, alpha=16, lora_dropout=0.1, target_modules="qkv")

    kwargs = build_lora_kwargs(config)

    assert kwargs["r"] == 8
    assert kwargs["lora_alpha"] == 16
    assert kwargs["lora_dropout"] == 0.1
    assert kwargs["target_modules"] == ["q_proj", "k_proj", "v_proj"]
    assert kwargs["bias"] == "none"
    assert kwargs["task_type"] == "CAUSAL_LM"


def test_chat_formatting_func_renders_single_example() -> None:
    formatting_func = build_chat_formatting_func(_Tokenizer())

    rendered = formatting_func({"messages": [{"role": "user", "content": "hello"}]})

    assert rendered == ["user: hello"]


def test_chat_formatting_func_renders_batch() -> None:
    formatting_func = build_chat_formatting_func(_Tokenizer())

    rendered = formatting_func(
        {
            "messages": [
                [{"role": "user", "content": "a"}],
                [{"role": "assistant", "content": "b"}],
            ]
        }
    )

    assert rendered == ["user: a", "assistant: b"]


def test_chat_formatting_func_rejects_missing_messages() -> None:
    formatting_func = build_chat_formatting_func(_Tokenizer())

    with pytest.raises(ValueError, match="messages"):
        formatting_func({})


def test_build_lora_kwargs_expands_all_linear_to_seven_modules() -> None:
    kwargs = build_lora_kwargs(_config(target_modules="all_linear"))

    assert len(kwargs["target_modules"]) == 7
    assert "gate_proj" in kwargs["target_modules"]
    assert "down_proj" in kwargs["target_modules"]


def test_build_sft_kwargs_maps_optimizer_fields() -> None:
    config = _config(
        learning_rate=5e-5,
        epochs=2,
        batch_size=2,
        grad_accum=4,
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_seq_len=1024,
        seed=42,
    )

    kwargs = build_sft_kwargs(config)

    assert kwargs["num_train_epochs"] == 2
    assert kwargs["per_device_train_batch_size"] == 2
    assert kwargs["gradient_accumulation_steps"] == 4
    assert kwargs["learning_rate"] == 5e-5
    assert kwargs["warmup_ratio"] == 0.05
    assert kwargs["weight_decay"] == 0.01
    assert kwargs["max_length"] == 1024
    assert kwargs["seed"] == 42


def test_build_sft_kwargs_disables_eval_when_no_val_set() -> None:
    kwargs = build_sft_kwargs(_config(val_jsonl=None, eval_strategy="epoch"))

    assert kwargs["eval_strategy"] == "no"


def test_build_sft_kwargs_propagates_eval_strategy_when_val_set() -> None:
    kwargs = build_sft_kwargs(
        _config(val_jsonl=Path("data/smoke/val.jsonl"), eval_strategy="epoch")
    )

    assert kwargs["eval_strategy"] == "epoch"


def test_build_sft_kwargs_carries_steps_when_strategy_is_steps() -> None:
    kwargs = build_sft_kwargs(
        _config(
            val_jsonl=Path("data/smoke/val.jsonl"),
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
        )
    )

    assert kwargs["eval_steps"] == 50
    assert kwargs["save_steps"] == 100


def test_build_sft_kwargs_loads_best_model_when_requested() -> None:
    kwargs = build_sft_kwargs(_config(val_jsonl=Path("data/smoke/val.jsonl")))

    assert kwargs["load_best_model_at_end"] is True
    assert kwargs["metric_for_best_model"] == "eval_loss"
    assert kwargs["greater_is_better"] is False


def test_build_sft_kwargs_skips_best_model_without_eval() -> None:
    kwargs = build_sft_kwargs(_config(val_jsonl=None, keep_best_only=True))

    assert "load_best_model_at_end" not in kwargs


def test_build_sft_kwargs_rejects_best_model_strategy_mismatch() -> None:
    config = _config(
        val_jsonl=Path("data/smoke/val.jsonl"),
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=100,
    )

    with pytest.raises(ValueError, match="save_strategy"):
        build_sft_kwargs(config)


def test_build_sft_kwargs_omits_eval_steps_when_eval_disabled_without_val() -> None:
    kwargs = build_sft_kwargs(_config(eval_strategy="steps", eval_steps=50, val_jsonl=None))

    assert kwargs["eval_strategy"] == "no"
    assert "eval_steps" not in kwargs


def test_build_sft_kwargs_omits_eval_steps_when_strategy_is_not_steps() -> None:
    kwargs = build_sft_kwargs(_config(val_jsonl=Path("data/smoke/val.jsonl")))

    assert "eval_steps" not in kwargs
    assert "save_steps" not in kwargs


def test_build_sft_kwargs_reports_wandb_when_project_set() -> None:
    kwargs = build_sft_kwargs(_config(wandb_project="anvil", wandb_run_name="smoke"))

    assert kwargs["report_to"] == ["wandb"]
    assert kwargs["run_name"] == "smoke"


def test_build_sft_kwargs_disables_wandb_when_unset() -> None:
    kwargs = build_sft_kwargs(_config())

    assert kwargs["report_to"] == []


def test_build_wandb_env_maps_project_and_entity() -> None:
    env = build_wandb_env(_config(wandb_project="anvil", wandb_entity="feRpicoral"))

    assert env == {"WANDB_PROJECT": "anvil", "WANDB_ENTITY": "feRpicoral"}


def test_ensure_wandb_available_skips_when_reporting_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "anvil.training.trl_trainer.importlib.util.find_spec",
        lambda name: pytest.fail("wandb should not be checked"),
    )

    ensure_wandb_available(_config())


def test_ensure_wandb_available_raises_when_reporting_enabled_and_wandb_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WANDB_ENTITY", "feRpicoral")
    monkeypatch.setattr("anvil.training.trl_trainer.importlib.util.find_spec", lambda name: None)

    with pytest.raises(ImportError, match="wandb"):
        ensure_wandb_available(_config(wandb_project="anvil"))


def test_ensure_wandb_available_raises_when_reporting_enabled_and_entity_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WANDB_ENTITY", raising=False)

    with pytest.raises(RuntimeError, match="WANDB_ENTITY"):
        ensure_wandb_available(_config(wandb_project="anvil"))


def test_ensure_wandb_available_accepts_entity_from_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.setattr(
        "anvil.training.trl_trainer.importlib.util.find_spec",
        lambda name: object(),
    )

    ensure_wandb_available(_config(wandb_project="anvil", wandb_entity="feRpicoral"))
