"""Tests for the pure helpers in `anvil.training.unsloth_trainer`."""

from __future__ import annotations

from pathlib import Path

import pytest

from anvil.training.qlora import TrainingConfig
from anvil.training.unsloth_trainer import (
    build_unsloth_load_kwargs,
    build_unsloth_peft_kwargs,
)


def _config(**overrides: object) -> TrainingConfig:
    defaults: dict[str, object] = {
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "backend": "unsloth",
        "output_dir": Path("outputs/full"),
        "train_jsonl": Path("data/full/train.jsonl"),
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)  # type: ignore[arg-type]


def test_load_kwargs_for_nf4_uses_4bit() -> None:
    kwargs = build_unsloth_load_kwargs(_config(quantization="nf4", max_seq_len=2048))

    assert kwargs["model_name"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert kwargs["max_seq_length"] == 2048
    assert kwargs["load_in_4bit"] is True
    assert kwargs["load_in_8bit"] is False
    assert kwargs["dtype"] is None


def test_load_kwargs_for_fp4_is_rejected() -> None:
    with pytest.raises(ValueError, match="fp4"):
        build_unsloth_load_kwargs(_config(quantization="fp4"))


def test_load_kwargs_for_int8_uses_8bit() -> None:
    kwargs = build_unsloth_load_kwargs(_config(quantization="int8"))

    assert kwargs["load_in_4bit"] is False
    assert kwargs["load_in_8bit"] is True


def test_load_kwargs_for_bf16_disables_quantization() -> None:
    kwargs = build_unsloth_load_kwargs(_config(quantization="bf16"))

    assert kwargs["load_in_4bit"] is False
    assert kwargs["load_in_8bit"] is False


def test_peft_kwargs_maps_lora_fields() -> None:
    config = _config(
        rank=16,
        alpha=32,
        lora_dropout=0.05,
        target_modules="all_linear",
        seed=42,
    )

    kwargs = build_unsloth_peft_kwargs(config)

    assert kwargs["r"] == 16
    assert kwargs["lora_alpha"] == 32
    assert kwargs["lora_dropout"] == 0.05
    assert "q_proj" in kwargs["target_modules"]
    assert "down_proj" in kwargs["target_modules"]
    assert kwargs["bias"] == "none"
    assert kwargs["use_gradient_checkpointing"] == "unsloth"
    assert kwargs["random_state"] == 42


def test_peft_kwargs_qkv_target() -> None:
    kwargs = build_unsloth_peft_kwargs(_config(target_modules="qkv"))

    assert kwargs["target_modules"] == ["q_proj", "k_proj", "v_proj"]
