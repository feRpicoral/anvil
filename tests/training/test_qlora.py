from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from anvil.training.qlora import (
    TrainingConfig,
    load_config,
    lora_target_module_names,
)


def _kwargs(**overrides: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "backend": "trl",
        "output_dir": Path("/tmp/outputs"),
        "train_jsonl": Path("/tmp/train.jsonl"),
    }
    defaults.update(overrides)
    return defaults


def test_defaults_are_qlora_friendly() -> None:
    config = TrainingConfig(**_kwargs())

    assert config.rank == 16
    assert config.alpha == 32
    assert config.lora_dropout == 0.05
    assert config.quantization == "nf4"
    assert config.target_modules == "all_linear"
    assert config.epochs == 3
    assert config.max_seq_len == 2048


def test_effective_batch_size_is_product() -> None:
    config = TrainingConfig(**_kwargs(batch_size=2, grad_accum=4))

    assert config.effective_batch_size == 8


@pytest.mark.parametrize(
    ("field_name", "bad_value", "message"),
    [
        ("rank", 0, "rank must be"),
        ("alpha", 0, "alpha must be"),
        ("lora_dropout", 1.0, "lora_dropout"),
        ("lora_dropout", -0.1, "lora_dropout"),
        ("learning_rate", 0.0, "learning_rate"),
        ("learning_rate", float("inf"), "learning_rate"),
        ("epochs", 0, "epochs"),
        ("batch_size", 0, "batch_size"),
        ("grad_accum", 0, "grad_accum"),
        ("warmup_ratio", -0.1, "warmup_ratio"),
        ("warmup_ratio", 1.1, "warmup_ratio"),
        ("weight_decay", -0.01, "weight_decay"),
        ("weight_decay", float("nan"), "weight_decay"),
        ("weight_decay", float("inf"), "weight_decay"),
        ("max_seq_len", 0, "max_seq_len"),
        ("save_total_limit", 0, "save_total_limit"),
    ],
)
def test_range_validators_reject_bad_values(
    field_name: str,
    bad_value: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TrainingConfig(**_kwargs(**{field_name: bad_value}))


def test_backend_must_be_enum() -> None:
    with pytest.raises(ValueError, match="backend must be one of"):
        TrainingConfig(**_kwargs(backend="totally-not-a-backend"))


def test_quantization_must_be_enum() -> None:
    with pytest.raises(ValueError, match="quantization must be one of"):
        TrainingConfig(**_kwargs(quantization="int1"))


def test_target_modules_must_be_enum() -> None:
    with pytest.raises(ValueError, match="target_modules must be one of"):
        TrainingConfig(**_kwargs(target_modules="qk"))


def test_steps_strategy_requires_step_counts() -> None:
    with pytest.raises(ValueError, match="eval_steps"):
        TrainingConfig(**_kwargs(eval_strategy="steps"))
    with pytest.raises(ValueError, match="save_steps"):
        TrainingConfig(**_kwargs(save_strategy="steps"))


def test_steps_strategy_with_counts_is_valid() -> None:
    config = TrainingConfig(
        **_kwargs(
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
        )
    )

    assert config.eval_steps == 50
    assert config.save_steps == 100


def test_empty_base_model_rejected() -> None:
    with pytest.raises(ValueError, match="base_model"):
        TrainingConfig(**_kwargs(base_model=""))


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("q_only", ("q_proj",)),
        ("qkv", ("q_proj", "k_proj", "v_proj")),
        ("qkvo", ("q_proj", "k_proj", "v_proj", "o_proj")),
    ],
)
def test_lora_target_module_names_for_attention_subsets(
    target: str,
    expected: tuple[str, ...],
) -> None:
    assert lora_target_module_names(target) == expected  # type: ignore[arg-type]


def test_lora_target_module_names_all_linear_includes_mlp() -> None:
    names = lora_target_module_names("all_linear")

    assert {"q_proj", "k_proj", "v_proj", "o_proj"}.issubset(names)
    assert {"gate_proj", "up_proj", "down_proj"}.issubset(names)


def test_lora_target_module_names_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown target_modules"):
        lora_target_module_names("garbage")  # type: ignore[arg-type]


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "train.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_config_minimal_fields(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        """
        base_model = "Qwen/Qwen2.5-0.5B-Instruct"
        backend = "trl"
        output_dir = "outputs/smoke"
        train_jsonl = "data/smoke/train.jsonl"
        """,
    )

    config = load_config(path)

    assert config.base_model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config.backend == "trl"
    assert config.output_dir == Path("outputs/smoke")
    assert config.train_jsonl == Path("data/smoke/train.jsonl")
    assert config.val_jsonl is None


def test_load_config_overrides_optional_fields(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        """
        base_model = "Qwen/Qwen2.5-0.5B-Instruct"
        backend = "trl"
        output_dir = "outputs/smoke"
        train_jsonl = "data/smoke/train.jsonl"
        val_jsonl = "data/smoke/val.jsonl"
        rank = 8
        alpha = 16
        learning_rate = 5e-5
        epochs = 1
        max_seq_len = 1024
        eval_strategy = "steps"
        eval_steps = 25
        wandb_project = "anvil-smoke"
        """,
    )

    config = load_config(path)

    assert config.rank == 8
    assert config.alpha == 16
    assert config.learning_rate == 5e-5
    assert config.epochs == 1
    assert config.max_seq_len == 1024
    assert config.eval_strategy == "steps"
    assert config.eval_steps == 25
    assert config.wandb_project == "anvil-smoke"
    assert config.val_jsonl == Path("data/smoke/val.jsonl")


def test_load_config_rejects_missing_required(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        """
        base_model = "Qwen/Qwen2.5-0.5B-Instruct"
        backend = "trl"
        """,
    )

    with pytest.raises(ValueError, match="missing required key"):
        load_config(path)


def test_load_config_rejects_wrong_field_type(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        """
        base_model = "Qwen/Qwen2.5-0.5B-Instruct"
        backend = "trl"
        output_dir = "outputs/smoke"
        train_jsonl = "data/smoke/train.jsonl"
        epochs = "three"
        """,
    )

    with pytest.raises(ValueError, match="epochs must be an integer"):
        load_config(path)


@pytest.mark.parametrize("field_name", ["base_model", "backend", "output_dir", "train_jsonl"])
def test_load_config_rejects_wrong_required_field_type(
    tmp_path: Path,
    field_name: str,
) -> None:
    values = {
        "base_model": '"Qwen/Qwen2.5-0.5B-Instruct"',
        "backend": '"trl"',
        "output_dir": '"outputs/smoke"',
        "train_jsonl": '"data/smoke/train.jsonl"',
    }
    values[field_name] = "123"
    path = _write_config(tmp_path, "\n".join(f"{key} = {value}" for key, value in values.items()))

    with pytest.raises(ValueError, match=f"{field_name} must be a string"):
        load_config(path)


def test_load_config_rejects_wrong_val_jsonl_type(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        """
        base_model = "Qwen/Qwen2.5-0.5B-Instruct"
        backend = "trl"
        output_dir = "outputs/smoke"
        train_jsonl = "data/smoke/train.jsonl"
        val_jsonl = false
        """,
    )

    with pytest.raises(ValueError, match="val_jsonl must be a string"):
        load_config(path)


def test_load_config_passes_through_validation_errors(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        """
        base_model = "Qwen/Qwen2.5-0.5B-Instruct"
        backend = "trl"
        output_dir = "outputs/smoke"
        train_jsonl = "data/smoke/train.jsonl"
        rank = 0
        """,
    )

    with pytest.raises(ValueError, match="rank must be"):
        load_config(path)
