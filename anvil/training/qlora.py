"""Backend-agnostic QLoRA training config.

The same `TrainingConfig` drives both the M1 smoke (TRL `SFTTrainer` +
non-quantized LoRA) and the paid run (Unsloth + QLoRA on a 4090). The
config does not import any CUDA-coupled module so it can be parsed,
validated, and tested under the regular `uv sync` install on macOS;
the actual trainer wiring lives in a separate driver that imports
`trl`/`unsloth`/`peft`/`bitsandbytes` on demand.
"""

from __future__ import annotations

import dataclasses
import math
import tomllib
from pathlib import Path
from typing import Final, Literal

Backend = Literal["unsloth", "trl"]
Quantization = Literal["nf4", "fp4", "int8", "bf16"]
LoraTargets = Literal["q_only", "qkv", "qkvo", "all_linear"]
EvalStrategy = Literal["epoch", "steps", "no"]
SaveStrategy = Literal["epoch", "steps", "no"]

_BACKENDS: Final[tuple[Backend, ...]] = ("unsloth", "trl")
_QUANTIZATIONS: Final[tuple[Quantization, ...]] = ("nf4", "fp4", "int8", "bf16")
_LORA_TARGETS: Final[tuple[LoraTargets, ...]] = ("q_only", "qkv", "qkvo", "all_linear")
_EVAL_STRATEGIES: Final[tuple[EvalStrategy, ...]] = ("epoch", "steps", "no")
_SAVE_STRATEGIES: Final[tuple[SaveStrategy, ...]] = ("epoch", "steps", "no")

_TARGET_MODULE_NAMES: Final[dict[LoraTargets, tuple[str, ...]]] = {
    "q_only": ("q_proj",),
    "qkv": ("q_proj", "k_proj", "v_proj"),
    "qkvo": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "all_linear": (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
}


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    """Backend-agnostic config consumed by both Unsloth and TRL drivers."""

    base_model: str
    backend: Backend
    output_dir: Path
    train_jsonl: Path
    val_jsonl: Path | None = None

    rank: int = 16
    alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: LoraTargets = "all_linear"
    quantization: Quantization = "nf4"

    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 1
    grad_accum: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_seq_len: int = 2048
    seed: int = 0

    eval_strategy: EvalStrategy = "epoch"
    eval_steps: int | None = None
    save_strategy: SaveStrategy = "epoch"
    save_steps: int | None = None
    save_total_limit: int = 2
    keep_best_only: bool = True

    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_name: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {_BACKENDS}, got {self.backend!r}")
        if self.quantization not in _QUANTIZATIONS:
            raise ValueError(
                f"quantization must be one of {_QUANTIZATIONS}, got {self.quantization!r}"
            )
        if self.target_modules not in _LORA_TARGETS:
            raise ValueError(
                f"target_modules must be one of {_LORA_TARGETS}, got {self.target_modules!r}"
            )
        if self.eval_strategy not in _EVAL_STRATEGIES:
            raise ValueError(
                f"eval_strategy must be one of {_EVAL_STRATEGIES}, got {self.eval_strategy!r}"
            )
        if self.save_strategy not in _SAVE_STRATEGIES:
            raise ValueError(
                f"save_strategy must be one of {_SAVE_STRATEGIES}, got {self.save_strategy!r}"
            )

        if not self.base_model:
            raise ValueError("base_model must be non-empty")

        if self.rank < 1:
            raise ValueError("rank must be >= 1")
        if self.alpha < 1:
            raise ValueError("alpha must be >= 1")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("lora_dropout must be in [0, 1)")

        if self.learning_rate <= 0 or not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be a positive finite number")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.grad_accum < 1:
            raise ValueError("grad_accum must be >= 1")
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be in [0, 1]")
        if self.weight_decay < 0 or not math.isfinite(self.weight_decay):
            raise ValueError("weight_decay must be a non-negative finite number")
        if self.max_seq_len < 1:
            raise ValueError("max_seq_len must be >= 1")

        if self.eval_strategy == "steps" and (self.eval_steps is None or self.eval_steps < 1):
            raise ValueError("eval_steps must be a positive int when eval_strategy='steps'")
        if self.save_strategy == "steps" and (self.save_steps is None or self.save_steps < 1):
            raise ValueError("save_steps must be a positive int when save_strategy='steps'")
        if self.save_total_limit < 1:
            raise ValueError("save_total_limit must be >= 1")

    @property
    def effective_batch_size(self) -> int:
        """Per-step token-window batch, i.e. `batch_size * grad_accum`."""
        return self.batch_size * self.grad_accum


def lora_target_module_names(target: LoraTargets) -> tuple[str, ...]:
    """Expand a symbolic target-modules name into Llama-family projection names."""
    if target not in _TARGET_MODULE_NAMES:
        raise ValueError(f"unknown target_modules: {target!r}")
    return _TARGET_MODULE_NAMES[target]


def load_config(path: Path) -> TrainingConfig:
    """Parse a TOML training config into a `TrainingConfig`."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    try:
        base_model = _required_str(raw, "base_model")
        backend = _required_str(raw, "backend")
        output_dir = Path(_required_str(raw, "output_dir"))
        train_jsonl = Path(_required_str(raw, "train_jsonl"))
    except KeyError as exc:
        raise ValueError(f"{path}: missing required key {exc}") from exc

    val_jsonl = Path(_required_str(raw, "val_jsonl")) if "val_jsonl" in raw else None

    fields = {
        "base_model": base_model,
        "backend": backend,
        "output_dir": output_dir,
        "train_jsonl": train_jsonl,
        "val_jsonl": val_jsonl,
    }
    fields.update(_optional_fields(raw))

    try:
        return TrainingConfig(**fields)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{path}: invalid field — {exc}") from exc


def _required_str(raw: dict[str, object], key: str) -> str:
    value = raw[key]
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _optional_fields(raw: dict[str, object]) -> dict[str, object]:
    """Pull recognised optional keys from a TOML dict, preserving defaults."""
    optional_keys = {
        "rank": int,
        "alpha": int,
        "lora_dropout": float,
        "target_modules": str,
        "quantization": str,
        "learning_rate": float,
        "epochs": int,
        "batch_size": int,
        "grad_accum": int,
        "warmup_ratio": float,
        "weight_decay": float,
        "max_seq_len": int,
        "seed": int,
        "eval_strategy": str,
        "eval_steps": int,
        "save_strategy": str,
        "save_steps": int,
        "save_total_limit": int,
        "keep_best_only": bool,
        "wandb_project": str,
        "wandb_entity": str,
        "wandb_run_name": str,
    }
    out: dict[str, object] = {}
    for key, expected_type in optional_keys.items():
        if key not in raw:
            continue
        value = raw[key]
        if expected_type is bool and not isinstance(value, bool):
            raise ValueError(f"{key} must be a boolean")
        if expected_type is int and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError(f"{key} must be an integer")
        if expected_type is float and (
            isinstance(value, bool) or not isinstance(value, int | float)
        ):
            raise ValueError(f"{key} must be a number")
        if expected_type is str and not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        out[key] = value
    return out
