"""TRL-backed QLoRA training driver.

`train(config)` lazy-imports `trl`, `peft`, `transformers`, `datasets`,
`wandb`, and (for NF4/FP4/INT8 paths) `bitsandbytes` so the module itself
stays importable on a bare `uv sync` install — those CUDA-coupled deps
come from `constraints/train.txt`. The pure helpers below run in CI
without the stack installed and carry the only logic worth unit-testing;
the trainer invocation is exercised end-to-end by `make train-smoke`.

Install the stack with:

    uv pip install -c constraints/train.txt trl peft transformers \\
        accelerate datasets wandb

…and add `bitsandbytes` on a CUDA host when you want NF4/FP4/INT8.
"""

from __future__ import annotations

import importlib.util
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, cast

from anvil.training.qlora import TrainingConfig, lora_target_module_names

_INSTALL_HINT = (
    "Training stack not installed. Run:\n"
    "  uv pip install -c constraints/train.txt trl peft transformers accelerate datasets wandb\n"
    "(plus `bitsandbytes` on a CUDA host for NF4/FP4/INT8)."
)


class ChatTemplateTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...


def train(config: TrainingConfig, resume_from: Path | None = None) -> int:
    """Run TRL `SFTTrainer` end-to-end and save the LoRA adapter."""
    try:
        import datasets  # noqa: F401
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc

    ensure_wandb_available(config)
    return _train_impl(config, resume_from)


def ensure_wandb_available(config: TrainingConfig) -> None:
    if not config.wandb_project:
        return
    if not config.wandb_entity and not os.environ.get("WANDB_ENTITY", "").strip():
        raise RuntimeError("W&B reporting requires WANDB_ENTITY or wandb_entity")
    try:
        spec = importlib.util.find_spec("wandb")
    except (ImportError, ValueError) as exc:
        raise ImportError(_INSTALL_HINT) from exc
    if spec is None:
        raise ImportError(_INSTALL_HINT)


def load_messages_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a messages-format JSONL into a list of `{"messages": [...]}` rows."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row is not a JSON object")
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{path}:{line_number}: row missing non-empty 'messages' list")
            for index, message in enumerate(messages):
                if not isinstance(message, dict):
                    raise ValueError(
                        f"{path}:{line_number}: messages[{index}] is not a JSON object"
                    )
                role = message.get("role")
                content = message.get("content")
                if not isinstance(role, str) or not role:
                    raise ValueError(f"{path}:{line_number}: messages[{index}].role must be set")
                if not isinstance(content, str) or not content:
                    raise ValueError(f"{path}:{line_number}: messages[{index}].content must be set")
            records.append(row)
    if not records:
        raise ValueError(f"{path}: no records found")
    return records


def build_chat_formatting_func(
    tokenizer: ChatTemplateTokenizer,
) -> Callable[[dict[str, Any]], list[str]]:
    def formatting_func(example: dict[str, Any]) -> list[str]:
        messages = example.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("example missing non-empty messages list")
        if all(isinstance(message, dict) for message in messages):
            chat_messages = cast(list[dict[str, str]], messages)
            return [
                tokenizer.apply_chat_template(
                    chat_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            ]
        if all(isinstance(batch_item, list) for batch_item in messages):
            batch = cast(list[list[dict[str, str]]], messages)
            return [
                tokenizer.apply_chat_template(
                    batch_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for batch_messages in batch
            ]
        raise ValueError("example messages must be a chat message list")

    return formatting_func


def build_lora_kwargs(config: TrainingConfig) -> dict[str, Any]:
    """Materialize the LoraConfig kwargs from a `TrainingConfig`."""
    return {
        "r": config.rank,
        "lora_alpha": config.alpha,
        "lora_dropout": config.lora_dropout,
        "target_modules": list(lora_target_module_names(config.target_modules)),
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def build_sft_kwargs(config: TrainingConfig) -> dict[str, Any]:
    """Materialize the SFTConfig kwargs from a `TrainingConfig`."""
    eval_strategy = config.eval_strategy if config.val_jsonl is not None else "no"
    kwargs: dict[str, Any] = {
        "output_dir": str(config.output_dir),
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.grad_accum,
        "learning_rate": config.learning_rate,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "max_length": config.max_seq_len,
        "seed": config.seed,
        "save_strategy": config.save_strategy,
        "eval_strategy": eval_strategy,
        "save_total_limit": config.save_total_limit,
        "logging_steps": 5,
        "logging_strategy": "steps",
        "report_to": ["wandb"] if config.wandb_project else [],
    }
    if config.wandb_run_name:
        kwargs["run_name"] = config.wandb_run_name
    if eval_strategy == "steps" and config.eval_steps is not None:
        kwargs["eval_steps"] = config.eval_steps
    if config.save_strategy == "steps" and config.save_steps is not None:
        kwargs["save_steps"] = config.save_steps
    if config.keep_best_only and eval_strategy != "no":
        if config.save_strategy != eval_strategy:
            raise ValueError("save_strategy must match eval_strategy when keep_best_only=True")
        kwargs["load_best_model_at_end"] = True
        kwargs["metric_for_best_model"] = "eval_loss"
        kwargs["greater_is_better"] = False
    return kwargs


def build_wandb_env(config: TrainingConfig) -> dict[str, str]:
    """Materialize W&B environment variables from training config."""
    env: dict[str, str] = {}
    if config.wandb_project:
        env["WANDB_PROJECT"] = config.wandb_project
    if config.wandb_entity:
        env["WANDB_ENTITY"] = config.wandb_entity
    return env


def apply_wandb_env(config: TrainingConfig) -> None:
    os.environ.update(build_wandb_env(config))


def build_model_kwargs(config: TrainingConfig) -> dict[str, Any]:
    """Translate `config.quantization` into `from_pretrained` kwargs.

    `bf16` keeps the model in BF16 (used for the M1 smoke where bitsandbytes
    is unavailable). `nf4` / `fp4` / `int8` go through `BitsAndBytesConfig`
    and require a CUDA host.
    """
    try:
        import torch
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc

    kwargs: dict[str, Any] = {}
    if config.quantization == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif config.quantization in {"nf4", "fp4", "int8"}:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
        if config.quantization == "int8":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            return kwargs
        quant_type = config.quantization
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=quant_type == "nf4",
        )
    return kwargs


def _train_impl(config: TrainingConfig, resume_from: Path | None) -> int:
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    train_records = load_messages_jsonl(config.train_jsonl)
    val_records = load_messages_jsonl(config.val_jsonl) if config.val_jsonl is not None else None

    train_dataset = Dataset.from_list(train_records)
    val_dataset = Dataset.from_list(val_records) if val_records is not None else None

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.base_model, **build_model_kwargs(config))

    apply_wandb_env(config)
    sft_config = SFTConfig(**build_sft_kwargs(config))
    lora_config = LoraConfig(**build_lora_kwargs(config))

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        formatting_func=build_chat_formatting_func(tokenizer),
    )
    trainer.train(resume_from_checkpoint=str(resume_from) if resume_from is not None else None)
    trainer.save_model(str(config.output_dir / "final"))
    return 0
