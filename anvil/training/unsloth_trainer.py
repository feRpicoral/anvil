"""Unsloth-backed QLoRA training driver for the paid Linux + CUDA run.

Same `train(config, resume_from)` Protocol as the TRL driver. The only
difference is the model + adapter setup: Unsloth's `FastLanguageModel`
loads the base model in NF4 with fused Triton kernels and prepares the
LoRA path, then TRL `SFTTrainer` + `SFTConfig` drive the loop the same
way the TRL backend does (so `build_sft_kwargs` is shared).

Linux + CUDA only. Install:

    uv pip install -c constraints/train.txt unsloth trl peft transformers \\
        accelerate datasets bitsandbytes
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from anvil.training.qlora import TrainingConfig, lora_target_module_names
from anvil.training.trl_trainer import (
    apply_wandb_env,
    build_chat_formatting_func,
    build_sft_kwargs,
    load_messages_jsonl,
)

_INSTALL_HINT = (
    "Unsloth training stack not installed. Run:\n"
    "  uv pip install -c constraints/train.txt unsloth trl peft transformers \\\n"
    "      accelerate datasets bitsandbytes"
)


def train(config: TrainingConfig, resume_from: Path | None = None) -> int:
    """Run Unsloth + TRL SFT end-to-end and save the LoRA adapter."""
    build_unsloth_load_kwargs(config)
    try:
        import unsloth  # noqa: F401

        if config.quantization in {"nf4", "int8"}:
            import bitsandbytes  # noqa: F401
        import datasets  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc

    return _train_impl(config, resume_from)


def build_unsloth_load_kwargs(config: TrainingConfig) -> dict[str, Any]:
    """Materialize the kwargs for `FastLanguageModel.from_pretrained`."""
    if config.quantization == "fp4":
        raise ValueError("Unsloth backend does not support fp4; use nf4, int8, or bf16")
    load_in_4bit = config.quantization == "nf4"
    return {
        "model_name": config.base_model,
        "max_seq_length": config.max_seq_len,
        "load_in_4bit": load_in_4bit,
        "load_in_8bit": config.quantization == "int8",
        "dtype": None,
    }


def build_unsloth_peft_kwargs(config: TrainingConfig) -> dict[str, Any]:
    """Materialize the kwargs for `FastLanguageModel.get_peft_model`."""
    return {
        "r": config.rank,
        "lora_alpha": config.alpha,
        "lora_dropout": config.lora_dropout,
        "target_modules": list(lora_target_module_names(config.target_modules)),
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": config.seed,
    }


def _train_impl(config: TrainingConfig, resume_from: Path | None) -> int:
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(**build_unsloth_load_kwargs(config))
    model = FastLanguageModel.get_peft_model(model, **build_unsloth_peft_kwargs(config))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_records = load_messages_jsonl(config.train_jsonl)
    val_records = load_messages_jsonl(config.val_jsonl) if config.val_jsonl is not None else None
    train_dataset = Dataset.from_list(train_records)
    val_dataset = Dataset.from_list(val_records) if val_records is not None else None

    apply_wandb_env(config)
    sft_config = SFTConfig(**build_sft_kwargs(config))
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        formatting_func=build_chat_formatting_func(tokenizer),
    )
    trainer.train(resume_from_checkpoint=str(resume_from) if resume_from is not None else None)

    final_dir = config.output_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    return 0
