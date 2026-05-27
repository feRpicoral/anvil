"""Drive QLoRA training with TRL (smoke) or Unsloth (paid).

Loads a `TrainingConfig` from TOML, validates input paths, and dispatches
to the chosen backend driver. The drivers import their CUDA-coupled deps
on demand; with `--dry-run` the CLI validates local config/data wiring
without loading a model.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from anvil.training.qlora import TrainingConfig, load_config


def run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    _validate_inputs(config)

    if args.dry_run:
        _print_summary(config, file=sys.stderr)
        return 0

    return _run_training(config, args.resume_from)


def _validate_inputs(config: TrainingConfig) -> None:
    if not config.train_jsonl.is_file():
        raise FileNotFoundError(f"train_jsonl not found: {config.train_jsonl}")
    if config.val_jsonl is not None and not config.val_jsonl.is_file():
        raise FileNotFoundError(f"val_jsonl not found: {config.val_jsonl}")


def _print_summary(config: TrainingConfig, *, file: TextIO) -> None:
    file.write(f"train: backend={config.backend} base_model={config.base_model}\n")
    file.write(f"  data: train={config.train_jsonl} val={config.val_jsonl}\n")
    file.write(
        f"  lora: rank={config.rank} alpha={config.alpha} "
        f"target={config.target_modules} dropout={config.lora_dropout}\n"
    )
    file.write(
        f"  optim: lr={config.learning_rate} epochs={config.epochs} "
        f"batch={config.batch_size}x{config.grad_accum}={config.effective_batch_size}\n"
    )
    file.write(f"  seq_len={config.max_seq_len} quantization={config.quantization}\n")
    file.write(f"  output_dir={config.output_dir}\n")
    if config.wandb_project:
        file.write(f"  wandb: project={config.wandb_project} run={config.wandb_run_name}\n")


def _run_training(config: TrainingConfig, resume_from: Path | None) -> int:
    if config.backend == "trl":
        from anvil.training.trl_trainer import train as trl_train

        return trl_train(config, resume_from)
    if config.backend == "unsloth":
        from anvil.training.unsloth_trainer import train as unsloth_train

        return unsloth_train(config, resume_from)
    raise ValueError(f"unsupported backend: {config.backend!r}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train QLoRA adapters.")
    parser.add_argument("--config", type=Path, required=True, help="TOML training config.")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume from a saved checkpoint directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config + print summary without loading any model.",
    )
    return parser.parse_args(argv)


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
