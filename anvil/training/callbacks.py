"""Training-callback policies.

The actual `transformers.TrainerCallback` subclasses live in the trainer
driver (which imports `transformers` on demand). This module owns the
decision logic in pure-Python policy classes so it parses, tests, and
lives inside `uv.lock` without the CUDA-coupled deps.

Each policy is a small dataclass with:
  - immutable parameters (patience, min_delta, keep_n, ...)
  - mutable state (best_loss, bad_steps, history)
  - one method per signal (`update(...)`, `record(...)`)
that returns the decision the wrapping callback should act on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class EarlyStoppingPolicy:
    """Stop training when validation loss stops improving.

    `update(val_loss)` returns True the moment `bad_steps >= patience`.
    An improvement is a drop of at least `min_delta` below `best_loss`.
    """

    patience: int = 2
    min_delta: float = 0.0
    best_loss: float = math.inf
    bad_steps: int = 0

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("patience must be >= 1")
        if self.min_delta < 0 or not math.isfinite(self.min_delta):
            raise ValueError("min_delta must be a non-negative finite number")

    def update(self, val_loss: float) -> bool:
        """Record `val_loss` and return True if training should stop."""
        if not math.isfinite(val_loss):
            raise ValueError("val_loss must be finite")
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.bad_steps = 0
        else:
            self.bad_steps += 1
        return self.bad_steps >= self.patience

    def reset(self) -> None:
        """Forget all observed losses. Useful for resuming runs."""
        self.best_loss = math.inf
        self.bad_steps = 0


@dataclass
class CheckpointRotation:
    """Decide which checkpoints to keep and which to delete.

    The latest `keep_n` checkpoints are always retained. When
    `keep_best=True`, the checkpoint with the lowest `val_loss` is also
    pinned even if it falls outside the recent window.
    """

    keep_n: int = 2
    keep_best: bool = True
    best_step: int | None = None
    best_loss: float = math.inf
    history: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.keep_n < 1:
            raise ValueError("keep_n must be >= 1")

    def record(self, step: int, val_loss: float | None = None) -> list[int]:
        """Record a new checkpoint at `step`. Returns steps the caller should delete.

        `val_loss=None` means we have no validation signal yet; the best-
        checkpoint slot stays whatever it was.
        """
        if step in self.history:
            raise ValueError(f"step {step} already recorded")
        self.history.append(step)
        if val_loss is not None:
            if not math.isfinite(val_loss):
                raise ValueError("val_loss must be finite")
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_step = step

        recent = set(self.history[-self.keep_n :])
        to_keep: set[int] = set(recent)
        if self.keep_best and self.best_step is not None:
            to_keep.add(self.best_step)
        return [step for step in self.history if step not in to_keep]


@dataclass(frozen=True)
class WandbConfig:
    """Minimum fields for `wandb.init()` driven by `TrainingConfig`."""

    project: str
    entity: str | None = None
    run_name: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.project:
            raise ValueError("project must be non-empty")


def wandb_init_kwargs(config: WandbConfig) -> dict[str, object]:
    """Translate a `WandbConfig` into kwargs for `wandb.init()`.

    Empty optional fields are omitted so the W&B library applies its own
    defaults (account-level entity, generated run name, etc.).
    """
    kwargs: dict[str, object] = {"project": config.project}
    if config.entity:
        kwargs["entity"] = config.entity
    if config.run_name:
        kwargs["name"] = config.run_name
    if config.tags:
        kwargs["tags"] = list(config.tags)
    return kwargs
