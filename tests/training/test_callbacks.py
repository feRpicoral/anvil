from __future__ import annotations

import math

import pytest

from anvil.training.callbacks import (
    CheckpointRotation,
    EarlyStoppingPolicy,
    WandbConfig,
    wandb_init_kwargs,
)


def test_early_stop_resets_on_improvement() -> None:
    policy = EarlyStoppingPolicy(patience=2)

    assert policy.update(1.0) is False
    assert policy.update(0.9) is False
    assert policy.update(0.8) is False
    assert policy.bad_steps == 0
    assert policy.best_loss == 0.8


def test_early_stop_stops_after_patience_bad_steps() -> None:
    policy = EarlyStoppingPolicy(patience=2)

    assert policy.update(1.0) is False
    assert policy.update(1.1) is False
    assert policy.update(1.2) is True


def test_early_stop_min_delta_respects_tolerance() -> None:
    policy = EarlyStoppingPolicy(patience=1, min_delta=0.05)

    policy.update(1.0)
    # 0.96 is an improvement (drop of 0.04) but smaller than min_delta.
    assert policy.update(0.96) is True


def test_early_stop_recovers_after_intermittent_plateau() -> None:
    policy = EarlyStoppingPolicy(patience=3)

    policy.update(1.0)
    policy.update(1.05)
    policy.update(1.05)
    policy.update(0.5)

    assert policy.bad_steps == 0


def test_early_stop_reset_clears_state() -> None:
    policy = EarlyStoppingPolicy(patience=2)
    policy.update(1.0)
    policy.update(1.1)

    policy.reset()

    assert policy.bad_steps == 0
    assert policy.best_loss == math.inf


def test_early_stop_rejects_invalid_constructor_args() -> None:
    with pytest.raises(ValueError, match="patience"):
        EarlyStoppingPolicy(patience=0)
    with pytest.raises(ValueError, match="min_delta"):
        EarlyStoppingPolicy(min_delta=-0.1)
    with pytest.raises(ValueError, match="min_delta"):
        EarlyStoppingPolicy(min_delta=float("inf"))


def test_early_stop_rejects_non_finite_loss() -> None:
    policy = EarlyStoppingPolicy()

    with pytest.raises(ValueError, match="val_loss"):
        policy.update(float("nan"))


def test_checkpoint_rotation_keeps_only_recent_when_no_best() -> None:
    rotation = CheckpointRotation(keep_n=2, keep_best=False)

    assert rotation.record(100) == []
    assert rotation.record(200) == []
    assert rotation.record(300) == [100]
    assert rotation.record(400) == [100, 200]


def test_checkpoint_rotation_pins_best_outside_recent_window() -> None:
    rotation = CheckpointRotation(keep_n=2, keep_best=True)

    rotation.record(100, val_loss=2.0)
    rotation.record(200, val_loss=0.5)  # best
    rotation.record(300, val_loss=1.0)
    delete_at_400 = rotation.record(400, val_loss=1.5)

    assert 200 not in delete_at_400
    assert 100 in delete_at_400


def test_checkpoint_rotation_updates_best_when_lower_loss_seen() -> None:
    rotation = CheckpointRotation(keep_n=1, keep_best=True)

    rotation.record(100, val_loss=2.0)
    rotation.record(200, val_loss=0.5)
    rotation.record(300, val_loss=0.3)

    assert rotation.best_step == 300
    assert rotation.best_loss == 0.3


def test_checkpoint_rotation_handles_missing_val_loss() -> None:
    rotation = CheckpointRotation(keep_n=2, keep_best=True)

    rotation.record(100)
    rotation.record(200)
    delete_at_300 = rotation.record(300)

    assert rotation.best_step is None
    assert delete_at_300 == [100]


def test_checkpoint_rotation_rejects_invalid_constructor_args() -> None:
    with pytest.raises(ValueError, match="keep_n"):
        CheckpointRotation(keep_n=0)


def test_checkpoint_rotation_rejects_duplicate_step() -> None:
    rotation = CheckpointRotation()
    rotation.record(100)

    with pytest.raises(ValueError, match="already recorded"):
        rotation.record(100)


def test_checkpoint_rotation_rejects_non_finite_loss() -> None:
    rotation = CheckpointRotation()

    with pytest.raises(ValueError, match="val_loss"):
        rotation.record(100, val_loss=float("inf"))


def test_wandb_config_rejects_empty_project() -> None:
    with pytest.raises(ValueError, match="project"):
        WandbConfig(project="")


def test_wandb_init_kwargs_minimal() -> None:
    kwargs = wandb_init_kwargs(WandbConfig(project="anvil"))

    assert kwargs == {"project": "anvil"}


def test_wandb_init_kwargs_full() -> None:
    config = WandbConfig(
        project="anvil",
        entity="feRpicoral",
        run_name="smoke-2026-05-27",
        tags=("smoke", "trl"),
    )

    kwargs = wandb_init_kwargs(config)

    assert kwargs == {
        "project": "anvil",
        "entity": "feRpicoral",
        "name": "smoke-2026-05-27",
        "tags": ["smoke", "trl"],
    }


def test_wandb_init_kwargs_drops_empty_strings() -> None:
    config = WandbConfig(project="anvil", entity=None, run_name=None)

    kwargs = wandb_init_kwargs(config)

    assert "entity" not in kwargs
    assert "name" not in kwargs
    assert "tags" not in kwargs
