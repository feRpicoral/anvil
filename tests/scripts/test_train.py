from __future__ import annotations

import argparse
import io
from pathlib import Path

import pytest

from scripts.train import _print_summary, _validate_inputs, parse_args, run


def _write_smoke_dataset(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"messages": [{"role": "system", "content": "x"}]}\n', encoding="utf-8")
    return path


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "train.toml"
    path.write_text(body, encoding="utf-8")
    return path


def _smoke_config_body(tmp_path: Path) -> str:
    train_path = _write_smoke_dataset(tmp_path / "data" / "train.jsonl")
    val_path = _write_smoke_dataset(tmp_path / "data" / "val.jsonl")
    return f"""
        base_model = "Qwen/Qwen2.5-0.5B-Instruct"
        backend = "trl"
        output_dir = "{tmp_path / "outputs"}"
        train_jsonl = "{train_path}"
        val_jsonl = "{val_path}"
        epochs = 1
        max_seq_len = 256
    """


def test_parse_args_requires_config() -> None:
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_args_dry_run_default_false() -> None:
    args = parse_args(["--config", "configs/train-smoke.toml"])

    assert args.dry_run is False
    assert args.resume_from is None
    assert args.config == Path("configs/train-smoke.toml")


def test_run_dry_run_returns_zero(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _smoke_config_body(tmp_path))
    args = argparse.Namespace(config=config_path, resume_from=None, dry_run=True)

    rc = run(args)

    assert rc == 0


def test_run_real_path_dispatches_to_trl_trainer(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _smoke_config_body(tmp_path))
    args = argparse.Namespace(config=config_path, resume_from=None, dry_run=False)

    # CI doesn't install the TRL stack, so the lazy import raises ImportError
    # with the canonical install hint. That confirms dispatch landed in the
    # trl_trainer module rather than the old NotImplementedError stub.
    with pytest.raises(ImportError, match=r"constraints/train\.txt"):
        run(args)


def test_run_unsloth_backend_raises_not_implemented(tmp_path: Path) -> None:
    train_path = _write_smoke_dataset(tmp_path / "data" / "train.jsonl")
    config_path = _write_config(
        tmp_path,
        f"""
        base_model = "meta-llama/Llama-3.1-8B-Instruct"
        backend = "unsloth"
        output_dir = "{tmp_path / "outputs"}"
        train_jsonl = "{train_path}"
        """,
    )
    args = argparse.Namespace(config=config_path, resume_from=None, dry_run=False)

    with pytest.raises(NotImplementedError, match="Unsloth"):
        run(args)


def test_run_raises_for_missing_train_jsonl(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        f"""
        base_model = "Qwen/Qwen2.5-0.5B-Instruct"
        backend = "trl"
        output_dir = "{tmp_path / "outputs"}"
        train_jsonl = "{tmp_path / "missing.jsonl"}"
        """,
    )
    args = argparse.Namespace(config=config_path, resume_from=None, dry_run=True)

    with pytest.raises(FileNotFoundError, match="train_jsonl"):
        run(args)


def test_run_raises_for_missing_val_jsonl(tmp_path: Path) -> None:
    train_path = _write_smoke_dataset(tmp_path / "data" / "train.jsonl")
    config_path = _write_config(
        tmp_path,
        f"""
        base_model = "Qwen/Qwen2.5-0.5B-Instruct"
        backend = "trl"
        output_dir = "{tmp_path / "outputs"}"
        train_jsonl = "{train_path}"
        val_jsonl = "{tmp_path / "missing.jsonl"}"
        """,
    )
    args = argparse.Namespace(config=config_path, resume_from=None, dry_run=True)

    with pytest.raises(FileNotFoundError, match="val_jsonl"):
        run(args)


def test_validate_inputs_accepts_existing_files(tmp_path: Path) -> None:
    from anvil.training.qlora import TrainingConfig

    train_path = _write_smoke_dataset(tmp_path / "train.jsonl")
    val_path = _write_smoke_dataset(tmp_path / "val.jsonl")
    config = TrainingConfig(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        backend="trl",
        output_dir=tmp_path / "outputs",
        train_jsonl=train_path,
        val_jsonl=val_path,
    )

    _validate_inputs(config)


def test_print_summary_includes_required_fields(tmp_path: Path) -> None:
    from anvil.training.qlora import TrainingConfig

    train_path = _write_smoke_dataset(tmp_path / "train.jsonl")
    config = TrainingConfig(
        base_model="Qwen/Qwen2.5-0.5B-Instruct",
        backend="trl",
        output_dir=tmp_path / "outputs",
        train_jsonl=train_path,
        rank=8,
        alpha=16,
        epochs=1,
        wandb_project="anvil-smoke",
    )
    buf = io.StringIO()

    _print_summary(config, file=buf)

    text = buf.getvalue()
    assert "Qwen2.5-0.5B-Instruct" in text
    assert "rank=8" in text
    assert "alpha=16" in text
    assert "epochs=1" in text
    assert "anvil-smoke" in text


def test_smoke_config_file_loads(tmp_path: Path) -> None:
    from anvil.training.qlora import load_config

    config = load_config(Path("configs/train-smoke.toml"))

    assert config.base_model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config.backend == "trl"
    assert config.epochs == 1
    assert config.max_seq_len == 1024
