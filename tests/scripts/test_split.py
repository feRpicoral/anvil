from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import pytest

from anvil.data.splits import SplitContaminationError
from anvil.data.synthesis import GenerationResult
from scripts.split import run


def _record(contract_text: str, contract_type: str = "nda") -> GenerationResult:
    return GenerationResult(
        contract_type=contract_type,  # type: ignore[arg-type]
        contract_text=contract_text,
        extraction={},
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        backend="test",
    )


def _write_input(records: list[GenerationResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(dataclasses.asdict(record)) + "\n")


def _balanced_unique(n_per_type: int) -> list[GenerationResult]:
    records: list[GenerationResult] = []
    for contract_type in ("nda", "msa", "license"):
        for i in range(n_per_type):
            records.append(_record(f"{contract_type} body {i}", contract_type=contract_type))
    return records


def test_run_writes_three_jsonl_files(tmp_path: Path) -> None:
    records = _balanced_unique(10)
    input_path = tmp_path / "in.jsonl"
    output_dir = tmp_path / "out"
    _write_input(records, input_path)
    args = argparse.Namespace(
        input=input_path,
        output_dir=output_dir,
        ratios=[0.8, 0.1, 0.1],
        seed=0,
        allow_overlap=False,
    )

    rc = run(args)

    assert rc == 0
    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "val.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()
    train_rows = (output_dir / "train.jsonl").read_text(encoding="utf-8").strip().splitlines()
    val_rows = (output_dir / "val.jsonl").read_text(encoding="utf-8").strip().splitlines()
    test_rows = (output_dir / "test.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(train_rows) + len(val_rows) + len(test_rows) == 30


def test_run_outputs_messages_shape(tmp_path: Path) -> None:
    records = _balanced_unique(10)
    input_path = tmp_path / "in.jsonl"
    output_dir = tmp_path / "out"
    _write_input(records, input_path)
    args = argparse.Namespace(
        input=input_path,
        output_dir=output_dir,
        ratios=[0.8, 0.1, 0.1],
        seed=0,
        allow_overlap=False,
    )

    run(args)

    first_train = json.loads(
        (output_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert "messages" in first_train
    roles = [m["role"] for m in first_train["messages"]]
    assert roles == ["system", "user", "assistant"]


def test_run_fails_when_overlap_detected_by_default(tmp_path: Path) -> None:
    shared = _record("identical body across records", contract_type="nda")
    # Multiple non-unique records of the same type to force a leak between splits.
    records = [
        shared,
        shared,
        _record("nda body 2", contract_type="nda"),
        _record("msa body 0", contract_type="msa"),
        _record("msa body 1", contract_type="msa"),
        _record("msa body 2", contract_type="msa"),
        _record("license body 0", contract_type="license"),
        _record("license body 1", contract_type="license"),
        _record("license body 2", contract_type="license"),
    ]
    input_path = tmp_path / "in.jsonl"
    _write_input(records, input_path)
    args = argparse.Namespace(
        input=input_path,
        output_dir=tmp_path / "out",
        ratios=[0.6, 0.2, 0.2],
        seed=0,
        allow_overlap=False,
    )

    with pytest.raises(SplitContaminationError):
        run(args)


def test_run_allow_overlap_skips_guard(tmp_path: Path) -> None:
    shared = _record("identical body across records", contract_type="nda")
    records = [
        shared,
        shared,
        _record("nda body 2", contract_type="nda"),
        _record("msa body 0", contract_type="msa"),
        _record("msa body 1", contract_type="msa"),
        _record("msa body 2", contract_type="msa"),
        _record("license body 0", contract_type="license"),
        _record("license body 1", contract_type="license"),
        _record("license body 2", contract_type="license"),
    ]
    input_path = tmp_path / "in.jsonl"
    output_dir = tmp_path / "out"
    _write_input(records, input_path)
    args = argparse.Namespace(
        input=input_path,
        output_dir=output_dir,
        ratios=[0.6, 0.2, 0.2],
        seed=0,
        allow_overlap=True,
    )

    rc = run(args)

    assert rc == 0
    assert (output_dir / "train.jsonl").exists()
