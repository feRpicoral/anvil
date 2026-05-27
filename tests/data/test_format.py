from __future__ import annotations

import json
from pathlib import Path

import pytest

from anvil.data.format import (
    EXTRACTION_INSTRUCTION,
    iter_jsonl,
    to_messages,
    write_jsonl,
)
from anvil.data.prompts import ContractType
from anvil.data.synthesis import GenerationResult


def _record(contract_type: ContractType = "nda") -> GenerationResult:
    return GenerationResult(
        contract_type=contract_type,
        contract_text="# NDA\n\nThis is an example NDA for testing.",
        extraction={
            "parties": [
                {"name": "Acme", "role": "disclosing_party"},
                {"name": "Globex", "role": "receiving_party"},
            ],
            "effective_date": "2026-02-15",
            "term": {
                "duration_months": 24,
                "is_perpetual": False,
                "auto_renew": False,
                "renewal_notice_days": None,
            },
            "governing_law": "Delaware",
            "jurisdiction": None,
            "confidentiality": None,
            "termination": {"triggers": [], "notice_days": None, "cure_period_days": None},
            "indemnification": None,
            "dispute_resolution": {"forum": "litigation", "venue": None, "governing_rules": None},
        },
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        backend="test",
    )


def test_to_messages_returns_three_turns() -> None:
    messages = to_messages(_record())

    assert [m["role"] for m in messages] == ["system", "user", "assistant"]


def test_to_messages_system_carries_extraction_instruction() -> None:
    messages = to_messages(_record())

    assert messages[0]["content"] == EXTRACTION_INSTRUCTION


def test_to_messages_user_is_contract_text() -> None:
    record = _record()

    messages = to_messages(record)

    assert messages[1]["content"] == record.contract_text


def test_to_messages_assistant_is_canonical_json() -> None:
    record = _record()

    messages = to_messages(record)

    parsed = json.loads(messages[2]["content"])
    assert parsed == record.extraction
    # sort_keys=True yields a stable serialization across runs.
    assert messages[2]["content"] == json.dumps(
        record.extraction, sort_keys=True, ensure_ascii=False
    )


def test_write_jsonl_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir" / "smoke.jsonl"

    n = write_jsonl([_record()], target)

    assert n == 1
    assert target.exists()


def test_write_jsonl_round_trips_through_iter_jsonl(tmp_path: Path) -> None:
    records = [_record("nda"), _record("msa"), _record("license")]
    path = tmp_path / "out.jsonl"

    write_jsonl(records, path)
    rows = list(iter_jsonl(path))

    assert len(rows) == 3
    for row, record in zip(rows, records, strict=True):
        assert row["messages"][1]["content"] == record.contract_text
        assert json.loads(row["messages"][2]["content"]) == record.extraction


def test_iter_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "with-blank.jsonl"
    path.write_text(
        '{"messages": [{"role": "user", "content": "a"}]}\n\n{"messages": [{"role": "user", "content": "b"}]}\n',
        encoding="utf-8",
    )

    rows = list(iter_jsonl(path))

    assert len(rows) == 2


def test_write_jsonl_zero_records(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"

    n = write_jsonl([], path)

    assert n == 0
    assert path.read_text(encoding="utf-8") == ""


def test_assistant_content_preserves_non_ascii(tmp_path: Path) -> None:
    record = _record()
    extraction_with_unicode = {
        **record.extraction,
        "governing_law": "Bürgerliches Gesetzbuch (Germany)",
    }
    record = GenerationResult(
        contract_type=record.contract_type,
        contract_text=record.contract_text,
        extraction=extraction_with_unicode,
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        backend="test",
    )
    path = tmp_path / "unicode.jsonl"

    write_jsonl([record], path)
    row = next(iter(iter_jsonl(path)))

    assert "Bürgerliches" in row["messages"][2]["content"]


@pytest.mark.parametrize("contract_type", ["nda", "msa", "license"])
def test_each_contract_type_round_trips(tmp_path: Path, contract_type: ContractType) -> None:
    record = _record(contract_type)
    path = tmp_path / "out.jsonl"

    write_jsonl([record], path)
    row = next(iter(iter_jsonl(path)))

    assert row["messages"][0]["role"] == "system"
    assert row["messages"][1]["content"] == record.contract_text
