from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from anvil.data.synthesis import GenerationResult
from scripts.curate import curate, run


def _record(contract_text: str, extraction: dict[str, object] | None = None) -> GenerationResult:
    return GenerationResult(
        contract_type="nda",
        contract_text=contract_text,
        extraction=extraction
        if extraction is not None
        else {
            "parties": [
                {"name": "A", "role": "disclosing_party"},
                {"name": "B", "role": "receiving_party"},
            ],
            "effective_date": "2026-01-01",
            "term": {
                "duration_months": 12,
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


def _write_input(records: list[GenerationResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(dataclasses.asdict(record)) + "\n")


def test_curate_rejects_short_text() -> None:
    records = [_record("a" * 1000), _record("short")]

    survivors, report = curate(records)

    assert len(survivors) == 1
    assert report.reasons.get("length") == 1


def test_curate_rejects_invalid_extraction() -> None:
    bad = _record("a" * 1000, extraction={"parties": []})

    survivors, report = curate([bad])

    assert survivors == []
    assert report.reasons.get("schema") == 1


def test_curate_rejects_non_english() -> None:
    text = "Это конфиденциальное соглашение " * 50
    survivors, report = curate([_record(text)])

    assert survivors == []
    assert report.reasons.get("language") == 1


def test_curate_drops_near_duplicates_keeps_first() -> None:
    body = "This Mutual NDA between Acme Corp. and Globex governs disclosure. " * 20
    a = _record(body)
    b = _record(body)
    c = _record("# Different NDA\n\n" + "Cloudforge and Beacon enter into this agreement. " * 20)

    survivors, report = curate([a, b, c])

    survivor_texts = [s.contract_text for s in survivors]
    assert survivor_texts == [a.contract_text, c.contract_text]
    assert report.reasons.get("duplicate") == 1


def test_curate_no_dedup_keeps_repeats() -> None:
    body = "This Mutual NDA between Acme and Globex governs disclosure. " * 20
    records = [_record(body) for _ in range(5)]

    survivors, report = curate(records, dedup=False)

    assert len(survivors) == 5
    assert "duplicate" not in report.reasons


def test_run_writes_curated_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    records = [_record("a" * 1000), _record("short")]
    _write_input(records, input_path)
    args = argparse.Namespace(input=input_path, output=output_path, no_dedup=False)

    rc = run(args)

    assert rc == 0
    rows = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1


def test_run_passes_no_dedup_flag(tmp_path: Path) -> None:
    body = "This Mutual NDA between parties is comprehensive. " * 30
    records = [_record(body) for _ in range(3)]
    input_path = tmp_path / "in.jsonl"
    output_path = tmp_path / "out.jsonl"
    _write_input(records, input_path)
    args = argparse.Namespace(input=input_path, output=output_path, no_dedup=True)

    rc = run(args)

    assert rc == 0
    rows = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 3
