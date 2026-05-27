from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from anvil.data.curation import (
    CurationOutcome,
    find_near_duplicate_groups,
    is_mostly_english,
    length_in_range,
    validate_extraction,
)

FIXTURES = Path(__file__).parent / "fixtures" / "contracts"


def _load(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def test_validate_extraction_accepts_valid_fixture() -> None:
    payload = _load(FIXTURES / "valid" / "nda_basic.json")

    outcome = validate_extraction(payload)

    assert outcome.accepted is True
    assert outcome.reason is None
    assert outcome.extraction is not None
    assert outcome.extraction.parties[0].name == "Acme Corp."


def test_validate_extraction_rejects_extra_field() -> None:
    payload = _load(FIXTURES / "invalid" / "extra_field.json")

    outcome = validate_extraction(payload)

    assert outcome.accepted is False
    assert outcome.reason is not None
    assert "schema_invalid" in outcome.reason
    assert outcome.extraction is None


def test_validate_extraction_rejects_missing_party() -> None:
    payload = _load(FIXTURES / "invalid" / "missing_party.json")

    outcome = validate_extraction(payload)

    assert outcome.accepted is False
    assert outcome.extraction is None


def test_length_in_range_accepts_typical_contract() -> None:
    text = "x" * 2000

    assert length_in_range(text) is True


def test_length_in_range_rejects_too_short() -> None:
    text = "x" * 100

    assert length_in_range(text) is False


def test_length_in_range_rejects_too_long() -> None:
    text = "x" * 60_000

    assert length_in_range(text) is False


def test_length_in_range_respects_custom_bounds() -> None:
    text = "x" * 100

    assert length_in_range(text, min_chars=50, max_chars=200) is True


def test_is_mostly_english_accepts_legal_prose() -> None:
    text = "This Agreement is entered into as of January 15, 2026, by and between Acme Corp. and Globex Industries LLC."

    assert is_mostly_english(text) is True


def test_is_mostly_english_rejects_cyrillic() -> None:
    text = "Это соглашение заключено между сторонами на следующих условиях."

    assert is_mostly_english(text) is False


def test_is_mostly_english_rejects_empty() -> None:
    assert is_mostly_english("") is False


def test_find_near_duplicate_groups_identifies_clear_pair() -> None:
    a = "This Mutual Non-Disclosure Agreement is entered into as of February 1, 2026."
    b = "This Mutual Non-Disclosure Agreement is entered into as of February 1, 2026."
    c = "Master Services Agreement between Cloudforge Inc. and Beacon Financial Services."

    groups = find_near_duplicate_groups([a, b, c], threshold=0.95)

    assert groups == [{0, 1}]


def test_find_near_duplicate_groups_returns_empty_when_unique() -> None:
    a = "NDA between Party A and Party B governing the disclosure of trade secrets."
    b = "Master Services Agreement covering ongoing IT consulting deliverables."
    c = "Perpetual software license between Lyra Technologies and Orion Robotics."

    groups = find_near_duplicate_groups([a, b, c], threshold=0.9)

    assert groups == []


def test_find_near_duplicate_groups_handles_prefix_truncation() -> None:
    shared_prefix = (
        "Standard Master Services Agreement entered into as of January 1, 2026 between vendor and client. "
        * 5
    )
    a = shared_prefix + "Vendor obligations: hosting, support, on-call. " * 20
    b = shared_prefix + "Vendor obligations: implementation, training. " * 20

    groups = find_near_duplicate_groups([a, b], threshold=0.9, prefix_chars=200)

    assert groups == [{0, 1}]


def test_curation_outcome_is_immutable() -> None:
    outcome = CurationOutcome(accepted=True)

    try:
        outcome.accepted = False  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("CurationOutcome should be frozen")
