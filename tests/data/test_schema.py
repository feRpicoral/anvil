from __future__ import annotations

import json
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from anvil.data.schema import (
    ContractExtraction,
    DisputeForum,
    PartyRole,
    contract_extraction_json_schema,
)

FIXTURES = Path(__file__).parent / "fixtures" / "contracts"
VALID_FIXTURES = sorted((FIXTURES / "valid").glob("*.json"))
INVALID_FIXTURES = sorted((FIXTURES / "invalid").glob("*.json"))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load(path: Path) -> dict[str, object]:
    payload: dict[str, object] = json.loads(_read(path))
    return payload


@pytest.mark.parametrize("fixture", VALID_FIXTURES, ids=lambda p: p.stem)
def test_valid_fixture_parses(fixture: Path) -> None:
    extraction = ContractExtraction.model_validate_json(_read(fixture))

    assert len(extraction.parties) >= 2
    assert extraction.term is not None
    assert extraction.termination is not None
    assert extraction.dispute_resolution is not None


@pytest.mark.parametrize("fixture", INVALID_FIXTURES, ids=lambda p: p.stem)
def test_invalid_fixture_rejected(fixture: Path) -> None:
    with pytest.raises(ValidationError):
        ContractExtraction.model_validate_json(_read(fixture))


def test_party_role_enum_is_closed() -> None:
    valid_roles = {
        "disclosing_party",
        "receiving_party",
        "buyer",
        "seller",
        "vendor",
        "client",
        "licensor",
        "licensee",
        "other",
    }

    assert set(get_args(PartyRole)) == valid_roles


def test_dispute_forum_enum_is_closed() -> None:
    valid_forums = {"arbitration", "litigation", "mediation_then_arbitration", "other"}

    assert set(get_args(DisputeForum)) == valid_forums


def test_perpetual_term_accepted() -> None:
    extraction = ContractExtraction.model_validate_json(
        _read(FIXTURES / "valid" / "perpetual_license.json")
    )

    assert extraction.term.is_perpetual is True
    assert extraction.term.duration_months is None


def test_msa_carries_auto_renew_and_indemnification() -> None:
    extraction = ContractExtraction.model_validate_json(
        _read(FIXTURES / "valid" / "msa_basic.json")
    )

    assert extraction.term.auto_renew is True
    assert extraction.term.renewal_notice_days == 90
    assert extraction.indemnification is not None
    assert extraction.indemnification.cap_multiplier == 2.0


def test_negative_duration_rejected() -> None:
    payload: dict[str, object] = {
        "parties": [
            {"name": "A", "role": "buyer"},
            {"name": "B", "role": "seller"},
        ],
        "effective_date": None,
        "term": {
            "duration_months": -1,
            "is_perpetual": False,
            "auto_renew": False,
            "renewal_notice_days": None,
        },
        "governing_law": None,
        "jurisdiction": None,
        "confidentiality": None,
        "termination": {"triggers": [], "notice_days": None, "cure_period_days": None},
        "indemnification": None,
        "dispute_resolution": {"forum": "litigation", "venue": None, "governing_rules": None},
    }

    with pytest.raises(ValidationError):
        ContractExtraction.model_validate(payload)


def test_stringly_typed_numbers_and_bools_are_rejected() -> None:
    payload = _load(FIXTURES / "valid" / "nda_basic.json")
    term = payload["term"]
    assert isinstance(term, dict)
    term["duration_months"] = "24"
    term["auto_renew"] = "false"

    with pytest.raises(ValidationError):
        ContractExtraction.model_validate(payload)


def test_perpetual_term_rejects_fixed_duration() -> None:
    payload = _load(FIXTURES / "valid" / "perpetual_license.json")
    term = payload["term"]
    assert isinstance(term, dict)
    term["duration_months"] = 12

    with pytest.raises(ValidationError):
        ContractExtraction.model_validate(payload)


def test_renewal_notice_requires_auto_renew() -> None:
    payload = _load(FIXTURES / "valid" / "nda_basic.json")
    term = payload["term"]
    assert isinstance(term, dict)
    term["renewal_notice_days"] = 30

    with pytest.raises(ValidationError):
        ContractExtraction.model_validate(payload)


def test_indemnification_rejects_multiple_cap_forms() -> None:
    payload = _load(FIXTURES / "valid" / "msa_basic.json")
    indemnification = payload["indemnification"]
    assert isinstance(indemnification, dict)
    indemnification["cap_usd"] = 1_000_000

    with pytest.raises(ValidationError):
        ContractExtraction.model_validate(payload)


def test_json_schema_serializes_for_openai_strict_outputs() -> None:
    schema = contract_extraction_json_schema()

    assert schema["type"] == "object"
    assert "parties" in schema["properties"]
    assert "dispute_resolution" in schema["properties"]
    _assert_all_object_properties_required(schema)


def _assert_all_object_properties_required(value: object) -> None:
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            assert set(value["required"]) == set(properties)
        for child in value.values():
            _assert_all_object_properties_required(child)
    elif isinstance(value, list):
        for child in value:
            _assert_all_object_properties_required(child)
