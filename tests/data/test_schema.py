from __future__ import annotations

from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError

from anvil.data.schema import ContractExtraction, DisputeForum, PartyRole

FIXTURES = Path(__file__).parent / "fixtures" / "contracts"
VALID_FIXTURES = sorted((FIXTURES / "valid").glob("*.json"))
INVALID_FIXTURES = sorted((FIXTURES / "invalid").glob("*.json"))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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
    payload = {
        "parties": [
            {"name": "A", "role": "buyer"},
            {"name": "B", "role": "seller"},
        ],
        "term": {
            "duration_months": -1,
            "is_perpetual": False,
            "auto_renew": False,
            "renewal_notice_days": None,
        },
        "termination": {"triggers": [], "notice_days": None, "cure_period_days": None},
        "dispute_resolution": {"forum": "litigation", "venue": None, "governing_rules": None},
    }

    with pytest.raises(ValidationError):
        ContractExtraction.model_validate(payload)


def test_json_schema_serializes() -> None:
    schema = ContractExtraction.model_json_schema()

    assert schema["type"] == "object"
    assert "parties" in schema["properties"]
    assert "dispute_resolution" in schema["properties"]
