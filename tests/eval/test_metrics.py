from __future__ import annotations

import json
from datetime import date
from typing import Any

from anvil.data.schema import (
    ContractExtraction,
    DisputeResolution,
    Party,
    Term,
    Termination,
)
from anvil.eval.metrics import (
    aggregate_scores,
    json_validity_rate,
    macro_average,
    score_extraction,
    validate_json_output,
)


def _minimal_payload() -> dict[str, Any]:
    return {
        "parties": [
            {"name": "Acme Corp.", "role": "disclosing_party"},
            {"name": "Globex Industries LLC", "role": "receiving_party"},
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
        "termination": {
            "triggers": ["material breach"],
            "notice_days": 30,
            "cure_period_days": 15,
        },
        "indemnification": None,
        "dispute_resolution": {
            "forum": "litigation",
            "venue": "Wilmington, Delaware",
            "governing_rules": None,
        },
    }


def _build(payload: dict[str, Any]) -> ContractExtraction:
    return ContractExtraction.model_validate(payload)


def test_validate_json_output_accepts_valid_extraction() -> None:
    text = json.dumps(_minimal_payload())

    outcome = validate_json_output(text)

    assert outcome.valid is True
    assert outcome.reason is None


def test_validate_json_output_rejects_malformed_json() -> None:
    outcome = validate_json_output("{not valid json")

    assert outcome.valid is False
    assert outcome.reason is not None
    assert "json_decode" in outcome.reason


def test_validate_json_output_rejects_non_object() -> None:
    outcome = validate_json_output("[1, 2, 3]")

    assert outcome.valid is False
    assert outcome.reason == "not_a_json_object"


def test_validate_json_output_rejects_schema_violation() -> None:
    payload = _minimal_payload()
    payload["parties"] = [payload["parties"][0]]
    outcome = validate_json_output(json.dumps(payload))

    assert outcome.valid is False
    assert outcome.reason is not None
    assert "schema" in outcome.reason


def test_json_validity_rate_empty_returns_zero() -> None:
    assert json_validity_rate([]) == 0.0


def test_json_validity_rate_mixed_set() -> None:
    valid = json.dumps(_minimal_payload())
    invalid = "{bad}"

    rate = json_validity_rate([valid, invalid, valid, invalid, valid])

    assert rate == 3 / 5


def test_score_extraction_identical_inputs_score_one() -> None:
    extraction = _build(_minimal_payload())

    scores = score_extraction(extraction, extraction)

    for field, score in scores.items():
        assert score == 1.0, f"{field} should be 1.0 on identity, got {score}"


def test_score_extraction_returns_all_scored_fields() -> None:
    scores = score_extraction(_build(_minimal_payload()), _build(_minimal_payload()))

    assert set(scores) == {
        "parties",
        "effective_date",
        "term",
        "governing_law",
        "jurisdiction",
        "confidentiality",
        "termination_triggers",
        "termination_notice_days",
        "termination_cure_period_days",
        "indemnification",
        "dispute_forum",
        "dispute_venue",
        "dispute_governing_rules",
    }


def test_score_extraction_party_role_mismatch_drops_parties() -> None:
    payload = _minimal_payload()
    flipped = _minimal_payload()
    flipped["parties"] = [
        {"name": "Acme Corp.", "role": "receiving_party"},
        {"name": "Globex Industries LLC", "role": "disclosing_party"},
    ]

    scores = score_extraction(_build(payload), _build(flipped))

    assert scores["parties"] == 0.0


def test_score_extraction_party_name_match_is_case_insensitive() -> None:
    payload = _minimal_payload()
    cased = _minimal_payload()
    cased["parties"][0]["name"] = "ACME CORP."

    scores = score_extraction(_build(payload), _build(cased))

    assert scores["parties"] == 1.0


def test_score_extraction_effective_date_uses_exact_match() -> None:
    payload = _minimal_payload()
    shifted = _minimal_payload()
    shifted["effective_date"] = "2026-03-15"

    scores = score_extraction(_build(payload), _build(shifted))

    assert scores["effective_date"] == 0.0


def test_score_extraction_term_partial_credit() -> None:
    payload = _minimal_payload()
    near = _minimal_payload()
    near["term"]["duration_months"] = 36

    scores = score_extraction(_build(payload), _build(near))

    assert scores["term"] == 0.75


def test_score_extraction_termination_triggers_set_f1() -> None:
    payload = _minimal_payload()
    payload["termination"]["triggers"] = ["material breach", "insolvency"]
    other = _minimal_payload()
    other["termination"]["triggers"] = ["material breach", "convenience with notice"]

    scores = score_extraction(_build(payload), _build(other))

    assert scores["termination_triggers"] == 0.5


def test_score_extraction_termination_cure_period_exact_match() -> None:
    payload = _minimal_payload()
    other = _minimal_payload()
    other["termination"]["cure_period_days"] = 30

    scores = score_extraction(_build(payload), _build(other))

    assert scores["termination_cure_period_days"] == 0.0


def test_score_extraction_handles_null_confidentiality() -> None:
    payload = _minimal_payload()
    matching = _minimal_payload()

    scores = score_extraction(_build(payload), _build(matching))

    assert scores["confidentiality"] == 1.0


def test_score_extraction_penalizes_missing_optional_block() -> None:
    payload = _minimal_payload()
    with_block = _minimal_payload()
    with_block["confidentiality"] = {
        "scope": "All proprietary information.",
        "duration_months": 60,
        "carveouts": ["public domain"],
    }

    scores = score_extraction(_build(payload), _build(with_block))

    assert scores["confidentiality"] == 0.0


def test_score_extraction_confidentiality_scope_contributes_to_score() -> None:
    payload = _minimal_payload()
    payload["confidentiality"] = {
        "scope": "All proprietary information.",
        "duration_months": 60,
        "carveouts": ["public domain"],
    }
    other = _minimal_payload()
    other["confidentiality"] = {
        "scope": "Only technical information.",
        "duration_months": 60,
        "carveouts": ["public domain"],
    }

    scores = score_extraction(_build(payload), _build(other))

    assert scores["confidentiality"] == 2 / 3


def test_score_extraction_indemnification_scope_contributes_to_score() -> None:
    payload = _minimal_payload()
    payload["indemnification"] = {
        "scope": "Third-party IP claims.",
        "cap_usd": 100000,
        "cap_multiplier": None,
    }
    other = _minimal_payload()
    other["indemnification"] = {
        "scope": "Any and all losses.",
        "cap_usd": 100000,
        "cap_multiplier": None,
    }

    scores = score_extraction(_build(payload), _build(other))

    assert scores["indemnification"] == 2 / 3


def test_score_extraction_governing_law_normalization() -> None:
    payload = _minimal_payload()
    payload["governing_law"] = "  delaware  "
    canonical = _minimal_payload()
    canonical["governing_law"] = "Delaware"

    scores = score_extraction(_build(payload), _build(canonical))

    assert scores["governing_law"] == 1.0


def test_score_extraction_jurisdiction_none_matches_none() -> None:
    payload = _minimal_payload()  # jurisdiction=None
    same = _minimal_payload()

    scores = score_extraction(_build(payload), _build(same))

    assert scores["jurisdiction"] == 1.0


def test_score_extraction_empty_string_does_not_match_null() -> None:
    payload = _minimal_payload()
    payload["jurisdiction"] = ""
    same = _minimal_payload()

    scores = score_extraction(_build(payload), _build(same))

    assert scores["jurisdiction"] == 0.0


def test_macro_average_empty_returns_zero() -> None:
    assert macro_average({}) == 0.0


def test_macro_average_mean() -> None:
    scores = {"a": 1.0, "b": 0.0, "c": 0.5}

    assert macro_average(scores) == 0.5


def test_aggregate_scores_means_each_field() -> None:
    samples = [
        {"parties": 1.0, "term": 0.5},
        {"parties": 0.5, "term": 1.0},
        {"parties": 0.0, "term": 0.0},
    ]

    aggregated = aggregate_scores(samples)

    assert aggregated["parties"] == 0.5
    assert aggregated["term"] == 0.5


def test_aggregate_scores_handles_missing_fields() -> None:
    samples = [
        {"parties": 1.0},
        {"term": 0.5},
    ]

    aggregated = aggregate_scores(samples)

    assert aggregated["parties"] == 0.5
    assert aggregated["term"] == 0.25


def test_aggregate_scores_empty_returns_empty() -> None:
    assert aggregate_scores([]) == {}


def test_score_extraction_dispute_forum_exact() -> None:
    payload = _minimal_payload()
    arbitration = _minimal_payload()
    arbitration["dispute_resolution"]["forum"] = "arbitration"

    scores = score_extraction(_build(payload), _build(arbitration))

    assert scores["dispute_forum"] == 0.0


def test_score_extraction_dispute_venue_normalized() -> None:
    payload = _minimal_payload()
    payload["dispute_resolution"]["venue"] = "WILMINGTON, delaware"
    canonical = _minimal_payload()

    scores = score_extraction(_build(payload), _build(canonical))

    assert scores["dispute_venue"] == 1.0


def test_score_extraction_dispute_governing_rules_normalized() -> None:
    payload = _minimal_payload()
    payload["dispute_resolution"]["governing_rules"] = "AAA Commercial Rules"
    other = _minimal_payload()
    other["dispute_resolution"]["governing_rules"] = " aaa   commercial rules "

    scores = score_extraction(_build(payload), _build(other))

    assert scores["dispute_governing_rules"] == 1.0


def test_can_construct_extraction_directly() -> None:
    extraction = ContractExtraction(
        parties=[
            Party(name="Acme", role="disclosing_party"),
            Party(name="Globex", role="receiving_party"),
        ],
        effective_date=date(2026, 1, 1),
        term=Term(
            duration_months=12,
            is_perpetual=False,
            auto_renew=False,
            renewal_notice_days=None,
        ),
        governing_law="Delaware",
        jurisdiction=None,
        confidentiality=None,
        termination=Termination(triggers=[], notice_days=None, cure_period_days=None),
        indemnification=None,
        dispute_resolution=DisputeResolution(forum="litigation", venue=None, governing_rules=None),
    )

    scores = score_extraction(extraction, extraction)

    assert macro_average(scores) == 1.0
