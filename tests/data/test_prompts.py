from __future__ import annotations

import pytest

from anvil.data.prompts import (
    CLAUSE_COMPLEXITIES,
    CONTRACT_TYPES,
    PromptParameters,
    render_user_prompt,
    system_prompt,
)


def _params(contract_type: str, **overrides: object) -> PromptParameters:
    defaults: dict[str, object] = {
        "contract_type": contract_type,
        "party_a": "Acme Corp.",
        "party_a_jurisdiction": "Delaware",
        "party_b": "Globex Industries LLC",
        "party_b_jurisdiction": "Delaware",
        "effective_date": "2026-02-15",
        "term_months": 24,
        "governing_law": "State of Delaware",
        "dispute_forum": "litigation",
        "edge_case": "none",
        "clause_complexity": "standard",
        "extras": {},
    }
    defaults.update(overrides)
    return PromptParameters(**defaults)  # type: ignore[arg-type]


def test_system_prompt_is_non_empty_and_describes_response_shape() -> None:
    prompt = system_prompt()

    assert "contract_text" in prompt
    assert "extraction" in prompt
    assert "JSON" in prompt


def test_contract_types_constant_matches_keys() -> None:
    assert set(CONTRACT_TYPES) == {"nda", "msa", "license"}


def test_clause_complexities_constant() -> None:
    assert set(CLAUSE_COMPLEXITIES) == {"minimal", "standard", "comprehensive"}


def test_prompt_parameters_extras_default_to_none() -> None:
    parameters = PromptParameters(
        contract_type="nda",
        party_a="Acme Corp.",
        party_a_jurisdiction="Delaware",
        party_b="Globex Industries LLC",
        party_b_jurisdiction="Delaware",
        effective_date="2026-02-15",
        term_months=24,
        governing_law="State of Delaware",
        dispute_forum="litigation",
        edge_case="none",
    )

    assert parameters.extras is None


@pytest.mark.parametrize("contract_type", ["nda", "msa", "license"])
def test_each_contract_type_renders(contract_type: str) -> None:
    extras: dict[str, str] = {}
    if contract_type == "nda":
        extras = {"confidentiality_months": "60"}
    elif contract_type == "msa":
        extras = {
            "auto_renew": "true",
            "renewal_notice_days": "90",
            "indemnification_cap": "2x annual fees",
        }
    elif contract_type == "license":
        extras = {
            "term_kind": "perpetual",
            "term_description": "perpetual, no fixed expiration",
            "indemnification_cap": "$5,000,000",
        }
    rendered = render_user_prompt(_params(contract_type, extras=extras))

    assert "Acme Corp." in rendered
    assert "State of Delaware" in rendered
    assert "{" not in rendered, "unrendered template variables remain"


def test_render_msa_requires_msa_specific_extras() -> None:
    with pytest.raises(KeyError):
        render_user_prompt(_params("msa", extras={}))


def test_render_license_with_term_description() -> None:
    rendered = render_user_prompt(
        _params(
            "license",
            extras={
                "term_kind": "fixed-term",
                "term_description": "24 months",
                "indemnification_cap": "1.5x annual fees",
            },
        )
    )

    assert "fixed-term" in rendered
    assert "24 months" in rendered


def test_render_handles_perpetual_license() -> None:
    rendered = render_user_prompt(
        _params(
            "license",
            term_months=None,
            extras={
                "term_kind": "perpetual",
                "term_description": "perpetual, no fixed expiration",
                "indemnification_cap": "$5,000,000",
            },
        )
    )

    assert "perpetual" in rendered
    assert "{" not in rendered


def test_term_months_renders_as_na_when_none_for_nda() -> None:
    rendered = render_user_prompt(
        _params(
            "nda",
            term_months=None,
            extras={"confidentiality_months": "60"},
        )
    )

    assert "n/a" in rendered
