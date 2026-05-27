from __future__ import annotations

import pytest

from anvil.data.parameters import generate_parameters
from anvil.data.prompts import CLAUSE_COMPLEXITIES, CONTRACT_TYPES, ContractType


@pytest.mark.parametrize("contract_type", CONTRACT_TYPES)
def test_generate_parameters_is_deterministic(contract_type: ContractType) -> None:
    a = generate_parameters(contract_type, sample_index=42, base_seed=7)
    b = generate_parameters(contract_type, sample_index=42, base_seed=7)

    assert a == b


@pytest.mark.parametrize("contract_type", CONTRACT_TYPES)
def test_different_seeds_produce_different_parameters(contract_type: ContractType) -> None:
    a = generate_parameters(contract_type, sample_index=0, base_seed=1)
    b = generate_parameters(contract_type, sample_index=0, base_seed=2)

    assert a != b


@pytest.mark.parametrize("contract_type", CONTRACT_TYPES)
def test_parties_are_distinct(contract_type: ContractType) -> None:
    for index in range(50):
        params = generate_parameters(contract_type, sample_index=index, base_seed=0)
        assert params.party_a != params.party_b


@pytest.mark.parametrize("contract_type", CONTRACT_TYPES)
def test_clause_complexity_in_valid_set(contract_type: ContractType) -> None:
    for index in range(20):
        params = generate_parameters(contract_type, sample_index=index, base_seed=0)
        assert params.clause_complexity in CLAUSE_COMPLEXITIES


@pytest.mark.parametrize("contract_type", CONTRACT_TYPES)
def test_dispute_forum_in_valid_set(contract_type: ContractType) -> None:
    valid_forums = {"litigation", "arbitration", "mediation_then_arbitration"}
    for index in range(20):
        params = generate_parameters(contract_type, sample_index=index, base_seed=0)
        assert params.dispute_forum in valid_forums


def test_nda_extras_carry_confidentiality_months() -> None:
    params = generate_parameters("nda", sample_index=0, base_seed=0)

    assert params.extras is not None
    assert "confidentiality_months" in params.extras
    assert params.extras["confidentiality_months"].isdigit()


def test_msa_extras_carry_renewal_and_indemnification() -> None:
    params = generate_parameters("msa", sample_index=0, base_seed=0)

    assert params.extras is not None
    assert params.extras["auto_renew"] in {"true", "false"}
    assert "renewal_notice_days" in params.extras
    assert params.extras["indemnification_cap"]


def test_license_extras_carry_term_kind() -> None:
    params = generate_parameters("license", sample_index=0, base_seed=0)

    assert params.extras is not None
    assert params.extras["term_kind"] in {"perpetual", "fixed-term", "subscription"}
    assert params.extras["term_description"]
    assert params.extras["indemnification_cap"]


def test_effective_date_is_iso() -> None:
    params = generate_parameters("nda", sample_index=0, base_seed=0)
    year, month, day = params.effective_date.split("-")

    assert year in {"2024", "2025", "2026"}
    assert 1 <= int(month) <= 12
    assert 1 <= int(day) <= 28


def test_diversity_across_samples() -> None:
    party_a_set: set[str] = set()
    governing_law_set: set[str] = set()
    edge_case_set: set[str] = set()
    for index in range(60):
        params = generate_parameters("nda", sample_index=index, base_seed=0)
        party_a_set.add(params.party_a)
        governing_law_set.add(params.governing_law)
        edge_case_set.add(params.edge_case)

    assert len(party_a_set) >= 10, "party_a should vary across samples"
    assert len(governing_law_set) >= 5, "governing_law should vary"
    assert len(edge_case_set) >= 3, "edge_case should vary"


def test_term_months_can_be_none_for_perpetual() -> None:
    seen_perpetual = False
    for index in range(200):
        params = generate_parameters("license", sample_index=index, base_seed=0)
        if params.term_months is None:
            seen_perpetual = True
            break

    assert seen_perpetual
