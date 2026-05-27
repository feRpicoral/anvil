"""Deterministic diversity-axis parameter generator for paid synthesis.

The paid run draws 4k samples; without explicit variation the generator
collapses to a handful of stylistic patterns (cross-batch mode collapse).
This module produces a `PromptParameters` per `(contract_type, sample_index)`
that varies parties, jurisdictions, term length, dispute forum, edge-case
flag, and clause complexity from curated pools.

Seeded by `(base_seed, contract_type, sample_index)` so two runs with the
same base_seed produce identical parameter sequences and the same paid
spend produces the same dataset.
"""

from __future__ import annotations

import random
from typing import Final

from anvil.data.prompts import (
    CLAUSE_COMPLEXITIES,
    ClauseComplexity,
    ContractType,
    PromptParameters,
)

_COMPANY_NAMES: Final[tuple[str, ...]] = (
    "Acme Corp.",
    "Globex Industries LLC",
    "Cloudforge Inc.",
    "Beacon Financial Services",
    "Lyra Technologies Ltd.",
    "Orion Robotics, Inc.",
    "Vega Analytics, Inc.",
    "Polaris Manufacturing Co.",
    "Helios Renewables plc",
    "Aurora Energy Pty Ltd",
    "Stellar Logistics S.A.",
    "Meridian Biotech Corporation",
    "Atlas Aerospace Holdings",
    "Pinnacle Pharma Group",
    "Cascade Software Solutions",
    "Solstice Capital Partners",
    "Halcyon Trading Co.",
    "Borealis Materials Inc.",
    "Nimbus Cloud Services Ltd.",
    "Sentinel Cybersecurity Inc.",
    "Tangent Robotics GmbH",
    "Ember Studios LLC",
    "Quartz Mining Ltd.",
    "Vanguard Industrial Group",
    "Equinox Health Systems",
    "Crescent Telecommunications",
    "Apex Logistics Co.",
    "Brookline Asset Management",
    "Dynasty Foods Inc.",
    "Eclipse Media Holdings",
)

_PARTY_JURISDICTIONS: Final[tuple[str, ...]] = (
    "Delaware",
    "California",
    "New York",
    "Texas",
    "Massachusetts",
    "Illinois",
    "Washington",
    "Ontario, Canada",
    "England and Wales",
    "Singapore",
    "New South Wales, Australia",
    "Germany",
    "Ireland",
    "Switzerland",
    "Netherlands",
)

_GOVERNING_LAW: Final[tuple[str, ...]] = (
    "State of Delaware",
    "State of California",
    "State of New York",
    "Commonwealth of Massachusetts",
    "State of Texas",
    "England and Wales",
    "Republic of Singapore",
    "Federal Republic of Germany",
    "Switzerland",
    "Province of Ontario, Canada",
)

_DISPUTE_FORUMS: Final[tuple[str, ...]] = (
    "litigation",
    "arbitration",
    "mediation_then_arbitration",
)

_TERM_LENGTHS: Final[tuple[int, ...]] = (12, 24, 36, 48, 60)

_EDGE_CASES: Final[tuple[str, ...]] = (
    "none",
    "cross-border data transfer concerns",
    "subsidiary as third-party beneficiary",
    "regulated industry (financial services)",
    "regulated industry (healthcare)",
    "parent guarantee required",
    "non-compete carve-out",
)

_NDA_CONFIDENTIALITY_MONTHS: Final[tuple[int, ...]] = (24, 36, 60, 84, 120)

_MSA_AUTO_RENEW_NOTICES: Final[tuple[int, ...]] = (30, 60, 90, 120)

_INDEMNIFICATION_CAPS: Final[tuple[str, ...]] = (
    "$1,000,000",
    "$5,000,000",
    "$10,000,000",
    "1.5x annual fees",
    "2x annual fees",
    "3x annual fees",
)

_LICENSE_TERM_KINDS: Final[tuple[str, ...]] = ("perpetual", "fixed-term", "subscription")


def generate_parameters(
    contract_type: ContractType,
    sample_index: int,
    base_seed: int = 0,
) -> PromptParameters:
    """Produce a `PromptParameters` deterministic in `(base_seed, type, index)`."""
    rng = random.Random(f"{base_seed}|{contract_type}|{sample_index}")

    party_a = rng.choice(_COMPANY_NAMES)
    party_b = rng.choice([name for name in _COMPANY_NAMES if name != party_a])
    party_a_jurisdiction = rng.choice(_PARTY_JURISDICTIONS)
    party_b_jurisdiction = rng.choice(_PARTY_JURISDICTIONS)
    effective_date = _random_date(rng)
    governing_law = rng.choice(_GOVERNING_LAW)
    dispute_forum = rng.choice(_DISPUTE_FORUMS)
    edge_case = rng.choice(_EDGE_CASES)
    clause_complexity: ClauseComplexity = rng.choice(CLAUSE_COMPLEXITIES)
    term_months, extras = _contract_specific_terms_and_extras(contract_type, rng)

    return PromptParameters(
        contract_type=contract_type,
        party_a=party_a,
        party_a_jurisdiction=party_a_jurisdiction,
        party_b=party_b,
        party_b_jurisdiction=party_b_jurisdiction,
        effective_date=effective_date,
        term_months=term_months,
        governing_law=governing_law,
        dispute_forum=dispute_forum,
        edge_case=edge_case,
        clause_complexity=clause_complexity,
        extras=extras,
    )


def _random_date(rng: random.Random) -> str:
    year = rng.choice([2024, 2025, 2026])
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    return f"{year}-{month:02d}-{day:02d}"


def _contract_specific_terms_and_extras(
    contract_type: ContractType,
    rng: random.Random,
) -> tuple[int | None, dict[str, str]]:
    if contract_type == "nda":
        return rng.choice(_TERM_LENGTHS), {
            "confidentiality_months": str(rng.choice(_NDA_CONFIDENTIALITY_MONTHS))
        }
    if contract_type == "msa":
        auto_renew = rng.choice((True, False))
        return (
            rng.choice(_TERM_LENGTHS),
            {
                "auto_renew": "true" if auto_renew else "false",
                "renewal_notice_days": str(rng.choice(_MSA_AUTO_RENEW_NOTICES))
                if auto_renew
                else "n/a",
                "indemnification_cap": rng.choice(_INDEMNIFICATION_CAPS),
            },
        )
    if contract_type == "license":
        term_kind = rng.choice(_LICENSE_TERM_KINDS)
        if term_kind == "perpetual":
            term_months = None
            term_description = "perpetual, no fixed expiration"
        else:
            term_months = rng.choice(_TERM_LENGTHS)
            term_description = f"{term_months} months"
        return (
            term_months,
            {
                "term_kind": term_kind,
                "term_description": term_description,
                "indemnification_cap": rng.choice(_INDEMNIFICATION_CAPS),
            },
        )
    raise ValueError(f"unknown contract_type: {contract_type!r}")
