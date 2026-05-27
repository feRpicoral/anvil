"""Pydantic schema for structured extraction from legal contracts.

The schema covers eight critical fields: parties, effective date, term,
governing law, confidentiality, termination, indemnification, and dispute
resolution. It doubles as the JSON schema fed to OpenAI structured outputs
during synthesis (`response_format={"type": "json_schema", "strict": true}`)
and as the validator at eval time.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PartyRole = Literal[
    "disclosing_party",
    "receiving_party",
    "buyer",
    "seller",
    "vendor",
    "client",
    "licensor",
    "licensee",
    "other",
]

DisputeForum = Literal[
    "arbitration",
    "litigation",
    "mediation_then_arbitration",
    "other",
]


class StrictModel(BaseModel):
    """Base with extra='forbid' so unknown fields fail validation."""

    model_config = ConfigDict(extra="forbid")


class Party(StrictModel):
    name: str = Field(..., min_length=1, description="Legal name as written in the contract.")
    role: PartyRole = Field(..., description="Role this party plays in the agreement.")


class Term(StrictModel):
    duration_months: int | None = Field(
        None,
        ge=0,
        description="Initial term length in months. Null if perpetual or at-will.",
    )
    is_perpetual: bool = Field(False, description="True if the contract has no fixed end date.")
    auto_renew: bool = Field(
        False, description="True if the contract auto-renews unless cancelled."
    )
    renewal_notice_days: int | None = Field(
        None,
        ge=0,
        description="Days of notice required to prevent auto-renewal. Null when not applicable.",
    )


class Confidentiality(StrictModel):
    scope: str = Field(
        ...,
        min_length=1,
        description="Narrative description of what is treated as confidential.",
    )
    duration_months: int | None = Field(
        None,
        ge=0,
        description="Duration of the confidentiality obligation. Null if perpetual.",
    )
    carveouts: list[str] = Field(
        default_factory=list,
        description="Standard carveouts such as public-domain or independently-developed information.",
    )


class Termination(StrictModel):
    triggers: list[str] = Field(
        default_factory=list,
        description="Events that allow termination (breach, convenience, insolvency).",
    )
    notice_days: int | None = Field(
        None,
        ge=0,
        description="Required notice period for termination.",
    )
    cure_period_days: int | None = Field(
        None,
        ge=0,
        description="Cure period after notice of breach.",
    )


class Indemnification(StrictModel):
    scope: str = Field(
        ...,
        min_length=1,
        description="Narrative description of indemnification scope.",
    )
    cap_usd: int | None = Field(
        None,
        ge=0,
        description="Hard-dollar liability cap in USD. Null if uncapped or expressed as a multiplier.",
    )
    cap_multiplier: float | None = Field(
        None,
        ge=0,
        description="Liability cap expressed as a multiplier of contract value. Null if not used.",
    )


class DisputeResolution(StrictModel):
    forum: DisputeForum = Field(..., description="Primary forum for dispute resolution.")
    venue: str | None = Field(
        None,
        description="Geographic venue (e.g. 'Wilmington, Delaware').",
    )
    governing_rules: str | None = Field(
        None,
        description="Procedural rules (AAA Commercial, JAMS, ICC).",
    )


class ContractExtraction(StrictModel):
    """Structured extraction from a single legal contract."""

    parties: list[Party] = Field(
        ...,
        min_length=2,
        description="All contracting parties; at least two are required.",
    )
    effective_date: date | None = Field(
        None,
        description="Effective date of the contract. Null if not stated.",
    )
    term: Term
    governing_law: str | None = Field(
        None,
        description="Body of law that governs interpretation (e.g. 'State of Delaware').",
    )
    jurisdiction: str | None = Field(
        None,
        description="Courts with authority over disputes when forum is litigation.",
    )
    confidentiality: Confidentiality | None = Field(
        None,
        description="Null if the contract has no confidentiality clause.",
    )
    termination: Termination
    indemnification: Indemnification | None = Field(
        None,
        description="Null if the contract has no indemnification clause.",
    )
    dispute_resolution: DisputeResolution
