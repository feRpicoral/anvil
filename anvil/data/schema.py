"""Pydantic schema for structured extraction from legal contracts.

The schema covers eight critical fields: parties, effective date, term,
governing law, confidentiality, termination, indemnification, and dispute
resolution. It doubles as the JSON schema fed to OpenAI structured outputs
during synthesis (`response_format={"type": "json_schema", "strict": true}`)
and as the validator at eval time.
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

StrictStr = Annotated[str, Field(strict=True)]
NonEmptyStr = Annotated[str, Field(strict=True, min_length=1)]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
NonNegativeFloat = Annotated[float, Field(strict=True, ge=0)]

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
    """Base that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")


class Party(StrictModel):
    name: NonEmptyStr = Field(..., description="Legal name as written in the contract.")
    role: PartyRole = Field(..., description="Role this party plays in the agreement.")


class Term(StrictModel):
    duration_months: NonNegativeInt | None = Field(
        ...,
        description="Initial term length in months. Null if perpetual or at-will.",
    )
    is_perpetual: bool = Field(
        ...,
        strict=True,
        description="True if the contract has no fixed end date.",
    )
    auto_renew: bool = Field(
        ...,
        strict=True,
        description="True if the contract auto-renews unless cancelled.",
    )
    renewal_notice_days: NonNegativeInt | None = Field(
        ...,
        description="Days of notice required to prevent auto-renewal. Null when not applicable.",
    )

    @model_validator(mode="after")
    def validate_term_consistency(self) -> Term:
        if self.is_perpetual and self.duration_months is not None:
            raise ValueError("perpetual terms cannot also have a fixed duration")
        if not self.auto_renew and self.renewal_notice_days is not None:
            raise ValueError("renewal notice days require auto-renewal")
        return self


class Confidentiality(StrictModel):
    scope: NonEmptyStr = Field(
        ...,
        description="Narrative description of what is treated as confidential.",
    )
    duration_months: NonNegativeInt | None = Field(
        ...,
        description="Duration of the confidentiality obligation. Null if perpetual.",
    )
    carveouts: list[StrictStr] = Field(
        ...,
        description="Standard carveouts such as public-domain or independently-developed information.",
    )


class Termination(StrictModel):
    triggers: list[StrictStr] = Field(
        ...,
        description="Events that allow termination (breach, convenience, insolvency).",
    )
    notice_days: NonNegativeInt | None = Field(
        ...,
        description="Required notice period for termination.",
    )
    cure_period_days: NonNegativeInt | None = Field(
        ...,
        description="Cure period after notice of breach.",
    )


class Indemnification(StrictModel):
    scope: NonEmptyStr = Field(
        ...,
        description="Narrative description of indemnification scope.",
    )
    cap_usd: NonNegativeInt | None = Field(
        ...,
        description="Hard-dollar liability cap in USD. Null if uncapped or expressed as a multiplier.",
    )
    cap_multiplier: NonNegativeFloat | None = Field(
        ...,
        description="Liability cap expressed as a multiplier of contract value. Null if not used.",
    )

    @model_validator(mode="after")
    def validate_cap_consistency(self) -> Indemnification:
        if self.cap_usd is not None and self.cap_multiplier is not None:
            raise ValueError("indemnification cap cannot be both USD and multiplier")
        return self


class DisputeResolution(StrictModel):
    forum: DisputeForum = Field(..., description="Primary forum for dispute resolution.")
    venue: StrictStr | None = Field(
        ...,
        description="Geographic venue (e.g. 'Wilmington, Delaware').",
    )
    governing_rules: StrictStr | None = Field(
        ...,
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
        ...,
        description="Effective date of the contract. Null if not stated.",
    )
    term: Term
    governing_law: StrictStr | None = Field(
        ...,
        description="Body of law that governs interpretation (e.g. 'State of Delaware').",
    )
    jurisdiction: StrictStr | None = Field(
        ...,
        description="Courts with authority over disputes when forum is litigation.",
    )
    confidentiality: Confidentiality | None = Field(
        ...,
        description="Null if the contract has no confidentiality clause.",
    )
    termination: Termination
    indemnification: Indemnification | None = Field(
        ...,
        description="Null if the contract has no indemnification clause.",
    )
    dispute_resolution: DisputeResolution


def contract_extraction_json_schema() -> dict[str, Any]:
    """Return a JSON schema compatible with OpenAI strict Structured Outputs."""
    schema = ContractExtraction.model_json_schema()
    _require_all_object_properties(schema)
    return schema


def _require_all_object_properties(value: Any) -> None:
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            value["required"] = list(properties)
        for child in value.values():
            _require_all_object_properties(child)
    elif isinstance(value, list):
        for child in value:
            _require_all_object_properties(child)
