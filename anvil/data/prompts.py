"""Prompt templates for synthesizing legal contracts with structured outputs.

Each contract type pairs a system prompt (drafting persona + output contract)
with a parameterized user prompt that varies across diversity axes
(jurisdiction, term length, edge-case flags). Both prompts are plain Python
strings rendered with `str.format`; we avoid Jinja to keep the runtime
dependency surface narrow.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

ContractType = Literal["nda", "msa", "license"]
ClauseComplexity = Literal["minimal", "standard", "comprehensive"]

CONTRACT_TYPES: tuple[ContractType, ...] = ("nda", "msa", "license")
CLAUSE_COMPLEXITIES: tuple[ClauseComplexity, ...] = ("minimal", "standard", "comprehensive")

_SYSTEM_PROMPT = """You are an expert legal-document drafter producing synthetic training data.

For each request you return a single JSON object with two top-level keys:

  - "contract_text": the contract itself, in plain Markdown, 600-2500 words. Use realistic legal phrasing, numbered sections, and defined terms.
  - "extraction": a structured extraction that conforms to the supplied JSON schema and is consistent with "contract_text" down to every party name, date, dollar figure, and duration.

Rules:
  - Every fact in "extraction" must appear verbatim or in clearly-derived form in "contract_text". Inconsistency is a defect.
  - Use only the party roles and dispute forums enumerated in the schema.
  - Do not invent commentary, preamble, or trailing prose outside the JSON object.
"""

_USER_PROMPTS: dict[ContractType, str] = {
    "nda": (
        "Draft a {clause_complexity} mutual non-disclosure agreement.\n\n"
        "  - Disclosing party: {party_a} ({party_a_jurisdiction})\n"
        "  - Receiving party: {party_b} ({party_b_jurisdiction})\n"
        "  - Effective date: {effective_date}\n"
        "  - Agreement term: {term_months} months\n"
        "  - Confidentiality survival: {confidentiality_months} months\n"
        "  - Governing law: {governing_law}\n"
        "  - Dispute forum: {dispute_forum}\n"
        "  - Edge-case flag: {edge_case}\n"
    ),
    "msa": (
        "Draft a {clause_complexity} master services agreement.\n\n"
        "  - Vendor: {party_a} ({party_a_jurisdiction})\n"
        "  - Client: {party_b} ({party_b_jurisdiction})\n"
        "  - Effective date: {effective_date}\n"
        "  - Initial term: {term_months} months, auto-renewing: {auto_renew}\n"
        "  - Renewal notice: {renewal_notice_days} days\n"
        "  - Governing law: {governing_law}\n"
        "  - Indemnification cap: {indemnification_cap}\n"
        "  - Dispute forum: {dispute_forum}\n"
        "  - Edge-case flag: {edge_case}\n"
    ),
    "license": (
        "Draft a {clause_complexity} {term_kind} software-license agreement.\n\n"
        "  - Licensor: {party_a} ({party_a_jurisdiction})\n"
        "  - Licensee: {party_b} ({party_b_jurisdiction})\n"
        "  - Effective date: {effective_date}\n"
        "  - Term: {term_description}\n"
        "  - Governing law: {governing_law}\n"
        "  - IP-indemnification cap: {indemnification_cap}\n"
        "  - Dispute forum: {dispute_forum}\n"
        "  - Edge-case flag: {edge_case}\n"
    ),
}


@dataclass(frozen=True)
class PromptParameters:
    """The variation bundle passed to a contract-specific user prompt.

    Fields that do not apply to a given contract type are ignored by the
    relevant template. `extras` carries contract-specific overrides.
    """

    contract_type: ContractType
    party_a: str
    party_a_jurisdiction: str
    party_b: str
    party_b_jurisdiction: str
    effective_date: str
    term_months: int | None
    governing_law: str
    dispute_forum: str
    edge_case: str
    clause_complexity: ClauseComplexity = "standard"
    extras: Mapping[str, str] | None = None


def system_prompt() -> str:
    """Return the shared drafting-persona system prompt."""
    return _SYSTEM_PROMPT


def render_user_prompt(parameters: PromptParameters) -> str:
    """Render the user prompt for `parameters.contract_type`.

    Missing template variables raise `KeyError` so synthesis fails fast on a
    misconfigured parameter set rather than emitting partial prompts.
    """
    template = _USER_PROMPTS[parameters.contract_type]
    values: dict[str, str] = {
        "clause_complexity": parameters.clause_complexity,
        "party_a": parameters.party_a,
        "party_a_jurisdiction": parameters.party_a_jurisdiction,
        "party_b": parameters.party_b,
        "party_b_jurisdiction": parameters.party_b_jurisdiction,
        "effective_date": parameters.effective_date,
        "term_months": str(parameters.term_months) if parameters.term_months is not None else "n/a",
        "governing_law": parameters.governing_law,
        "dispute_forum": parameters.dispute_forum,
        "edge_case": parameters.edge_case,
    }
    if parameters.extras:
        values.update(parameters.extras)
    return template.format(**values)
