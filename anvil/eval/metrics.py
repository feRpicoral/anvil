"""Extraction-specific eval metrics.

JSON-validity gate first (a model that emits unparseable output is unusable
regardless of field accuracy), then per-field scoring on the parsed payload:

  - List fields (parties, termination triggers, confidentiality carveouts)
    → set-based precision/recall/F1 after normalization.
  - Scalar fields (effective_date, governing_law, jurisdiction, term shape,
    indemnification cap, dispute forum/venue/rules) → exact match in [0, 1]
    after normalization. None-vs-None is a match.

`score_extraction` returns a flat per-field dict so callers can macro-
average, per-field-break-down, or compute retention vs a baseline.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from anvil.data.schema import (
    Confidentiality,
    ContractExtraction,
    DisputeResolution,
    Indemnification,
    Party,
    Term,
    Termination,
)

_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class JsonValidityResult:
    """Outcome of validating a model output against `ContractExtraction`."""

    valid: bool
    reason: str | None = None


def validate_json_output(text: str) -> JsonValidityResult:
    """Try to parse `text` as a `ContractExtraction`."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return JsonValidityResult(valid=False, reason=f"json_decode: {exc.msg}")
    if not isinstance(payload, dict):
        return JsonValidityResult(valid=False, reason="not_a_json_object")
    try:
        ContractExtraction.model_validate(payload)
    except ValidationError as exc:
        return JsonValidityResult(valid=False, reason=f"schema: {exc.error_count()} errors")
    return JsonValidityResult(valid=True)


def json_validity_rate(outputs: Iterable[str]) -> float:
    """Fraction of `outputs` that parse against `ContractExtraction`.

    Returns 0.0 for an empty iterable so a caller plotting a zero bar
    doesn't blow up on an empty eval split.
    """
    total = 0
    valid = 0
    for output in outputs:
        total += 1
        if validate_json_output(output).valid:
            valid += 1
    return valid / total if total > 0 else 0.0


def score_extraction(
    predicted: ContractExtraction,
    gold: ContractExtraction,
) -> dict[str, float]:
    """Per-field scores comparing `predicted` against `gold`. Each in [0, 1]."""
    return {
        "parties": _set_f1(_party_set(predicted.parties), _party_set(gold.parties)),
        "effective_date": _exact(predicted.effective_date, gold.effective_date),
        "term": _term_score(predicted.term, gold.term),
        "governing_law": _normalized_exact(predicted.governing_law, gold.governing_law),
        "jurisdiction": _normalized_exact(predicted.jurisdiction, gold.jurisdiction),
        "confidentiality": _confidentiality_score(predicted.confidentiality, gold.confidentiality),
        "termination_triggers": _set_f1(
            _normalized_set(predicted.termination.triggers),
            _normalized_set(gold.termination.triggers),
        ),
        "termination_notice_days": _exact(
            predicted.termination.notice_days, gold.termination.notice_days
        ),
        "indemnification": _indemnification_score(predicted.indemnification, gold.indemnification),
        "dispute_forum": _exact(predicted.dispute_resolution.forum, gold.dispute_resolution.forum),
        "dispute_venue": _normalized_exact(
            predicted.dispute_resolution.venue, gold.dispute_resolution.venue
        ),
    }


def macro_average(scores: dict[str, float]) -> float:
    """Mean of `scores.values()`; 0.0 for empty input."""
    if not scores:
        return 0.0
    return sum(scores.values()) / len(scores)


def aggregate_scores(per_sample: list[dict[str, float]]) -> dict[str, float]:
    """Mean each field across samples.

    Fields missing from a sample contribute 0.0 to keep the average bounded;
    in practice `score_extraction` always returns the same field set so
    that path is defensive, not a normal code path.
    """
    if not per_sample:
        return {}
    fields: set[str] = set()
    for scores in per_sample:
        fields.update(scores.keys())
    n = len(per_sample)
    return {field: sum(scores.get(field, 0.0) for scores in per_sample) / n for field in fields}


def _normalize(value: str | None) -> str:
    if value is None:
        return ""
    return _WHITESPACE_RE.sub(" ", value.strip().lower())


def _normalized_set(values: Iterable[str]) -> set[str]:
    return {_normalize(v) for v in values if v is not None}


def _party_set(parties: Iterable[Party]) -> set[str]:
    return {f"{_normalize(p.name)}|{p.role}" for p in parties}


def _exact(predicted: Any, gold: Any) -> float:
    return 1.0 if predicted == gold else 0.0


def _normalized_exact(predicted: str | None, gold: str | None) -> float:
    return 1.0 if _normalize(predicted) == _normalize(gold) else 0.0


def _set_f1(predicted: set[str], gold: set[str]) -> float:
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    tp = len(predicted & gold)
    if tp == 0:
        return 0.0
    precision = tp / len(predicted)
    recall = tp / len(gold)
    return (2 * precision * recall) / (precision + recall)


def _term_score(predicted: Term, gold: Term) -> float:
    matches = (
        predicted.duration_months == gold.duration_months,
        predicted.is_perpetual == gold.is_perpetual,
        predicted.auto_renew == gold.auto_renew,
        predicted.renewal_notice_days == gold.renewal_notice_days,
    )
    return sum(1 for m in matches if m) / len(matches)


def _confidentiality_score(
    predicted: Confidentiality | None,
    gold: Confidentiality | None,
) -> float:
    if predicted is None and gold is None:
        return 1.0
    if predicted is None or gold is None:
        return 0.0
    duration_match = predicted.duration_months == gold.duration_months
    carveouts_f1 = _set_f1(
        _normalized_set(predicted.carveouts),
        _normalized_set(gold.carveouts),
    )
    return (duration_match + carveouts_f1) / 2


def _indemnification_score(
    predicted: Indemnification | None,
    gold: Indemnification | None,
) -> float:
    if predicted is None and gold is None:
        return 1.0
    if predicted is None or gold is None:
        return 0.0
    usd_match = predicted.cap_usd == gold.cap_usd
    multiplier_match = predicted.cap_multiplier == gold.cap_multiplier
    return (usd_match + multiplier_match) / 2


def _dispute_score(
    predicted: DisputeResolution,
    gold: DisputeResolution,
) -> float:
    forum_match = predicted.forum == gold.forum
    venue_match = _normalized_exact(predicted.venue, gold.venue)
    rules_match = _normalized_exact(predicted.governing_rules, gold.governing_rules)
    return (forum_match + venue_match + rules_match) / 3


def _termination_score(
    predicted: Termination,
    gold: Termination,
) -> float:
    triggers_f1 = _set_f1(_normalized_set(predicted.triggers), _normalized_set(gold.triggers))
    notice_match = predicted.notice_days == gold.notice_days
    cure_match = predicted.cure_period_days == gold.cure_period_days
    return (triggers_f1 + notice_match + cure_match) / 3
