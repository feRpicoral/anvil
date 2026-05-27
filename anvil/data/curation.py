"""Curation utilities for synthesized contract data.

Each function is small and pure so the synthesis pipeline can compose them
incrementally and emit per-record rejection reasons for the audit log.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from rapidfuzz import fuzz

from anvil.data.schema import ContractExtraction

_DEFAULT_MIN_CHARS = 500
_DEFAULT_MAX_CHARS = 50_000
_DEFAULT_DUP_THRESHOLD = 0.9
_DEFAULT_DUP_PREFIX_CHARS = 500
_ENGLISH_ALLOWED_PUNCT = frozenset(".,;:'\"-()[]{}!?/$%&*+=<>@#")
_ENGLISH_RATIO_THRESHOLD = 0.95


@dataclass(frozen=True)
class CurationOutcome:
    """The result of curating a single synthesis record."""

    accepted: bool
    reason: str | None = None
    extraction: ContractExtraction | None = None


def validate_extraction(payload: dict[str, Any]) -> CurationOutcome:
    """Validate a raw extraction payload against `ContractExtraction`."""
    try:
        extraction = ContractExtraction.model_validate(payload)
    except ValidationError as exc:
        return CurationOutcome(accepted=False, reason=f"schema_invalid: {exc.error_count()} errors")
    return CurationOutcome(accepted=True, extraction=extraction)


def length_in_range(
    text: str,
    min_chars: int = _DEFAULT_MIN_CHARS,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> bool:
    """Return True if `text` length sits within `[min_chars, max_chars]`."""
    return min_chars <= len(text) <= max_chars


def is_mostly_english(text: str, threshold: float = _ENGLISH_RATIO_THRESHOLD) -> bool:
    """Sanity check: at least `threshold` of characters are ASCII-friendly.

    Catches obvious language drift (Cyrillic, CJK, emoji-heavy) without paying
    the cost of a full language-detection model. Imperfect by design: short
    Latin-1 strings can slip through and very short English strings can fail.
    """
    if not text:
        return False
    allowed = sum(
        1
        for c in text
        if c.isascii() and (c.isalnum() or c.isspace() or c in _ENGLISH_ALLOWED_PUNCT)
    )
    return allowed / len(text) >= threshold


def find_near_duplicate_groups(
    texts: list[str],
    threshold: float = _DEFAULT_DUP_THRESHOLD,
    prefix_chars: int = _DEFAULT_DUP_PREFIX_CHARS,
) -> list[set[int]]:
    """Group indices of near-duplicate texts using Levenshtein ratio on prefixes.

    Pairwise O(N^2). Suitable for N up to a few thousand at the default prefix
    length; for larger corpora a MinHash + LSH path is the documented swap-in.

    Args:
        texts: Texts to compare.
        threshold: Ratio in [0, 1]; pairs at or above are grouped.
        prefix_chars: Maximum characters per text used in the comparison.

    Returns:
        Groups of indices where each group has at least two members. An index
        appears in at most one group; the first member of each group acts as
        the canonical survivor that downstream dedup would keep.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if prefix_chars <= 0:
        raise ValueError("prefix_chars must be positive")

    parent = list(range(len(texts)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    prefixes = [t[:prefix_chars] for t in texts]
    cutoff = threshold * 100
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if fuzz.ratio(prefixes[i], prefixes[j]) >= cutoff:
                union(i, j)

    groups_by_root: dict[int, set[int]] = {}
    for index in range(len(texts)):
        root = find(index)
        groups_by_root.setdefault(root, set()).add(index)
    return [group for group in groups_by_root.values() if len(group) > 1]
