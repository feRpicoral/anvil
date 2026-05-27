"""Train / val / test splitting with a hash-based anti-contamination guard.

The guard is load-bearing: a single sample appearing in two splits would
silently inflate eval numbers, so any overlap aborts the run rather than
emitting a warning.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from anvil.data.prompts import ContractType
from anvil.data.synthesis import GenerationResult

_WHITESPACE_RE = re.compile(r"\s+")
_DEFAULT_RATIOS = (0.8, 0.1, 0.1)


@dataclass(frozen=True)
class Splits:
    """Three disjoint partitions of generated records."""

    train: tuple[GenerationResult, ...]
    val: tuple[GenerationResult, ...]
    test: tuple[GenerationResult, ...]

    def counts(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}


class SplitContaminationError(RuntimeError):
    """Raised when the same contract appears in more than one split."""


def normalized_text_hash(text: str) -> str:
    """SHA-256 over whitespace-normalized, lowercased contract text.

    Cheap normalization catches the common contamination cases (re-emitted
    samples with cosmetic edits, leading/trailing whitespace, casing
    differences) without being fooled by paragraph-break churn.
    """
    collapsed = _WHITESPACE_RE.sub(" ", text).strip().lower()
    return hashlib.sha256(collapsed.encode("utf-8")).hexdigest()


def split_records(
    records: Sequence[GenerationResult],
    ratios: tuple[float, ...] = _DEFAULT_RATIOS,
    seed: int = 0,
) -> Splits:
    """Stratified shuffle-then-slice split by contract_type.

    Stratification keeps each contract type proportionally represented in
    train/val/test so a smoke eval doesn't accidentally hold out an entire
    class. Ratios sum to 1.0; the validator rejects any other input.
    """
    _validate_ratios(ratios)
    if not records:
        return Splits(train=(), val=(), test=())

    by_type: dict[ContractType, list[GenerationResult]] = defaultdict(list)
    for record in records:
        by_type[record.contract_type].append(record)

    rng = random.Random(seed)
    train: list[GenerationResult] = []
    val: list[GenerationResult] = []
    test: list[GenerationResult] = []

    for contract_type in sorted(by_type.keys()):
        bucket = list(by_type[contract_type])
        rng.shuffle(bucket)
        n = len(bucket)
        n_train, n_val, _n_test = _allocate_counts(n, ratios)
        train.extend(bucket[:n_train])
        val.extend(bucket[n_train : n_train + n_val])
        test.extend(bucket[n_train + n_val :])

    splits = Splits(train=tuple(train), val=tuple(val), test=tuple(test))
    verify_no_overlap(splits)
    return splits


def verify_no_overlap(splits: Splits) -> None:
    """Raise `SplitContaminationError` if any record appears in two splits."""
    train_hashes = _hash_set(splits.train)
    val_hashes = _hash_set(splits.val)
    test_hashes = _hash_set(splits.test)
    overlaps = {
        "train ∩ val": train_hashes & val_hashes,
        "train ∩ test": train_hashes & test_hashes,
        "val ∩ test": val_hashes & test_hashes,
    }
    leaks = {label: hashes for label, hashes in overlaps.items() if hashes}
    if leaks:
        summary = ", ".join(f"{label}: {len(hashes)}" for label, hashes in leaks.items())
        raise SplitContaminationError(f"split contamination detected: {summary}")


def _hash_set(records: Iterable[GenerationResult]) -> set[str]:
    return {normalized_text_hash(record.contract_text) for record in records}


def _validate_ratios(ratios: tuple[float, ...]) -> None:
    if len(ratios) != 3:
        raise ValueError("split ratios must contain exactly three values")
    if any(r < 0 for r in ratios):
        raise ValueError("split ratios must be non-negative")
    total = sum(ratios)
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(f"split ratios must sum to 1.0 (got {total})")


def _allocate_counts(n: int, ratios: tuple[float, ...]) -> tuple[int, int, int]:
    positive_indices = [index for index, ratio in enumerate(ratios) if ratio > 0]
    if n < len(positive_indices):
        raise ValueError(
            f"not enough records ({n}) to populate {len(positive_indices)} non-zero splits"
        )

    targets = [ratio * n for ratio in ratios]
    counts = [1 if index in positive_indices else 0 for index in range(3)]
    remaining = n - sum(counts)
    for _ in range(remaining):
        index = max(range(3), key=lambda i: (targets[i] - counts[i], ratios[i], -i))
        counts[index] += 1
    return counts[0], counts[1], counts[2]
