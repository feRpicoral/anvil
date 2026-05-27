from __future__ import annotations

import pytest

from anvil.data.prompts import ContractType
from anvil.data.splits import (
    SplitContaminationError,
    Splits,
    normalized_text_hash,
    split_records,
    verify_no_overlap,
)
from anvil.data.synthesis import GenerationResult


def _record(contract_type: ContractType, contract_text: str) -> GenerationResult:
    return GenerationResult(
        contract_type=contract_type,
        contract_text=contract_text,
        extraction={},
        input_tokens=0,
        output_tokens=0,
        cost_usd=0.0,
        backend="test",
    )


def _make_records(per_type: dict[ContractType, int]) -> list[GenerationResult]:
    records: list[GenerationResult] = []
    for contract_type, n in per_type.items():
        for i in range(n):
            records.append(_record(contract_type, f"{contract_type} contract body {i}"))
    return records


def test_normalized_text_hash_ignores_whitespace_and_case() -> None:
    a = normalized_text_hash("Hello   World")
    b = normalized_text_hash("hello\n\nworld")

    assert a == b


def test_normalized_text_hash_distinguishes_content() -> None:
    a = normalized_text_hash("Hello world")
    b = normalized_text_hash("Goodbye world")

    assert a != b


def test_split_records_returns_empty_splits_for_empty_input() -> None:
    splits = split_records([])

    assert splits.train == ()
    assert splits.val == ()
    assert splits.test == ()


def test_split_records_partitions_at_default_ratios() -> None:
    records = _make_records({"nda": 10, "msa": 10, "license": 10})

    splits = split_records(records)

    counts = splits.counts()
    assert counts["train"] + counts["val"] + counts["test"] == 30
    assert counts["train"] == 24
    assert counts["val"] == 3
    assert counts["test"] == 3


def test_split_records_keeps_each_contract_type_represented() -> None:
    records = _make_records({"nda": 5, "msa": 5, "license": 5})

    splits = split_records(records, seed=0)

    train_types = {r.contract_type for r in splits.train}
    val_types = {r.contract_type for r in splits.val}
    test_types = {r.contract_type for r in splits.test}
    assert train_types == {"nda", "msa", "license"}
    assert val_types == {"nda", "msa", "license"}
    assert test_types == {"nda", "msa", "license"}


def test_split_records_is_deterministic_for_same_seed() -> None:
    records = _make_records({"nda": 10, "msa": 10, "license": 10})

    a = split_records(records, seed=7)
    b = split_records(records, seed=7)

    assert [r.contract_text for r in a.train] == [r.contract_text for r in b.train]
    assert [r.contract_text for r in a.test] == [r.contract_text for r in b.test]


def test_split_records_differs_across_seeds() -> None:
    records = _make_records({"nda": 30, "msa": 30, "license": 30})

    a = split_records(records, seed=1)
    b = split_records(records, seed=2)

    assert [r.contract_text for r in a.train] != [r.contract_text for r in b.train]


def test_split_records_rejects_ratios_that_do_not_sum_to_one() -> None:
    records = _make_records({"nda": 4})

    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        split_records(records, ratios=(0.5, 0.5, 0.5))


def test_split_records_rejects_negative_ratios() -> None:
    records = _make_records({"nda": 4})

    with pytest.raises(ValueError, match="non-negative"):
        split_records(records, ratios=(1.2, -0.1, -0.1))


def test_split_records_rejects_wrong_ratio_count() -> None:
    records = _make_records({"nda": 4})

    with pytest.raises(ValueError, match="exactly three"):
        split_records(records, ratios=(0.8, 0.2))

    with pytest.raises(ValueError, match="exactly three"):
        split_records(records, ratios=(0.8, 0.1, 0.1, 0.0))


def test_verify_no_overlap_raises_when_record_duplicated() -> None:
    shared = _record("nda", "shared contract body")
    splits = Splits(train=(shared,), val=(shared,), test=())

    with pytest.raises(SplitContaminationError, match="train ∩ val"):
        verify_no_overlap(splits)


def test_verify_no_overlap_accepts_disjoint_splits() -> None:
    a = _record("nda", "first body")
    b = _record("nda", "second body")
    c = _record("nda", "third body")
    splits = Splits(train=(a,), val=(b,), test=(c,))

    verify_no_overlap(splits)


def test_split_records_after_shuffle_remains_contamination_free() -> None:
    records = _make_records({"nda": 50, "msa": 50, "license": 50})

    splits = split_records(records, seed=42)

    verify_no_overlap(splits)


def _all_records(splits: Splits) -> list[GenerationResult]:
    return [*splits.train, *splits.val, *splits.test]


def test_split_records_does_not_drop_or_duplicate_inputs() -> None:
    records = _make_records({"nda": 7, "msa": 11, "license": 5})

    splits = split_records(records, seed=3)

    assert len(_all_records(splits)) == len(records)
    assert {r.contract_text for r in _all_records(splits)} == {r.contract_text for r in records}


def test_splits_counts_helper() -> None:
    a = _record("nda", "a")
    b = _record("nda", "b")
    splits = Splits(train=(a,), val=(b,), test=())

    assert splits.counts() == {"train": 1, "val": 1, "test": 0}


def test_split_records_rejects_buckets_too_small_for_nonzero_splits() -> None:
    records = _make_records({"nda": 2})

    with pytest.raises(ValueError, match="not enough records"):
        split_records(records)


def test_split_records_allows_small_buckets_when_only_two_splits_are_nonzero() -> None:
    records = _make_records({"nda": 2})

    splits = split_records(records, ratios=(0.5, 0.5, 0.0))

    assert splits.counts() == {"train": 1, "val": 1, "test": 0}
