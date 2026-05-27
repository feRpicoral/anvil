"""Filter raw synthesis output through validation, length, language, dedup.

Each rejection carries a human-readable reason so an operator can audit
losses without re-running synthesis.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from anvil.data.curation import (
    find_near_duplicate_groups,
    is_mostly_english,
    length_in_range,
    validate_extraction,
)
from anvil.data.synthesis import GenerationResult
from scripts.synth import load_results


@dataclasses.dataclass(frozen=True)
class CurationReport:
    accepted: int
    rejected: int
    reasons: dict[str, int]


def curate(
    records: Sequence[GenerationResult],
    *,
    dedup: bool = True,
) -> tuple[list[GenerationResult], CurationReport]:
    """Apply all filters and (optionally) the dedup pass.

    Dedup is the right default for real synthesized data; the smoke pipeline
    disables it because fixture replay intentionally repeats canonical
    contracts to exercise plumbing at scale without spending money.
    """
    reasons: dict[str, int] = {}
    survivors: list[GenerationResult] = []

    for record in records:
        if not length_in_range(record.contract_text):
            reasons["length"] = reasons.get("length", 0) + 1
            continue
        if not is_mostly_english(record.contract_text):
            reasons["language"] = reasons.get("language", 0) + 1
            continue
        outcome = validate_extraction(record.extraction)
        if not outcome.accepted:
            reasons["schema"] = reasons.get("schema", 0) + 1
            continue
        survivors.append(record)

    if dedup and survivors:
        texts = [record.contract_text for record in survivors]
        duplicates = find_near_duplicate_groups(texts)
        drop_indices: set[int] = set()
        for group in duplicates:
            # Keep the first canonical member; drop the rest.
            members = sorted(group)
            drop_indices.update(members[1:])
        if drop_indices:
            reasons["duplicate"] = reasons.get("duplicate", 0) + len(drop_indices)
            survivors = [r for i, r in enumerate(survivors) if i not in drop_indices]

    report = CurationReport(
        accepted=len(survivors),
        rejected=len(records) - len(survivors),
        reasons=reasons,
    )
    return survivors, report


def write_jsonl(records: Sequence[GenerationResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(dataclasses.asdict(record), ensure_ascii=False))
            fh.write("\n")


def run(args: argparse.Namespace) -> int:
    records = load_results(args.input)
    survivors, report = curate(records, dedup=not args.no_dedup)
    write_jsonl(survivors, args.output)
    reasons = ", ".join(f"{k}={v}" for k, v in sorted(report.reasons.items())) or "(none)"
    print(
        f"curate: accepted {report.accepted}/{len(records)} "
        f"({report.rejected} rejected; reasons: {reasons}) → {args.output}",
        file=sys.stderr,
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate raw synthesis output.")
    parser.add_argument("--input", type=Path, required=True, help="Raw synthesis JSONL.")
    parser.add_argument("--output", type=Path, required=True, help="Curated JSONL.")
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Skip near-duplicate detection (use for fixture-replay smokes).",
    )
    return parser.parse_args(argv)


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
