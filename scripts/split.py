"""Split curated records into train/val/test and write messages JSONL."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from anvil.data.format import write_jsonl
from anvil.data.splits import split_records
from scripts.synth import load_results


def run(args: argparse.Namespace) -> int:
    records = load_results(args.input)
    splits = split_records(
        records,
        ratios=tuple(args.ratios),
        seed=args.seed,
        verify=not args.allow_overlap,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "train": write_jsonl(splits.train, args.output_dir / "train.jsonl"),
        "val": write_jsonl(splits.val, args.output_dir / "val.jsonl"),
        "test": write_jsonl(splits.test, args.output_dir / "test.jsonl"),
    }
    summary = ", ".join(f"{name}={n}" for name, n in counts.items())
    print(f"split: wrote {summary} → {args.output_dir}", file=sys.stderr)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split curated records and format messages JSONL.")
    parser.add_argument("--input", type=Path, required=True, help="Curated JSONL.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output dir for splits.")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=(0.8, 0.1, 0.1),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratios that sum to 1.0.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed.")
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Skip the anti-contamination guard (smoke / fixture replay only).",
    )
    return parser.parse_args(argv)


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
