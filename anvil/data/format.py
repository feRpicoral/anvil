"""Format curated records as chat-template messages JSONL for SFT.

We emit the canonical `{"messages": [...]}` shape that TRL `SFTTrainer`,
Unsloth, and most other fine-tuning frameworks accept as input. Each
framework then applies the base model's chat template at train time, so
we don't bake any model-specific tokens in the file.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from anvil.data.synthesis import GenerationResult

EXTRACTION_INSTRUCTION = (
    "Extract the structured contract fields from the document below. "
    "Return a single JSON object that conforms exactly to the contract-extraction schema. "
    "Include every schema field; use null where the contract does not specify a value."
)


def to_messages(record: GenerationResult) -> list[dict[str, str]]:
    """Render one record as a three-turn chat-format conversation."""
    return [
        {"role": "system", "content": EXTRACTION_INSTRUCTION},
        {"role": "user", "content": record.contract_text},
        {"role": "assistant", "content": _serialize_extraction(record.extraction)},
    ]


def write_jsonl(records: Sequence[GenerationResult], path: Path) -> int:
    """Write `records` as messages JSONL to `path`. Returns the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            messages = to_messages(record)
            fh.write(json.dumps({"messages": messages}, ensure_ascii=False))
            fh.write("\n")
            n += 1
    return n


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield each row of a messages JSONL file as a dict."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            yield json.loads(line)


def _serialize_extraction(extraction: dict[str, Any]) -> str:
    return json.dumps(extraction, ensure_ascii=False, sort_keys=True)
