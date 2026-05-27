"""Synthesis interface for contract training data.

`StructuredGenerator` is the protocol every backend implements (OpenAI,
Anthropic, recorded fixtures). The smoke path uses `FixtureGenerator` to
replay pre-recorded outputs so `make data-smoke` runs end-to-end without an
API key and without spending money. Real API generators land in a follow-up
PR; until then this module owns the abstraction and the fixture replay.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast, runtime_checkable

from pydantic import ValidationError

from anvil.data.prompts import ContractType
from anvil.data.schema import ContractExtraction, contract_extraction_json_schema

_SYNTHESIS_RESPONSE_NAME = "contract_synthesis"


@dataclass(frozen=True)
class GenerationResult:
    """A single synthesis output paired with its accounting metadata."""

    contract_type: ContractType
    contract_text: str
    extraction: dict[str, Any]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    backend: str


@runtime_checkable
class StructuredGenerator(Protocol):
    """Anything that can produce a `GenerationResult` for a prompt pair.

    Implementations stay stateless across calls; cost tracking is the
    caller's job (synthesizer or script).
    """

    async def generate(
        self,
        contract_type: ContractType,
        system_prompt: str,
        user_prompt: str,
        seed: int,
    ) -> GenerationResult: ...


def synthesis_response_schema() -> dict[str, Any]:
    """Wrapper JSON schema for OpenAI strict structured outputs.

    Returns a schema with `contract_text` and `extraction` as required
    siblings; the inner extraction schema is the strict variant that
    already marks every property required.
    """
    extraction_schema = contract_extraction_json_schema()
    extraction_defs = extraction_schema.pop("$defs", None)
    schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "contract_text": {
                "type": "string",
                "description": "The contract itself, in plain Markdown.",
            },
            "extraction": extraction_schema,
        },
        "required": ["contract_text", "extraction"],
    }
    if extraction_defs is not None:
        schema["$defs"] = extraction_defs
    return {
        "name": _SYNTHESIS_RESPONSE_NAME,
        "strict": True,
        "schema": schema,
    }


class FixtureGenerator:
    """Replay pre-recorded synthesis outputs from disk.

    Used by the smoke pipeline to produce N samples deterministically with
    zero API cost. Fixtures are cycled by `(contract_type, seed % count)`,
    so callers can request arbitrary batch sizes against a small fixture
    set and still get reproducible draws.
    """

    def __init__(self, fixtures_dir: Path) -> None:
        if not fixtures_dir.is_dir():
            raise FileNotFoundError(f"fixtures dir not found: {fixtures_dir}")
        self._fixtures_dir = fixtures_dir
        self._by_type = _index_fixtures(fixtures_dir)
        if not self._by_type:
            raise ValueError(f"no fixtures found under {fixtures_dir}")

    @property
    def contract_types(self) -> tuple[ContractType, ...]:
        return tuple(sorted(self._by_type.keys()))

    def count(self, contract_type: ContractType) -> int:
        return len(self._by_type.get(contract_type, []))

    async def generate(
        self,
        contract_type: ContractType,
        system_prompt: str,
        user_prompt: str,
        seed: int,
    ) -> GenerationResult:
        del system_prompt, user_prompt  # consumed by real backends, not the replay
        candidates = self._by_type.get(contract_type)
        if not candidates:
            raise KeyError(f"no fixtures recorded for contract_type={contract_type!r}")
        record = candidates[seed % len(candidates)]
        return GenerationResult(
            contract_type=contract_type,
            contract_text=record["contract_text"],
            extraction=record["extraction"],
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            backend="fixture",
        )


def _index_fixtures(fixtures_dir: Path) -> dict[ContractType, list[dict[str, Any]]]:
    by_type: dict[ContractType, list[dict[str, Any]]] = {}
    for path in sorted(fixtures_dir.glob("*.json")):
        record = _load_fixture(path)
        contract_type = record["contract_type"]
        by_type.setdefault(contract_type, []).append(record)
    return by_type


def _load_fixture(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name}: fixture must be a JSON object")
    record = cast(dict[str, Any], raw)
    contract_type = record.get("contract_type")
    if contract_type not in ("nda", "msa", "license"):
        raise ValueError(f"{path.name}: contract_type must be nda/msa/license")
    if "contract_text" not in record or "extraction" not in record:
        raise ValueError(f"{path.name}: missing contract_text or extraction")
    contract_text = record.get("contract_text")
    if not isinstance(contract_text, str):
        raise ValueError(f"{path.name}: contract_text must be a string")
    extraction = record.get("extraction")
    if not isinstance(extraction, dict):
        raise ValueError(f"{path.name}: extraction must be an object")
    try:
        validated = ContractExtraction.model_validate(extraction)
    except ValidationError as exc:
        raise ValueError(f"{path.name}: invalid extraction") from exc
    return {
        "contract_type": cast(ContractType, contract_type),
        "contract_text": contract_text,
        "extraction": validated.model_dump(mode="json"),
    }
