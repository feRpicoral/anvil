from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from anvil.data.schema import ContractExtraction
from anvil.data.synthesis import (
    FixtureGenerator,
    GenerationResult,
    StructuredGenerator,
    synthesis_response_schema,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "synthesis"


def test_synthesis_response_schema_is_strict() -> None:
    schema = synthesis_response_schema()

    assert schema["name"] == "contract_synthesis"
    assert schema["strict"] is True
    body = schema["schema"]
    assert body["type"] == "object"
    assert body["additionalProperties"] is False
    assert set(body["required"]) == {"contract_text", "extraction"}
    assert body["properties"]["contract_text"]["type"] == "string"
    assert body["properties"]["extraction"]["type"] == "object"


def test_fixture_generator_indexes_all_contract_types() -> None:
    generator = FixtureGenerator(FIXTURES_DIR)

    assert generator.contract_types == ("license", "msa", "nda")
    assert generator.count("nda") == 2
    assert generator.count("msa") == 1
    assert generator.count("license") == 2


def test_fixture_generator_returns_generation_result() -> None:
    generator = FixtureGenerator(FIXTURES_DIR)

    result = asyncio.run(
        generator.generate(
            contract_type="nda",
            system_prompt="ignored",
            user_prompt="ignored",
            seed=0,
        )
    )

    assert isinstance(result, GenerationResult)
    assert result.contract_type == "nda"
    assert result.backend == "fixture"
    assert result.cost_usd == 0.0
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert "Mutual" in result.contract_text


def test_fixture_generator_is_deterministic_for_same_seed() -> None:
    generator = FixtureGenerator(FIXTURES_DIR)

    a = asyncio.run(generator.generate("nda", "", "", seed=3))
    b = asyncio.run(generator.generate("nda", "", "", seed=3))

    assert a.contract_text == b.contract_text


def test_fixture_generator_cycles_seeds_modulo_count() -> None:
    generator = FixtureGenerator(FIXTURES_DIR)

    first = asyncio.run(generator.generate("nda", "", "", seed=0))
    wrapped = asyncio.run(generator.generate("nda", "", "", seed=generator.count("nda")))

    assert first.contract_text == wrapped.contract_text


def test_fixture_generator_extractions_parse_against_schema() -> None:
    generator = FixtureGenerator(FIXTURES_DIR)

    for contract_type in generator.contract_types:
        for seed in range(generator.count(contract_type)):
            result = asyncio.run(generator.generate(contract_type, "", "", seed=seed))
            ContractExtraction.model_validate(result.extraction)


def test_fixture_generator_raises_for_unknown_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FixtureGenerator(tmp_path / "does_not_exist")


def test_fixture_generator_raises_for_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no fixtures found"):
        FixtureGenerator(tmp_path)


def test_fixture_generator_raises_for_unknown_contract_type(tmp_path: Path) -> None:
    fixture = {
        "contract_type": "sla",
        "contract_text": "...",
        "extraction": {},
    }
    (tmp_path / "bad.json").write_text(__import__("json").dumps(fixture))

    with pytest.raises(ValueError, match="contract_type"):
        FixtureGenerator(tmp_path)


def test_fixture_generator_raises_when_keys_missing(tmp_path: Path) -> None:
    fixture = {"contract_type": "nda"}
    (tmp_path / "bad.json").write_text(__import__("json").dumps(fixture))

    with pytest.raises(ValueError, match="missing contract_text"):
        FixtureGenerator(tmp_path)


def test_fixture_generator_satisfies_protocol() -> None:
    generator = FixtureGenerator(FIXTURES_DIR)

    assert isinstance(generator, StructuredGenerator)
