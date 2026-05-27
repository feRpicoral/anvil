from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from anvil.data.synthesis import GenerationResult
from scripts.synth import (
    SynthConfig,
    build_generator,
    load_config,
    load_results,
    run,
    synthesize,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "tests" / "data" / "fixtures" / "synthesis"


def _write_config(tmp_path: Path, fixtures_dir: Path, num_samples: int = 6) -> Path:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'backend = "fixture"\n'
        f'fixtures_dir = "{fixtures_dir}"\n'
        f"num_samples = {num_samples}\n"
        f'output_dir = "{tmp_path / "out"}"\n'
        "seed = 0\n",
        encoding="utf-8",
    )
    return config_path


def test_load_config_parses_required_keys(tmp_path: Path) -> None:
    path = _write_config(tmp_path, FIXTURES_DIR, num_samples=12)

    config = load_config(path)

    assert config.backend == "fixture"
    assert config.fixtures_dir == FIXTURES_DIR
    assert config.num_samples == 12
    assert config.seed == 0


def test_load_config_rejects_missing_key(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text('backend = "fixture"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="missing required key"):
        load_config(path)


def test_build_generator_rejects_unknown_backend() -> None:
    config = SynthConfig(
        backend="unsupported",
        fixtures_dir=FIXTURES_DIR,
        num_samples=1,
        output_dir=Path("/tmp"),
        seed=0,
    )

    with pytest.raises(ValueError, match="unsupported backend"):
        build_generator(config)


def test_build_generator_fixture_requires_dir() -> None:
    config = SynthConfig(
        backend="fixture",
        fixtures_dir=None,
        num_samples=1,
        output_dir=Path("/tmp"),
        seed=0,
    )

    with pytest.raises(ValueError, match="requires fixtures_dir"):
        build_generator(config)


def test_synthesize_cycles_contract_types() -> None:
    import asyncio

    from anvil.data.synthesis import FixtureGenerator

    generator = FixtureGenerator(FIXTURES_DIR)

    results = asyncio.run(synthesize(generator, num_samples=9, base_seed=0))

    types = [result.contract_type for result in results]
    assert types == ["nda", "msa", "license"] * 3


def test_run_writes_raw_jsonl_with_n_rows(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, FIXTURES_DIR, num_samples=15)
    args = argparse.Namespace(config=config_path)

    rc = run(args)

    assert rc == 0
    output = tmp_path / "out" / "raw_synthesis.jsonl"
    rows = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 15
    record = GenerationResult(**json.loads(rows[0]))
    assert record.backend == "fixture"
    assert record.cost_usd == 0.0


def test_load_results_round_trips(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, FIXTURES_DIR, num_samples=6)
    run(argparse.Namespace(config=config_path))

    results = load_results(tmp_path / "out" / "raw_synthesis.jsonl")

    assert len(results) == 6
    for result in results:
        assert isinstance(result, GenerationResult)
