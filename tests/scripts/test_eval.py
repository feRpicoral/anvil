from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from anvil.eval.runner import FixturePredictor
from scripts.eval import (
    VariantConfig,
    build_predictor,
    load_eval_config,
    load_fixture_predictions,
    load_test_cases,
    run,
    write_comparison,
    write_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_FIXTURES = REPO_ROOT / "tests" / "data" / "fixtures" / "eval"


def _write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "eval.toml"
    path.write_text(body, encoding="utf-8")
    return path


def _three_variant_body(output_dir: Path) -> str:
    return f"""
test_jsonl = "{EVAL_FIXTURES / "test_cases.jsonl"}"
output_dir = "{output_dir}"

[[variant]]
name = "base"
predictor = "fixture"
fixtures_path = "{EVAL_FIXTURES / "predictions_invalid.jsonl"}"

[[variant]]
name = "finetuned"
predictor = "fixture"
fixtures_path = "{EVAL_FIXTURES / "predictions_perfect.jsonl"}"

[[variant]]
name = "gpt-4o"
predictor = "fixture"
fixtures_path = "{EVAL_FIXTURES / "predictions_perfect.jsonl"}"
"""


def test_load_test_cases_parses_messages_format() -> None:
    cases = load_test_cases(EVAL_FIXTURES / "test_cases.jsonl")

    assert len(cases) == 3
    assert {c.case_id for c in cases} == {"nda-de", "msa-ma", "license-perp"}
    nda = next(c for c in cases if c.case_id == "nda-de")
    assert "Acme Corp." in nda.contract_text
    assert nda.gold_extraction.parties[0].name == "Acme Corp."


def test_load_test_cases_assigns_case_id_when_missing(tmp_path: Path) -> None:
    src = EVAL_FIXTURES / "test_cases.jsonl"
    stripped_rows = [
        {k: v for k, v in json.loads(line).items() if k != "case_id"}
        for line in src.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    path = tmp_path / "no-ids.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for row in stripped_rows:
            fh.write(json.dumps(row) + "\n")

    cases = load_test_cases(path)

    assert [c.case_id for c in cases] == ["case-0000", "case-0001", "case-0002"]


def test_load_test_cases_assigns_dense_case_ids_with_blank_lines(tmp_path: Path) -> None:
    src = EVAL_FIXTURES / "test_cases.jsonl"
    rows = [
        {k: v for k, v in json.loads(line).items() if k != "case_id"}
        for line in src.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    path = tmp_path / "blank-lines.jsonl"
    path.write_text("\n" + "\n\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    cases = load_test_cases(path)

    assert [c.case_id for c in cases] == ["case-0000", "case-0001", "case-0002"]


def test_load_test_cases_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    row = json.loads(
        (EVAL_FIXTURES / "test_cases.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    path = tmp_path / "duplicate-cases.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate case_id"):
        load_test_cases(path)


def test_load_test_cases_raises_for_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no test cases"):
        load_test_cases(path)


def test_load_fixture_predictions_reads_rows() -> None:
    predictions = load_fixture_predictions(EVAL_FIXTURES / "predictions_perfect.jsonl")

    assert set(predictions.keys()) == {"nda-de", "msa-ma", "license-perp"}
    assert predictions["nda-de"].input_tokens == 250


def test_load_fixture_predictions_raises_for_empty(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="no fixture predictions"):
        load_fixture_predictions(path)


def test_load_fixture_predictions_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    row = json.loads(
        (EVAL_FIXTURES / "predictions_perfect.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    path = tmp_path / "duplicate-predictions.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate case_id"):
        load_fixture_predictions(path)


def test_load_fixture_predictions_rejects_negative_token_counts(tmp_path: Path) -> None:
    row = json.loads(
        (EVAL_FIXTURES / "predictions_perfect.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    row["input_tokens"] = -1
    path = tmp_path / "negative-tokens.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="input_tokens"):
        load_fixture_predictions(path)


def test_load_eval_config_parses_variants(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, _three_variant_body(tmp_path / "out"))

    config = load_eval_config(config_path)

    assert len(config.variants) == 3
    assert [v.name for v in config.variants] == ["base", "finetuned", "gpt-4o"]
    assert config.test_jsonl == EVAL_FIXTURES / "test_cases.jsonl"


def test_load_eval_config_rejects_duplicate_variant_names(tmp_path: Path) -> None:
    body = f"""
test_jsonl = "{EVAL_FIXTURES / "test_cases.jsonl"}"
output_dir = "{tmp_path / "out"}"

[[variant]]
name = "base"
predictor = "fixture"
fixtures_path = "{EVAL_FIXTURES / "predictions_perfect.jsonl"}"

[[variant]]
name = "base"
predictor = "fixture"
fixtures_path = "{EVAL_FIXTURES / "predictions_perfect.jsonl"}"
"""
    config_path = _write_config(tmp_path, body)

    with pytest.raises(ValueError, match="duplicate variant"):
        load_eval_config(config_path)


def test_load_eval_config_rejects_unknown_predictor(tmp_path: Path) -> None:
    body = f"""
test_jsonl = "{EVAL_FIXTURES / "test_cases.jsonl"}"
output_dir = "{tmp_path / "out"}"

[[variant]]
name = "base"
predictor = "wishful-thinking"
fixtures_path = "{EVAL_FIXTURES / "predictions_perfect.jsonl"}"
"""
    config_path = _write_config(tmp_path, body)

    with pytest.raises(ValueError, match="predictor must be"):
        load_eval_config(config_path)


def test_load_eval_config_rejects_unsafe_variant_names(tmp_path: Path) -> None:
    body = f"""
test_jsonl = "{EVAL_FIXTURES / "test_cases.jsonl"}"
output_dir = "{tmp_path / "out"}"

[[variant]]
name = "../outside"
predictor = "fixture"
fixtures_path = "{EVAL_FIXTURES / "predictions_perfect.jsonl"}"
"""
    config_path = _write_config(tmp_path, body)

    with pytest.raises(ValueError, match="invalid variant name"):
        load_eval_config(config_path)


def test_load_eval_config_rejects_empty_variants(tmp_path: Path) -> None:
    body = f"""
test_jsonl = "{EVAL_FIXTURES / "test_cases.jsonl"}"
output_dir = "{tmp_path / "out"}"
variant = []
"""
    config_path = _write_config(tmp_path, body)

    with pytest.raises(ValueError, match="non-empty"):
        load_eval_config(config_path)


def test_build_predictor_fixture_requires_path() -> None:
    config = VariantConfig(name="base", predictor="fixture", fixtures_path=None)

    with pytest.raises(ValueError, match="fixtures_path"):
        build_predictor(config)


def test_build_predictor_fixture_returns_fixture_predictor() -> None:
    config = VariantConfig(
        name="base",
        predictor="fixture",
        fixtures_path=EVAL_FIXTURES / "predictions_perfect.jsonl",
    )

    predictor = build_predictor(config)

    assert isinstance(predictor, FixturePredictor)


def test_build_predictor_local_requires_base_model() -> None:
    config = VariantConfig(name="base", predictor="local", base_model=None)

    with pytest.raises(ValueError, match="base_model"):
        build_predictor(config)


def test_write_summary_and_comparison_round_trip(tmp_path: Path) -> None:
    from anvil.eval.runner import VariantSummary

    summary = VariantSummary(
        variant="base",
        n_cases=3,
        json_validity_rate=0.33,
        field_scores={"parties": 1.0, "term": 0.75},
        total_input_tokens=750,
        total_output_tokens=15,
        total_cost_usd=0.0,
        mean_latency_ms=2.5,
    )

    summary_path = write_summary(tmp_path, summary)
    comparison_path = write_comparison(tmp_path, [summary])

    assert summary_path.exists()
    loaded = json.loads(summary_path.read_text(encoding="utf-8"))
    assert loaded["variant"] == "base"
    assert loaded["json_validity_rate"] == 0.33

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert comparison["n_cases"] == 3
    assert len(comparison["variants"]) == 1


def test_run_smoke_produces_three_variant_jsons(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    config_path = _write_config(tmp_path, _three_variant_body(output_dir))
    args = argparse.Namespace(config=config_path)

    rc = run(args)

    assert rc == 0
    base = json.loads((output_dir / "base.json").read_text(encoding="utf-8"))
    finetuned = json.loads((output_dir / "finetuned.json").read_text(encoding="utf-8"))
    gpt4o = json.loads((output_dir / "gpt-4o.json").read_text(encoding="utf-8"))
    comparison = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))

    assert base["json_validity_rate"] == pytest.approx(1 / 3)
    assert finetuned["json_validity_rate"] == 1.0
    assert gpt4o["json_validity_rate"] == 1.0
    assert comparison["n_cases"] == 3
    assert [v["variant"] for v in comparison["variants"]] == ["base", "finetuned", "gpt-4o"]


def test_smoke_config_file_loads() -> None:
    config = load_eval_config(Path("configs/eval-smoke.toml"))

    assert len(config.variants) == 3
    assert {v.name for v in config.variants} == {"base", "finetuned", "gpt-4o"}
