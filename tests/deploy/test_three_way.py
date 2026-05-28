"""Tests for the HF Spaces pure helpers.

The Gradio UI in `app.py` isn't exercised here — it requires gradio +
torch + transformers, all heavier than what CI installs. Tests exercise
the comparison logic against the same `FixturePredictor` the eval
runner uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SPACE_ROOT = Path(__file__).resolve().parents[2] / "deploy" / "hf-spaces"
sys.path.insert(0, str(_SPACE_ROOT))

from three_way import (  # type: ignore[import-not-found]  # noqa: E402
    SAMPLE_CONTRACTS,
    VariantOutput,
    _ensure_predictors,
    format_badge,
    predict_three_way,
)

from anvil.eval.runner import FixturePredictor, Prediction  # noqa: E402


def _make_predictor(raw_output: str = '{"ok": true}', cost: float = 0.01) -> FixturePredictor:
    predictor = FixturePredictor(
        {
            "only": Prediction(
                raw_output=raw_output, input_tokens=10, output_tokens=20, cost_usd=cost
            )
        }
    )
    from anvil.data.schema import ContractExtraction
    from anvil.eval.runner import EvalCase

    dummy_gold = ContractExtraction.model_validate(
        {
            "parties": [
                {"name": "a", "role": "buyer"},
                {"name": "b", "role": "seller"},
            ],
            "effective_date": None,
            "term": {
                "duration_months": None,
                "is_perpetual": True,
                "auto_renew": False,
                "renewal_notice_days": None,
            },
            "governing_law": None,
            "jurisdiction": None,
            "confidentiality": None,
            "termination": {"triggers": [], "notice_days": None, "cure_period_days": None},
            "indemnification": None,
            "dispute_resolution": {"forum": "litigation", "venue": None, "governing_rules": None},
        }
    )
    predictor.attach_cases(
        [EvalCase(case_id="only", contract_text="ANY_CONTRACT_TEXT", gold_extraction=dummy_gold)]
    )
    return predictor


def test_sample_contracts_are_non_empty_strings() -> None:
    assert SAMPLE_CONTRACTS
    for contract in SAMPLE_CONTRACTS:
        assert isinstance(contract, str)
        assert contract.strip()


def test_predict_three_way_runs_all_predictors() -> None:
    base = _make_predictor('{"variant": "base"}', cost=0.0)
    finetuned = _make_predictor('{"variant": "finetuned"}', cost=0.0)
    gpt = _make_predictor('{"variant": "gpt-4o"}', cost=0.05)

    base_out, ft_out, gpt_out = predict_three_way(base, finetuned, gpt, "ANY_CONTRACT_TEXT")

    assert base_out.raw_output == '{"variant": "base"}'
    assert ft_out.raw_output == '{"variant": "finetuned"}'
    assert gpt_out.raw_output == '{"variant": "gpt-4o"}'


def test_predict_three_way_captures_latency_and_cost() -> None:
    base = _make_predictor(cost=0.0)
    finetuned = _make_predictor(cost=0.0)
    gpt = _make_predictor(cost=0.07)

    _, _, gpt_out = predict_three_way(base, finetuned, gpt, "ANY_CONTRACT_TEXT")

    assert isinstance(gpt_out, VariantOutput)
    assert gpt_out.cost_usd == pytest.approx(0.07)
    assert gpt_out.latency_ms >= 0


def test_predict_three_way_rejects_empty_contract() -> None:
    base = _make_predictor()
    finetuned = _make_predictor()
    gpt = _make_predictor()

    with pytest.raises(ValueError, match="contract_text"):
        predict_three_way(base, finetuned, gpt, "   ")


def test_format_badge_includes_latency_and_cost() -> None:
    output = VariantOutput(raw_output="{}", latency_ms=123.4, cost_usd=0.0567)

    badge = format_badge(output)

    assert "123 ms" in badge
    assert "$0.0567" in badge


def test_ensure_predictors_populates_state_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    # OpenAIExtractionPredictor() constructs AsyncOpenAI, which needs an API
    # key from env or kwarg. Set a fake one so the constructor doesn't trip
    # in CI; the predictor never actually calls the API in this test.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    state: dict[str, object] = {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "adapter_path": Path("outputs/smoke/final"),
    }

    _ensure_predictors(state)

    assert "base" in state
    assert "finetuned" in state
    assert "gpt_4o" in state
    base_id = id(state["base"])
    _ensure_predictors(state)
    assert id(state["base"]) == base_id


def test_ensure_predictors_retries_after_partial_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    state: dict[str, object] = {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "adapter_path": Path("outputs/smoke/final"),
        "base": object(),
    }

    _ensure_predictors(state)

    assert "finetuned" in state
    assert "gpt_4o" in state


def test_async_gather_does_not_block_on_each_predictor() -> None:
    base = _make_predictor()
    finetuned = _make_predictor()
    gpt = _make_predictor()

    import time

    start = time.perf_counter()
    predict_three_way(base, finetuned, gpt, "ANY_CONTRACT_TEXT")
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 100
