"""Pure helpers for the HF Spaces three-way comparison.

Splits the Gradio app into a UI half (`app.py`) and a pure half (this
module) so the comparison logic can be unit-tested in CI without
importing `gradio` or loading any model. The UI module imports
`predict_three_way` and feeds the result into the three output columns.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

from anvil.eval.runner import ExtractionPredictor, Prediction


@dataclass(frozen=True)
class VariantOutput:
    """One variant's response on a single contract, ready for the UI."""

    raw_output: str
    latency_ms: float
    cost_usd: float


SAMPLE_CONTRACTS: tuple[str, ...] = (
    (
        "Mutual NDA between Acme Corp. (Delaware) and Globex Industries LLC "
        "(Delaware) effective February 15, 2026. 24-month term. Confidentiality "
        "survives 60 months. Governing law: State of Delaware. Litigation in "
        "Wilmington. Termination on material breach with 30 days notice and a "
        "15-day cure."
    ),
    (
        "MSA between Cloudforge Inc. (vendor, Massachusetts) and Beacon "
        "Financial Services (client, Massachusetts) effective January 1, "
        "2026. Initial term 36 months, auto-renewing for successive 12-month "
        "terms with 90 days notice. Massachusetts law. Indemnification capped "
        "at 2x annual fees. AAA arbitration in Boston."
    ),
    (
        "Perpetual software license between Lyra Technologies Ltd. (licensor, "
        "England and Wales) and Orion Robotics, Inc. (licensee, Delaware). No "
        "effective date. Perpetual term. England and Wales law. IP "
        "indemnification capped at $5,000,000. LCIA arbitration in London. "
        "Termination on material breach uncured for 30 days."
    ),
)


async def _predict_one(predictor: ExtractionPredictor, contract_text: str) -> VariantOutput:
    start = time.perf_counter()
    prediction: Prediction = await predictor.predict(contract_text)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return VariantOutput(
        raw_output=prediction.raw_output,
        latency_ms=elapsed_ms,
        cost_usd=prediction.cost_usd,
    )


async def _gather(
    base: ExtractionPredictor,
    finetuned: ExtractionPredictor,
    gpt_4o: ExtractionPredictor,
    contract_text: str,
) -> tuple[VariantOutput, VariantOutput, VariantOutput]:
    results = await asyncio.gather(
        _predict_one(base, contract_text),
        _predict_one(finetuned, contract_text),
        _predict_one(gpt_4o, contract_text),
    )
    return results[0], results[1], results[2]


def predict_three_way(
    base: ExtractionPredictor,
    finetuned: ExtractionPredictor,
    gpt_4o: ExtractionPredictor,
    contract_text: str,
) -> tuple[VariantOutput, VariantOutput, VariantOutput]:
    """Run all three predictors on `contract_text`. Synchronous wrapper."""
    if not contract_text or not contract_text.strip():
        raise ValueError("contract_text must be non-empty")
    return asyncio.run(_gather(base, finetuned, gpt_4o, contract_text))


def format_badge(output: VariantOutput) -> str:
    """Render the latency + cost badge line under each variant column."""
    return f"latency: {output.latency_ms:.0f} ms  ·  cost: ${output.cost_usd:.4f}"


def _ensure_predictors(state: dict[str, Any]) -> None:
    """Lazy-load the three predictors into `state` so the UI imports stay light."""
    if {"base", "finetuned", "gpt_4o"} <= state.keys():
        return
    from anvil.eval.local_predictor import LocalExtractionPredictor
    from anvil.eval.openai_predictor import OpenAIExtractionPredictor

    base_model = state["base_model"]
    adapter_path = state["adapter_path"]
    base = LocalExtractionPredictor(base_model=base_model)
    finetuned = LocalExtractionPredictor(base_model=base_model, adapter_path=adapter_path)
    gpt_4o = OpenAIExtractionPredictor()
    state.update({"base": base, "finetuned": finetuned, "gpt_4o": gpt_4o})
