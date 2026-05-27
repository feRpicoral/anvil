# Vendored from forge@2b9d733fb6b0df49a2c55fca7d879a4240843b72
# (tests/test_cost_model.py). Refresh alongside anvil/cost/inference_cost.py.
"""Worked numerical tests for the cost model.

Every assertion comes from a hand-computed expected value. The model is trivial
enough that any deviation indicates either a formula bug or a units mistake.
"""

from __future__ import annotations

import json

import pytest

from anvil.cost.inference_cost import (
    API_PRICING,
    GPU_TIERS,
    ApiPricing,
    build_self_hosted,
    compare,
    self_hosted_cost_per_1m_tokens,
)

# --- Formula tests -----------------------------------------------------------


def test_cost_known_value() -> None:
    """At 2,100 tok/s sustained on a $0.34/hr GPU, $/1M tokens = 0.34 * 1e6 / (3600 * 2100).

    Hand-computed: 340000 / 7,560,000 ≈ 0.04497.
    """
    cost = self_hosted_cost_per_1m_tokens(
        sustained_throughput_tps=2100.0,
        gpu_hourly_usd=0.34,
        utilization=1.0,
    )
    assert cost == pytest.approx(0.04497, abs=1e-4)


def test_cost_scales_linearly_with_gpu_price() -> None:
    base = self_hosted_cost_per_1m_tokens(sustained_throughput_tps=1000, gpu_hourly_usd=1.0)
    doubled = self_hosted_cost_per_1m_tokens(sustained_throughput_tps=1000, gpu_hourly_usd=2.0)
    assert doubled == pytest.approx(2.0 * base)


def test_cost_scales_inversely_with_throughput() -> None:
    base = self_hosted_cost_per_1m_tokens(sustained_throughput_tps=1000, gpu_hourly_usd=1.0)
    doubled_throughput = self_hosted_cost_per_1m_tokens(
        sustained_throughput_tps=2000, gpu_hourly_usd=1.0
    )
    assert doubled_throughput == pytest.approx(base / 2)


def test_cost_scales_inversely_with_utilization() -> None:
    full = self_hosted_cost_per_1m_tokens(
        sustained_throughput_tps=1000, gpu_hourly_usd=1.0, utilization=1.0
    )
    half = self_hosted_cost_per_1m_tokens(
        sustained_throughput_tps=1000, gpu_hourly_usd=1.0, utilization=0.5
    )
    assert half == pytest.approx(2.0 * full)


def test_cost_rejects_zero_throughput() -> None:
    with pytest.raises(ValueError, match="sustained_throughput_tps"):
        self_hosted_cost_per_1m_tokens(sustained_throughput_tps=0.0, gpu_hourly_usd=1.0)


def test_cost_rejects_zero_utilization() -> None:
    with pytest.raises(ValueError, match="utilization"):
        self_hosted_cost_per_1m_tokens(
            sustained_throughput_tps=1000, gpu_hourly_usd=1.0, utilization=0.0
        )


def test_cost_rejects_utilization_above_one() -> None:
    with pytest.raises(ValueError, match="utilization"):
        self_hosted_cost_per_1m_tokens(
            sustained_throughput_tps=1000, gpu_hourly_usd=1.0, utilization=1.5
        )


# --- Pricing tables ----------------------------------------------------------


def test_gpu_tiers_have_canonical_entry() -> None:
    """The benchmark target must always be in the pricing table."""
    assert "runpod-rtx-4090-community" in GPU_TIERS
    assert GPU_TIERS["runpod-rtx-4090-community"].hourly_usd > 0
    assert GPU_TIERS["runpod-rtx-4090-community"].vram_gb == 24


def test_api_pricing_has_canonical_entries() -> None:
    for key in ("gpt-4o", "claude-sonnet-4-6"):
        assert key in API_PRICING


def test_blended_per_1m_50_50() -> None:
    pricing = ApiPricing(name="X", input_usd_per_1m=2.0, output_usd_per_1m=10.0)
    assert pricing.blended_per_1m(input_share=0.5) == pytest.approx(6.0)


def test_blended_per_1m_all_input() -> None:
    pricing = ApiPricing(name="X", input_usd_per_1m=2.0, output_usd_per_1m=10.0)
    assert pricing.blended_per_1m(input_share=1.0) == pytest.approx(2.0)


def test_blended_per_1m_all_output() -> None:
    pricing = ApiPricing(name="X", input_usd_per_1m=2.0, output_usd_per_1m=10.0)
    assert pricing.blended_per_1m(input_share=0.0) == pytest.approx(10.0)


def test_blended_per_1m_rejects_out_of_range() -> None:
    pricing = ApiPricing(name="X", input_usd_per_1m=2.0, output_usd_per_1m=10.0)
    with pytest.raises(ValueError, match="input_share"):
        pricing.blended_per_1m(input_share=1.5)


# --- High-level builders -----------------------------------------------------


def test_build_self_hosted_renders_notes() -> None:
    sh = build_self_hosted(
        label="AWQ on 4090",
        gpu_tier_key="runpod-rtx-4090-community",
        sustained_throughput_tps=2100.0,
        utilization=0.9,
    )
    scenario = sh.to_scenario()
    assert scenario.label == "AWQ on 4090"
    assert "2100 tok/s sustained" in scenario.notes
    assert "90%" in scenario.notes
    # $/1M = 0.34 * 1e6 / (3600 * 2100 * 0.9) = ≈ 0.04996.
    assert scenario.usd_per_1m_tokens == pytest.approx(0.04996, abs=1e-4)


def test_build_self_hosted_unknown_gpu_raises_clean() -> None:
    with pytest.raises(KeyError):
        build_self_hosted(label="X", gpu_tier_key="nope", sustained_throughput_tps=1000)


def test_compare_bundles_self_hosted_and_api() -> None:
    sh = build_self_hosted(
        label="AWQ on 4090",
        gpu_tier_key="runpod-rtx-4090-community",
        sustained_throughput_tps=2100.0,
    )
    cmp = compare([sh], ["gpt-4o", "claude-sonnet-4-6"], input_share=0.5)

    labels_api = [r.label for r in cmp.api]
    assert "GPT-4o" in labels_api
    assert "Claude Sonnet 4.6" in labels_api
    gpt = next(r for r in cmp.api if r.label == "GPT-4o")
    assert cmp.self_hosted[0].usd_per_1m_tokens < gpt.usd_per_1m_tokens / 100


def test_compare_rejects_invalid_input_share() -> None:
    with pytest.raises(ValueError, match="input_share"):
        compare([], ["gpt-4o"], input_share=2.0)


def test_to_dict_round_trips_through_json() -> None:
    sh = build_self_hosted(
        label="AWQ on 4090",
        gpu_tier_key="runpod-rtx-4090-community",
        sustained_throughput_tps=2100.0,
    )
    cmp = compare([sh], ["gpt-4o"])
    encoded = json.dumps(cmp.to_dict())
    decoded = json.loads(encoded)
    assert decoded["input_share"] == cmp.input_share
    assert decoded["self_hosted"][0]["label"] == "AWQ on 4090"
