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
from anvil.data.pricing import ANTHROPIC_PRICES, OPENAI_PRICES


def test_cost_known_value() -> None:
    cost = self_hosted_cost_per_1m_tokens(
        sustained_throughput_tps=2100.0,
        gpu_hourly_usd=0.27,
        utilization=1.0,
    )
    assert cost == pytest.approx(0.03571, abs=1e-4)


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


def test_cost_rejects_negative_gpu_price() -> None:
    with pytest.raises(ValueError, match="gpu_hourly_usd"):
        self_hosted_cost_per_1m_tokens(sustained_throughput_tps=1000, gpu_hourly_usd=-1.0)


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


def test_gpu_tiers_have_current_runpod_entries() -> None:
    assert GPU_TIERS["runpod-rtx-a5000-community"].hourly_usd == pytest.approx(0.27)
    assert GPU_TIERS["runpod-rtx-a5000-community"].vram_gb == 24
    assert GPU_TIERS["runpod-a100-pcie-80gb-community"].hourly_usd == pytest.approx(1.39)
    assert GPU_TIERS["runpod-a100-sxm-80gb-community"].hourly_usd == pytest.approx(1.49)
    assert GPU_TIERS["runpod-h100-pcie-80gb-community"].hourly_usd == pytest.approx(2.89)
    assert GPU_TIERS["runpod-h100-sxm-80gb-community"].hourly_usd == pytest.approx(3.29)


def test_api_pricing_has_canonical_entries() -> None:
    for key in ("gpt-4o", "claude-sonnet-4-6"):
        assert key in API_PRICING


def test_api_pricing_uses_canonical_price_tables() -> None:
    assert API_PRICING["gpt-4o"].input_usd_per_1m == OPENAI_PRICES["gpt-4o"][0]
    assert API_PRICING["gpt-4o"].output_usd_per_1m == OPENAI_PRICES["gpt-4o"][1]
    assert (
        API_PRICING["claude-haiku-4-5"].input_usd_per_1m == ANTHROPIC_PRICES["claude-haiku-4-5"][0]
    )
    assert (
        API_PRICING["claude-haiku-4-5"].output_usd_per_1m == ANTHROPIC_PRICES["claude-haiku-4-5"][1]
    )


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


def test_build_self_hosted_renders_notes() -> None:
    sh = build_self_hosted(
        label="QLoRA on A5000",
        gpu_tier_key="runpod-rtx-a5000-community",
        sustained_throughput_tps=2100.0,
        utilization=0.9,
    )
    scenario = sh.to_scenario()
    assert scenario.label == "QLoRA on A5000"
    assert "2100 tok/s sustained" in scenario.notes
    assert "90%" in scenario.notes
    assert scenario.usd_per_1m_tokens == pytest.approx(0.03968, abs=1e-4)


def test_build_self_hosted_unknown_gpu_raises_clean() -> None:
    with pytest.raises(ValueError, match="unknown GPU tier key"):
        build_self_hosted(label="X", gpu_tier_key="nope", sustained_throughput_tps=1000)


def test_compare_bundles_self_hosted_and_api() -> None:
    sh = build_self_hosted(
        label="QLoRA on A5000",
        gpu_tier_key="runpod-rtx-a5000-community",
        sustained_throughput_tps=2100.0,
    )
    cmp = compare([sh], ["gpt-4o", "claude-sonnet-4-6"], input_share=0.5)

    labels_api = [r.label for r in cmp.api]
    assert "GPT-4o" in labels_api
    assert "Claude Sonnet 4.6" in labels_api
    gpt = next(r for r in cmp.api if r.label == "GPT-4o")
    assert cmp.self_hosted[0].usd_per_1m_tokens < gpt.usd_per_1m_tokens / 50


def test_compare_rejects_invalid_input_share() -> None:
    with pytest.raises(ValueError, match="input_share"):
        compare([], ["gpt-4o"], input_share=2.0)


def test_compare_rejects_unknown_api_key() -> None:
    with pytest.raises(ValueError, match="unknown API pricing key"):
        compare([], ["unknown"])


def test_to_dict_round_trips_through_json() -> None:
    sh = build_self_hosted(
        label="QLoRA on A5000",
        gpu_tier_key="runpod-rtx-a5000-community",
        sustained_throughput_tps=2100.0,
    )
    cmp = compare([sh], ["gpt-4o"])
    encoded = json.dumps(cmp.to_dict())
    decoded = json.loads(encoded)
    assert decoded["input_share"] == cmp.input_share
    assert decoded["self_hosted"][0]["label"] == "QLoRA on A5000"
