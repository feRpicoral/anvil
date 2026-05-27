# Vendored from forge@2b9d733fb6b0df49a2c55fca7d879a4240843b72
# (forge/cost/pricing.py + forge/cost/model.py, combined verbatim).
# When Forge updates its cost module, refresh this file from the upstream
# revision and bump the SHA in this header.
"""Self-hosted vs commercial API cost-per-1M-tokens model + pricing tables.

The math is intentionally trivial — `(hourly $) / (tokens-per-second * 3600 / 1e6)`
— but the value of this module is in making every input explicit. Every
inference-cost chart in the README must come from a structured payload
produced here, with the assumptions surfaced.

GPU hourly rates and commercial-API per-token prices change frequently. The
values here are pinned at a point in time and recorded under the project's
working notes; before any paid run that quotes these numbers, refresh both
tables against current rates and re-run the chart pipeline.

Sources:
- RunPod community tier pricing: https://runpod.io/pricing (as of 2026-05-26)
- OpenAI API pricing:            https://openai.com/api/pricing (as of 2026-05-26)
- Anthropic Claude API pricing:  https://www.anthropic.com/pricing (as of 2026-05-26)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GpuTier:
    name: str
    hourly_usd: float
    vram_gb: int
    notes: str


@dataclass(frozen=True)
class ApiPricing:
    """Per-1M-token prices for a commercial API."""

    name: str
    input_usd_per_1m: float
    output_usd_per_1m: float

    def blended_per_1m(self, input_share: float = 0.5) -> float:
        """Single-number cost assuming a given input/output mix.

        ``input_share`` is the fraction of tokens that are prompt vs.
        completion. Defaults to 50/50, matching the ShareGPT trace's rough mix.
        """
        if not 0.0 <= input_share <= 1.0:
            raise ValueError(f"input_share must be in [0, 1], got {input_share}")
        output_share = 1.0 - input_share
        return self.input_usd_per_1m * input_share + self.output_usd_per_1m * output_share


# Single source of truth for GPU pricing. Add tiers, don't redefine them inline.
GPU_TIERS: dict[str, GpuTier] = {
    "runpod-rtx-4090-community": GpuTier(
        name="RunPod RTX 4090 (Community)",
        hourly_usd=0.34,
        vram_gb=24,
        notes="The benchmark target. Fits Llama 3.1 8B BF16 + KV cache headroom.",
    ),
    "runpod-a100-40gb-community": GpuTier(
        name="RunPod A100 40GB (Community)",
        hourly_usd=1.19,
        vram_gb=40,
        notes="Out of budget for the scoped run; included for sensitivity analysis.",
    ),
    "runpod-a100-80gb-community": GpuTier(
        name="RunPod A100 80GB (Community)",
        hourly_usd=1.89,
        vram_gb=80,
        notes="Out of budget for the scoped run; included for sensitivity analysis.",
    ),
    "runpod-h100-80gb-community": GpuTier(
        name="RunPod H100 80GB (Community)",
        hourly_usd=2.99,
        vram_gb=80,
        notes="Native FP8. Out of budget; reference for cost-scaling discussion.",
    ),
}


# Approximate commercial API pricing as of 2026-05-26. Update the numbers AND
# the source links above when refreshing.
API_PRICING: dict[str, ApiPricing] = {
    "gpt-4o": ApiPricing(name="GPT-4o", input_usd_per_1m=2.50, output_usd_per_1m=10.00),
    "gpt-4o-mini": ApiPricing(name="GPT-4o mini", input_usd_per_1m=0.15, output_usd_per_1m=0.60),
    "claude-sonnet-4-6": ApiPricing(
        name="Claude Sonnet 4.6", input_usd_per_1m=3.00, output_usd_per_1m=15.00
    ),
    "claude-haiku-4-5": ApiPricing(
        name="Claude Haiku 4.5", input_usd_per_1m=0.80, output_usd_per_1m=4.00
    ),
}


def self_hosted_cost_per_1m_tokens(
    *,
    sustained_throughput_tps: float,
    gpu_hourly_usd: float,
    utilization: float = 1.0,
) -> float:
    """USD per 1M tokens at a given sustained throughput and GPU utilization.

    Args:
        sustained_throughput_tps: Total tokens/sec the server processes,
            *averaged over a real workload* (not peak). Use the value from the
            benchmark, not the optimistic ceiling.
        gpu_hourly_usd: Rented GPU price. Read from ``GPU_TIERS``.
        utilization: Fraction of time the GPU is producing tokens. 1.0 means a
            fully-loaded server; 0.5 means it's idle half the time.

    The formula:
        seconds per 1M tokens = 1e6 / (throughput * utilization)
        cost = (gpu_hourly_usd / 3600) * seconds_per_1m_tokens
             = gpu_hourly_usd * 1e6 / (3600 * throughput * utilization)
    """
    if sustained_throughput_tps <= 0:
        raise ValueError(f"sustained_throughput_tps must be > 0, got {sustained_throughput_tps}")
    if not 0.0 < utilization <= 1.0:
        raise ValueError(f"utilization must be in (0, 1], got {utilization}")
    return gpu_hourly_usd * 1e6 / (3600.0 * sustained_throughput_tps * utilization)


@dataclass(frozen=True)
class CostScenario:
    """One row of the cost comparison: a label, dollars, and provenance."""

    label: str
    usd_per_1m_tokens: float
    notes: str = ""


@dataclass(frozen=True)
class SelfHostedCost:
    """A self-hosted scenario's full inputs and computed cost."""

    label: str
    gpu_tier_key: str
    sustained_throughput_tps: float
    utilization: float
    usd_per_1m_tokens: float

    def to_scenario(self) -> CostScenario:
        notes = (
            f"{GPU_TIERS[self.gpu_tier_key].name} at "
            f"{self.sustained_throughput_tps:.0f} tok/s sustained, "
            f"{int(self.utilization * 100)}% utilization."
        )
        return CostScenario(label=self.label, usd_per_1m_tokens=self.usd_per_1m_tokens, notes=notes)


@dataclass(frozen=True)
class CostComparison:
    """A comparison ready for the chart pipeline.

    ``self_hosted`` and ``api`` are independent rows; the chart code stacks
    them. ``input_share`` is the assumption used to blend API per-token rates.
    """

    self_hosted: list[CostScenario]
    api: list[CostScenario]
    input_share: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "self_hosted": [
                {
                    "label": s.label,
                    "usd_per_1m_tokens": s.usd_per_1m_tokens,
                    "notes": s.notes,
                }
                for s in self.self_hosted
            ],
            "api": [
                {
                    "label": s.label,
                    "usd_per_1m_tokens": s.usd_per_1m_tokens,
                    "notes": s.notes,
                }
                for s in self.api
            ],
            "input_share": self.input_share,
            "notes": self.notes,
        }


def build_self_hosted(
    *,
    label: str,
    gpu_tier_key: str,
    sustained_throughput_tps: float,
    utilization: float = 1.0,
) -> SelfHostedCost:
    """Compute a ``SelfHostedCost`` from a benchmark number + GPU tier key.

    The key must exist in ``GPU_TIERS`` — typos crash loud rather than silently
    producing the wrong cost.
    """
    tier = GPU_TIERS[gpu_tier_key]
    usd = self_hosted_cost_per_1m_tokens(
        sustained_throughput_tps=sustained_throughput_tps,
        gpu_hourly_usd=tier.hourly_usd,
        utilization=utilization,
    )
    return SelfHostedCost(
        label=label,
        gpu_tier_key=gpu_tier_key,
        sustained_throughput_tps=sustained_throughput_tps,
        utilization=utilization,
        usd_per_1m_tokens=usd,
    )


def compare(
    self_hosted: list[SelfHostedCost],
    api_keys: list[str],
    *,
    input_share: float = 0.5,
) -> CostComparison:
    """Build a ``CostComparison`` from self-hosted scenarios + API keys.

    Each API key must exist in ``API_PRICING``.
    """
    if not 0.0 <= input_share <= 1.0:
        raise ValueError(f"input_share must be in [0, 1], got {input_share}")

    api: list[CostScenario] = []
    for key in api_keys:
        pricing: ApiPricing = API_PRICING[key]
        blended = pricing.blended_per_1m(input_share=input_share)
        api.append(
            CostScenario(
                label=pricing.name,
                usd_per_1m_tokens=blended,
                notes=(
                    f"input ${pricing.input_usd_per_1m:.2f}/1M, "
                    f"output ${pricing.output_usd_per_1m:.2f}/1M, "
                    f"blended at {int(input_share * 100)}% input."
                ),
            )
        )

    return CostComparison(
        self_hosted=[s.to_scenario() for s in self_hosted],
        api=api,
        input_share=input_share,
    )
