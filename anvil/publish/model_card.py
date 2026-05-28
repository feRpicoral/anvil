"""HF Hub model-card rendering.

Pure-Python templated Markdown. The caller fills a `ModelCardData` from
real run metadata (training config + eval comparison + cost report); this
module produces the YAML-fronted Markdown that HF Hub expects, with
conditional blocks for eval and cost so a pre-eval upload still produces
a readable card.

No Jinja dep — substitutions are f-strings, conditional blocks live in
small Python helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final


@dataclass(frozen=True)
class EvalSummaryRow:
    """One variant's headline numbers, ready for the comparison table."""

    variant: str
    json_validity_rate: float
    macro_f1: float
    notes: str = ""


@dataclass(frozen=True)
class CostSummary:
    """The cost numbers the README quotes."""

    training_total_usd: float
    self_hosted_per_1m_tokens: float
    primary_api_label: str
    primary_api_per_1m_tokens: float
    breakeven_monthly_m_tokens: float
    breakeven_months_horizon: int


@dataclass(frozen=True)
class ModelCardData:
    """All inputs the model card renders from."""

    model_name: str
    base_model: str
    license: str
    language: str
    task_name: str
    task_description: str
    training_data_description: str
    training_framework: str
    quantization: str
    lora_rank: int
    lora_alpha: int
    epochs: int
    learning_rate: float
    max_seq_len: int
    eval_summary: tuple[EvalSummaryRow, ...] = ()
    cost: CostSummary | None = None
    sources: tuple[str, ...] = ()
    tags: tuple[str, ...] = field(default_factory=lambda: ("text-generation", "lora", "qlora"))


_TAG_BLOCKLIST: Final[frozenset[str]] = frozenset({"", "<keep-me-or-cite>"})


def render(data: ModelCardData) -> str:
    """Render the model card as YAML-fronted Markdown."""
    return "".join(
        [
            _render_frontmatter(data),
            _render_overview(data),
            _render_intended_use(data),
            _render_training(data),
            _render_eval(data),
            _render_cost(data),
            _render_limitations(data),
            _render_citations(data),
        ]
    )


def _render_frontmatter(data: ModelCardData) -> str:
    tags = [tag for tag in data.tags if tag not in _TAG_BLOCKLIST]
    tag_lines = "\n".join(f"  - {tag}" for tag in tags)
    return (
        "---\n"
        f"base_model: {data.base_model}\n"
        f"language: {data.language}\n"
        f"license: {data.license}\n"
        "library_name: peft\n"
        "tags:\n"
        f"{tag_lines}\n"
        "---\n\n"
    )


def _render_overview(data: ModelCardData) -> str:
    return (
        f"# {data.model_name}\n\n"
        f"{data.task_description}\n\n"
        f"Fine-tuned from [`{data.base_model}`](https://huggingface.co/{data.base_model}) "
        f"with {data.training_framework} + {data.quantization}.\n\n"
    )


def _render_intended_use(data: ModelCardData) -> str:
    return (
        "## Intended use\n\n"
        f"{data.task_name}. {data.task_description}\n\n"
        "**Out of scope.** General-purpose chat or instruction following — the adapter is "
        "specialized to the extraction task and will degrade on unrelated prompts.\n\n"
    )


def _render_training(data: ModelCardData) -> str:
    return (
        "## Training\n\n"
        f"- **Data**: {data.training_data_description}\n"
        f"- **Framework**: {data.training_framework}\n"
        f"- **Quantization**: {data.quantization}\n"
        f"- **LoRA rank / alpha**: {data.lora_rank} / {data.lora_alpha}\n"
        f"- **Epochs**: {data.epochs}\n"
        f"- **Learning rate**: {data.learning_rate}\n"
        f"- **Max sequence length**: {data.max_seq_len}\n\n"
    )


def _render_eval(data: ModelCardData) -> str:
    if not data.eval_summary:
        return (
            "## Evaluation\n\n"
            "_Pending. The three-way comparison (base / fine-tuned / GPT-4o) is "
            "produced by `make eval-full` and replaces this section after the paid run._\n\n"
        )
    header = "| Variant | JSON validity | Macro F1 | Notes |\n|---|---|---|---|\n"
    rows = "".join(
        f"| {row.variant} | {row.json_validity_rate:.2%} | {row.macro_f1:.2%} | {row.notes} |\n"
        for row in data.eval_summary
    )
    return "## Evaluation\n\n" + header + rows + "\n"


def _render_cost(data: ModelCardData) -> str:
    if data.cost is None:
        return (
            "## Cost\n\n"
            "_Pending. Computed by `make cost-full` and replaces this section after "
            "the paid run._\n\n"
        )
    cost = data.cost
    ratio_note = (
        f"({cost.primary_api_per_1m_tokens / cost.self_hosted_per_1m_tokens:.0f}x cheaper "
        f"than {cost.primary_api_label})"
        if cost.self_hosted_per_1m_tokens > 0
        else f"(vs. {cost.primary_api_label})"
    )
    return (
        "## Cost\n\n"
        f"- **Training**: ${cost.training_total_usd:.2f} one-time.\n"
        f"- **Inference**: ${cost.self_hosted_per_1m_tokens:.4f}/1M tokens self-hosted "
        f"{ratio_note}.\n"
        f"- **Breakeven**: ~{cost.breakeven_monthly_m_tokens:.2f} M tokens/month over "
        f"{cost.breakeven_months_horizon} months.\n\n"
    )


def _render_limitations(data: ModelCardData) -> str:
    del data
    return (
        "## Limitations\n\n"
        "- Trained on synthesized contracts plus a hand-curated real-world test slice; "
        "performance on out-of-distribution legal text (regulated jurisdictions, niche "
        "clauses) is not validated.\n"
        "- JSON output is enforced by training, not by a runtime grammar — production "
        "use should still validate every parse.\n"
        "- No safety tuning beyond the base model's. Do not feed adversarial inputs.\n\n"
    )


def _render_citations(data: ModelCardData) -> str:
    if not data.sources:
        return ""
    lines = "\n".join(f"- {source}" for source in data.sources)
    return f"## References\n\n{lines}\n"
