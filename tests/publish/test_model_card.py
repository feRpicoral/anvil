from __future__ import annotations

from anvil.publish.model_card import (
    CostSummary,
    EvalSummaryRow,
    ModelCardData,
    render,
)


def _minimal_data(**overrides: object) -> ModelCardData:
    defaults: dict[str, object] = {
        "model_name": "anvil-llama31-8b-qlora-contracts",
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "license": "llama3.1",
        "language": "en",
        "task_name": "Contract field extraction",
        "task_description": "Extracts the eight critical fields from legal contracts.",
        "training_data_description": "~4000 synthetic NDAs/MSAs/licenses + 75 real CUAD samples.",
        "training_framework": "Unsloth + TRL SFTTrainer",
        "quantization": "NF4 (QLoRA)",
        "lora_rank": 16,
        "lora_alpha": 32,
        "epochs": 3,
        "learning_rate": 2.0e-4,
        "max_seq_len": 2048,
    }
    defaults.update(overrides)
    return ModelCardData(**defaults)  # type: ignore[arg-type]


def test_render_starts_with_yaml_frontmatter() -> None:
    card = render(_minimal_data())

    assert card.startswith("---\n")
    assert "base_model: meta-llama/Llama-3.1-8B-Instruct" in card
    assert "language: en" in card
    assert "license: llama3.1" in card
    assert "library_name: peft" in card


def test_render_includes_required_sections() -> None:
    card = render(_minimal_data())

    for section in (
        "# anvil-llama31-8b-qlora-contracts",
        "## Intended use",
        "## Training",
        "## Evaluation",
        "## Cost",
        "## Limitations",
    ):
        assert section in card, f"missing section: {section}"


def test_render_emits_default_tags() -> None:
    card = render(_minimal_data())

    for tag in ("text-generation", "lora", "qlora"):
        assert f"- {tag}" in card


def test_render_drops_blocklisted_tags() -> None:
    card = render(_minimal_data(tags=("text-generation", "", "<keep-me-or-cite>", "legal")))

    assert "- text-generation" in card
    assert "- legal" in card
    assert "<keep-me-or-cite>" not in card


def test_render_eval_block_is_placeholder_without_results() -> None:
    card = render(_minimal_data())

    assert "Pending" in card.split("## Evaluation", 1)[1].split("##", 1)[0]


def test_render_eval_block_renders_table_when_results_present() -> None:
    rows = (
        EvalSummaryRow(variant="base", json_validity_rate=0.30, macro_f1=0.25),
        EvalSummaryRow(variant="finetuned", json_validity_rate=0.99, macro_f1=0.82),
        EvalSummaryRow(variant="gpt-4o", json_validity_rate=1.00, macro_f1=0.85),
    )
    card = render(_minimal_data(eval_summary=rows))

    eval_block = card.split("## Evaluation", 1)[1].split("##", 1)[0]
    assert "| Variant |" in eval_block
    assert "| finetuned | 99.00% | 82.00%" in eval_block
    assert "| gpt-4o | 100.00% | 85.00%" in eval_block


def test_render_cost_block_is_placeholder_without_cost() -> None:
    card = render(_minimal_data())

    cost_block = card.split("## Cost", 1)[1].split("##", 1)[0]
    assert "Pending" in cost_block


def test_render_cost_block_renders_summary() -> None:
    cost = CostSummary(
        training_total_usd=56.76,
        self_hosted_per_1m_tokens=0.22,
        primary_api_label="GPT-4o",
        primary_api_per_1m_tokens=6.25,
        breakeven_monthly_m_tokens=1.6,
        breakeven_months_horizon=12,
    )
    card = render(_minimal_data(cost=cost))

    cost_block = card.split("## Cost", 1)[1].split("##", 1)[0]
    assert "$56.76" in cost_block
    assert "$0.2200/1M tokens self-hosted" in cost_block
    assert "GPT-4o" in cost_block
    assert "Breakeven" in cost_block
    assert "1.60 M tokens/month" in cost_block
    assert "12 months" in cost_block


def test_render_cost_block_handles_zero_self_hosted_cost() -> None:
    cost = CostSummary(
        training_total_usd=0.0,
        self_hosted_per_1m_tokens=0.0,
        primary_api_label="GPT-4o",
        primary_api_per_1m_tokens=6.25,
        breakeven_monthly_m_tokens=0.0,
        breakeven_months_horizon=12,
    )
    card = render(_minimal_data(cost=cost))

    cost_block = card.split("## Cost", 1)[1].split("##", 1)[0]
    assert "GPT-4o" in cost_block


def test_render_references_block_omitted_when_no_sources() -> None:
    card = render(_minimal_data())

    assert "## References" not in card


def test_render_references_block_included_when_sources_supplied() -> None:
    card = render(_minimal_data(sources=("https://example.com/cuad-paper",)))

    assert "## References" in card
    assert "https://example.com/cuad-paper" in card


def test_render_training_block_includes_hyperparameters() -> None:
    card = render(_minimal_data(lora_rank=8, lora_alpha=16, epochs=1, learning_rate=5e-5))

    training_block = card.split("## Training", 1)[1].split("##", 1)[0]
    assert "**LoRA rank / alpha**: 8 / 16" in training_block
    assert "**Epochs**: 1" in training_block
    assert "5e-05" in training_block


def test_render_includes_base_model_link() -> None:
    card = render(_minimal_data())

    assert "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct" in card
