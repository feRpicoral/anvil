"""Anvil three-way demo for Hugging Face Spaces.

Runs the same contract through three predictors (base Llama 3.1 8B,
fine-tuned LoRA, GPT-4o) and shows the output + latency + cost per
column. Heavy imports (`gradio`, `torch`, `transformers`) happen at
module import for Spaces; the comparison logic lives in `three_way.py`
so it can be unit-tested without those deps installed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr
from three_way import (
    SAMPLE_CONTRACTS,
    _ensure_predictors,
    format_badge,
    predict_three_way,
)

_STATE: dict[str, Any] = {
    "base_model": os.environ.get("ANVIL_BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
    "adapter_path": Path(os.environ.get("ANVIL_ADAPTER_PATH", "outputs/full/final")),
}


def _predict(contract_text: str) -> tuple[str, str, str, str, str, str]:
    _ensure_predictors(_STATE)
    base_out, ft_out, gpt_out = predict_three_way(
        _STATE["base"], _STATE["finetuned"], _STATE["gpt_4o"], contract_text
    )
    return (
        base_out.raw_output,
        format_badge(base_out),
        ft_out.raw_output,
        format_badge(ft_out),
        gpt_out.raw_output,
        format_badge(gpt_out),
    )


with gr.Blocks(title="Anvil: 3-way contract extraction") as demo:
    gr.Markdown(
        "# Anvil — three-way contract extraction\n\n"
        "Drop a contract on the left. The same text goes through three "
        "predictors so you can compare them side-by-side."
    )

    contract_box = gr.Textbox(
        label="Contract text",
        placeholder="Paste an NDA, MSA, or license here...",
        lines=12,
    )
    submit = gr.Button("Extract", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Base Llama 3.1 8B")
            base_code = gr.Code(language="json", label="extraction")
            base_badge = gr.Markdown()
        with gr.Column():
            gr.Markdown("### Fine-tuned (QLoRA)")
            ft_code = gr.Code(language="json", label="extraction")
            ft_badge = gr.Markdown()
        with gr.Column():
            gr.Markdown("### GPT-4o baseline")
            gpt_code = gr.Code(language="json", label="extraction")
            gpt_badge = gr.Markdown()

    submit.click(
        _predict,
        inputs=[contract_box],
        outputs=[base_code, base_badge, ft_code, ft_badge, gpt_code, gpt_badge],
    )
    gr.Examples(
        examples=[[contract] for contract in SAMPLE_CONTRACTS],
        inputs=[contract_box],
    )


if __name__ == "__main__":
    demo.launch()
