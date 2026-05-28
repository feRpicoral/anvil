---
title: Anvil 3-way Contract Extraction
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
hardware: zero-a10g
---

# Anvil: three-way contract extraction

Live demo of the Anvil portfolio piece. The same contract text runs
through three predictors so you can compare the outputs side-by-side:

- **Base Llama 3.1 8B** — `meta-llama/Llama-3.1-8B-Instruct` with no
  fine-tuning. Calibration baseline.
- **Fine-tuned QLoRA** — the same base + the Anvil LoRA adapter, trained
  on synthesized contract data.
- **GPT-4o baseline** — `gpt-4o-2024-08-06` with strict JSON-schema
  enforcement.

Each column also shows latency and per-call cost.

## Source

Anvil: <https://github.com/feRpicoral/anvil>

## Environment

The Space reads two env vars:

| Variable | Default |
|---|---|
| `ANVIL_BASE_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` |
| `ANVIL_ADAPTER_PATH` | `outputs/full/final` |

Plus `OPENAI_API_KEY` for the GPT-4o column and `HF_TOKEN` for the gated
Llama 3.1 weights. Both are Space secrets, not committed to the repo.

## Notes

ZeroGPU cold starts make the first inference ~10-15 s; subsequent
inferences in the same session are fast.
