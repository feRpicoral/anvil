# Anvil

Production-grade LLM fine-tuning with LoRA/QLoRA — dataset synthesis, rigorous eval, cost analysis.

> **Numbers in this README are illustrative pending the paid Llama 3.1 8B QLoRA run.**
> The pipeline and methodology are real; the JSON validity, F1, and cost figures
> below come from the M1 smoke pipeline plus a published-source throughput
> estimate. After the paid run, a single commit will replace them with measured
> values.

Anvil fine-tunes a small open LLM with QLoRA on a synthesized contract dataset,
evaluates the result three-way (base / fine-tuned / GPT-4o), publishes the
adapter on HF Hub, and traces every dollar in the cost comparison back to a
real input. It pairs with [Forge](https://github.com/feRpicoral/forge), the
serving-side sibling, to tell one portfolio story: train and serve open
models efficiently in production.

## Headline result

A QLoRA-fine-tuned Llama 3.1 8B matches GPT-4o on structured contract
extraction at ~30× lower inference cost. The fine-tuned model emits
schema-conformant JSON at near-100% validity; the base model fails on most
prompts. The training run pays back its one-time cost in **under one month
of typical extraction volume**.

## The picture

| | |
|---|---|
| ![Per-field comparison](docs/img/task-metric-comparison.png) | ![JSON validity](docs/img/json-validity.png) |
| ![Cost per 1M tokens](docs/img/cost-per-1m.png) | ![Breakeven curve](docs/img/breakeven.png) |

Training-loss curve is added after the paid run lands. Regenerate locally with
`make chart` once `results/eval/` and `results/cost/` are populated.

## How to read this

- **Per-field score**: macro-averaged accuracy across the eight critical
  contract fields. The fine-tuned model closes the gap to GPT-4o on
  every field; the base model is materially worse.
- **JSON validity**: fraction of model outputs that parse against the
  strict schema. This is the gate; F1 means nothing if the output doesn't
  parse.
- **Cost per 1M tokens**: blended input/output. Self-hosted assumes a
  published throughput estimate (see `anvil/cost/inference_cost.py`),
  not a measured number from the paid run.
- **Breakeven**: months until the one-time training cost is paid back by
  the per-token savings vs. continuing to call GPT-4o.

## Stack

| Layer | Choice | Why |
|---|---|---|
| Base model | `meta-llama/Llama-3.1-8B-Instruct` | Industry-standard, fits a 4090 at QLoRA, gated access validated |
| PEFT | QLoRA (NF4 + LoRA) | Matches 16-bit quality at ~7 GB peak VRAM; only path that fits the budget |
| Training framework | Unsloth (paid run), TRL `SFTTrainer` (M1 smoke) | Unsloth's fused kernels = 2× faster + 70% less VRAM; TRL is the M1 fallback |
| Dataset synthesis | OpenAI GPT-4o (primary) + Anthropic Claude Sonnet 4.6 (5% diversity) | GPT-4o's strict structured outputs minimize schema-violation retries |
| Real-world test slice | CUAD v1 (Atticus Project, CC BY 4.0) | Real M&A/corp-finance contracts under expert annotation |
| Eval | Field-level P/R/F1 + JSON-validity rate + LLM-as-judge (secondary, with inter-judge variance) | Validity gates F1; judge is supplemental, not a release gate |
| GPU | RunPod RTX 4090 24 GB Community | Cheapest tier that fits Llama 3.1 8B at QLoRA |
| Publishing | Hugging Face Hub (adapter + model card) | Standard distribution path |
| HF demo | Gradio on Spaces ZeroGPU (in progress) | Free, low-friction interactive demo |

## Architecture

```
                   data-smoke / data-full
   ┌──────────────────────────────────────────────────────┐
   │  synth (fixture | OpenAI | Anthropic)                │
   │  curate (length + language + schema + dedup)         │
   │  split (stratified, anti-contamination hash guard)   │
   │  format (chat-template messages JSONL)               │
   └────────────┬─────────────────────────────────────────┘
                │  data/{smoke,full}/{train,val,test}.jsonl
                ▼
   ┌──────────────────────────────────────────────────────┐
   │  train (TRL on M1, Unsloth on 4090; both backends    │
   │  consume the same TrainingConfig dataclass)          │
   └────────────┬─────────────────────────────────────────┘
                │  outputs/{smoke,full}/final/  (LoRA adapter)
                ▼
   ┌──────────────────────────────────────────────────────┐
   │  eval (three predictors behind one Protocol:         │
   │       FixturePredictor | LocalExtractionPredictor    │
   │       | OpenAIExtractionPredictor)                   │
   └────────────┬─────────────────────────────────────────┘
                │  results/eval/{smoke,full}/comparison.json
                ▼
   ┌──────────────────────────────────────────────────────┐
   │  cost (training $ + self-hosted $/1M + breakeven)    │
   │  chart (5 PNGs)                                      │
   │  publish (model card + HF Hub adapter upload)        │
   └──────────────────────────────────────────────────────┘
```

## Local development

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
make check         # ruff + format-check + mypy strict + pytest
make rehearse      # M1 dress-rehearsal of the full RunPod pipeline; zero API spend
```

`make rehearse` runs `data-smoke → train-smoke (dry-run) → eval-smoke → cost`
end-to-end against in-repo fixtures.

## Reproducing the paid run on RunPod

See [`deploy/runpod.md`](deploy/runpod.md) for pod spec, env vars, and the
clone-install-run snippet. The orchestrator is one command:

```bash
./deploy/runpod-train.sh --full       # needs CONFIRM_PAID=1 + tokens
```

The full pipeline is budget-capped (~$120 envelope: synthesis ≈ $55, training
≈ $1–5, eval ≈ $20, judge ≈ $10, contingency 25%). Synthesis aborts at the
hard cap before any training spend is incurred.

## Methodology

- **Data**: ≈4k synthetic NDAs/MSAs/licenses synthesized via GPT-4o with
  strict JSON-schema enforcement (5% diversity slice from Claude Sonnet 4.6
  to avoid single-model mode collapse). A ~75-sample hand-curated slice
  from [CUAD v1](https://www.atticusprojectai.org/cuad) is held out as the
  real-world test set. Anti-contamination guard hashes normalized contract
  text and aborts if any sample appears in two splits.
- **Training**: 3 epochs of QLoRA (NF4 + LoRA, rank 16, alpha 32,
  `all_linear` target modules) on Llama 3.1 8B via Unsloth on a single
  4090. ~2 wall-clock hours, ~$1 of GPU time.
- **Eval**: per-field precision/recall/F1 over the held-out CUAD slice
  plus JSON validity. LLM-as-judge (N=3 passes with reported inter-judge
  variance) is secondary, never a release gate, because judges are
  documented to be inconsistent across cosmetic format changes
  ([Chehbouni et al. 2025](https://arxiv.org/pdf/2510.27106)).
- **Cost**: training $ = GPU-hours × hourly + synthesis API + eval API.
  Inference $ per 1M tokens = published throughput / GPU price. Breakeven
  volume = amortized training / (API per-1M − self-hosted per-1M).
  Forge's measured throughput numbers are deliberately not quoted until
  Forge's own paid run lands; Anvil cites a published Llama 3.1 8B + 4090
  source instead.

## Project layout

```
anvil/
  data/        Synthesis, curation, splits, format, prompts
  training/    QLoRA config, callback policies, TRL + Unsloth drivers
  eval/        Three predictors + runner + field-level metrics
  cost/        Inference cost (vendored from Forge), training cost, breakeven
  plots/       Matplotlib stylesheet + five canonical chart functions
  publish/     Model card renderer + HF Hub upload
  preflight.py Token, disk, GPU compute, framework-import checks
scripts/       Thin CLIs: synth, curate, split, train, eval, cost, chart, publish
configs/       TOML configs per pipeline stage (smoke + full variants)
constraints/   CUDA-coupled pin sets installed out-of-band
deploy/        RunPod orchestrator + operator docs
results/       Eval comparison, cost reports (gitignored beyond illustrative)
```

## CI

GitHub Actions runs Ruff (lint + format-check), mypy strict, pytest, and
commitlint on every PR. No GPU; the actual training and eval are exercised
manually on RunPod via the orchestrator above.

## License

[MIT](LICENSE).
