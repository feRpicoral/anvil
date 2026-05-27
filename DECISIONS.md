# Decisions

Forward-looking record of choices and rationale for Anvil. Mirrors the format
of [`forge/DECISIONS.md`](../forge/DECISIONS.md). Status tokens are plain text
(`planned` / `in progress` / `partial` / `complete` / `awaiting GPU`) per the
user's global instructions. Each decision pins the exact version actually
used after the relevant phase lands, never a speculative range.

## Project status

| Phase | Goal | Status |
|---|---|---|
| 0 | Repo bootstrap (public repo, GitHub governance) | complete |
| 1 | Tooling foundation (uv, Ruff, mypy, pytest, pre-commit, CI, Makefile) | complete |
| 2 | Research + `DECISIONS.md` (framework, technique, compat matrix, budget envelope) | complete |
| 3 | Dataset pipeline (synthesis, curation, dedup, splits, chat-template formatting) | planned |
| 4 | Training pipeline (QLoRA config, W&B tracking, smoke + full configs) | planned |
| 5 | Eval harness (3-way: base / fine-tuned / GPT-4o; JSON output) | planned |
| 6 | Cost model (training + inference; reuses vendored Forge cost module) | planned |
| 7 | Smoke (M1) + paid run (RunPod) | planned |
| 8 | Publish (HF Hub adapter + model card) | planned |
| 9 | Charts + impact-first README + single-commit illustrative-to-real swap | planned |
| 10 | HF Spaces 3-way side-by-side demo | planned |
| 11 | v1.0.0 tag + polish | planned |

## Scope

Anvil is a portfolio engineering artifact, not a SaaS. It fine-tunes a small
open LLM with QLoRA on a synthesized, curated dataset, evaluates the result
defensibly against the base model and a big-model prompting baseline,
publishes the adapter on HF Hub, and traces every dollar in the cost
comparison back to a measured number. It pairs with Forge (which serves
trained models) to tell one portfolio narrative: train + serve open models
efficiently in production.

What Anvil is **not**:

- A product or SaaS (no UI, no auth, no billing).
- A serving / inference optimization piece. That is Forge.
- A multi-task fine-tuning framework. One task, depth over breadth.
- RLHF / preference tuning. Phase 2 research confirms preference signal does
  not apply to a deterministic-output extraction task (see "PEFT technique"
  below).

## Stack overview

| Layer | Choice | Status |
|---|---|---|
| Language | Python 3.12 | locked |
| Dependency manager | uv (+ `uv.lock`) | locked |
| Lint / format | Ruff (line 100) | locked |
| Type checker | mypy strict | locked |
| Tests | pytest + pytest-cov | locked |
| CI | GitHub Actions (no GPU) | locked |
| Pre-commit | pre-commit + commitlint | locked |
| Task | Structured extraction from legal contracts (NDAs / MSAs) | locked |
| Base model | `meta-llama/Llama-3.1-8B-Instruct` | locked |
| PEFT technique | QLoRA (NF4 base + LoRA adapters) | locked |
| Training framework (paid GPU) | Unsloth (`2026.5.x`) | locked |
| Training framework (M1 smoke) | TRL `SFTTrainer` + PEFT (LoRA, non-quantized) | locked |
| Dataset synthesis (primary) | OpenAI `gpt-4o-2024-08-06` (structured outputs) | locked |
| Dataset synthesis (diversity) | Anthropic Claude Sonnet 4.6 (~5% slice) | locked |
| Real-world test slice | CUAD v1 (Atticus Project, CC BY 4.0) | locked |
| Experiment tracking | Weights & Biases | locked |
| GPU cloud (paid run) | RunPod RTX 4090 24 GB Community | locked |
| Publishing | Hugging Face Hub (adapter + model card) | locked |
| HF demo | Gradio + Spaces ZeroGPU (Phase 10) | planned |

Exact versions are recorded under "Pinned versions" after the first
successful install of the training stack against `constraints/train.txt`
(Phase 4) and the eval stack against `constraints/eval.txt` (Phase 5).

## Decisions

### Task: structured extraction from legal contracts

- **Why**: Three dimensions where fine-tuning a small open model beats
  prompting a big API model on this task.
  1. **Privacy.** Contracts can carry confidentiality clauses that forbid
     sending the text to a third-party API. A self-hosted small model is
     the only legal answer for many B2B customers.
  2. **JSON-validity rate.** A model fine-tuned to always emit a schema-
     valid payload obviates retry loops and structural fallbacks at
     inference time.
  3. **Cost at volume.** Recurring high-throughput extraction over many
     contracts is where a $0.0X / 1M cost-per-token self-hosted model
     beats API spend by an order of magnitude.
- **Schema** (eight critical fields, Phase 3 freezes the Pydantic model):
  parties, effective date, term length + renewal terms, jurisdiction and
  governing law, confidentiality scope, termination triggers,
  indemnification, dispute resolution.
- **Training data**: synthesized NDAs and MSAs (Phase 3), ~4k samples,
  GPT-4o primary with 5% Claude Sonnet diversity injection.
- **Real-world test slice**: ~75 hand-curated samples from the Contract
  Understanding Atticus Dataset ([CUAD v1](https://www.atticusprojectai.org/cuad)).
  Real M&A and corp-finance contracts under expert annotation, CC BY 4.0.
  Distinct from synthesized data so it tests generalization, not
  memorization.
- **Rejected**:
  - **Classification / routing** has lower portfolio differentiation. The
    head-to-head against GPT-4o on a binary or short-label task is harder
    to read at a glance.
  - **Tone / format generation** is judge-heavy and noisy. The documented
    LLM-as-judge inconsistency ([Chehbouni et al.,
    2025](https://arxiv.org/pdf/2510.27106)) makes the eval expensive to
    defend rigorously on a portfolio.
  - **Tool calling** has significant overlap with existing OSS work and
    the eval framing is brittle.
- **Sources**:
  - [CUAD on Hugging Face](https://huggingface.co/datasets/theatticusproject/cuad-qa)
  - [CUAD paper, Hendrycks et al. 2021](https://arxiv.org/pdf/2103.06268)

### Base model: `meta-llama/Llama-3.1-8B-Instruct`

- **Why**: Parity with Forge (same model end-to-end produces one
  continuous portfolio narrative), gated-token access already validated
  in Forge's pre-flight, fits a 24 GB consumer GPU at QLoRA with
  substantial headroom ([Local AI Master,
  2026](https://localaimaster.com/blog/qlora-fine-tuning-guide)).
- **Rejected**:
  - **Mistral 7B Instruct v0.3.** Smaller community for QLoRA recipes;
    breaks the Forge narrative.
  - **Qwen 2.5 7B Instruct.** Strong baseline but breaks the Forge
    narrative. Kept as the smoke base (smaller `Qwen 2.5 0.5B Instruct`)
    because it's ungated and CPU/MPS-friendly.

### PEFT technique: QLoRA (NF4 base + LoRA adapters)

- **Why**: NF4 quantization of the frozen base is information-theoretically
  better than INT4 for normally-distributed pretrained weights and matches
  the 16-bit baseline on instruction-tuning workloads ([QLoRA, Dettmers et
  al. 2023](https://arxiv.org/abs/2305.14314); [bitsandbytes 4-bit
  reference](https://huggingface.co/docs/bitsandbytes/en/reference/nn/linear4bit)).
  Peak VRAM for Llama 3.1 8B at seq-len 2048 batch 1 lands at ~7 GB with
  Unsloth, leaving headroom on the 4090 for gradient checkpointing, larger
  effective batch sizes, and longer sequences if needed.
  `target_modules` starts at the PEFT default (`q/k/v/o` plus
  `gate/up/down`), tuned only if the first LoRA baseline plateaus.
- **Rejected**:
  - **LoRA without quantization.** Llama 3.1 8B in BF16 is ~16 GB of
    weights alone; combined with AdamW optimizer state and activations
    at seq-len 2048 it does not fit on a 4090. QLoRA matches the 16-bit
    quality baseline, so there is nothing to gain.
  - **DoRA.** Strong on reasoning benchmarks ([Liu et al.
    2024](https://arxiv.org/abs/2402.09353)), but the deltas are
    reasoning-flavored. Anvil's task is structured extraction with
    deterministic output, where reasoning-benchmark improvements are
    unlikely to transfer. DoRA also adds parameter count and training
    time. Revisit only if the LoRA baseline plateaus.
  - **Full fine-tuning.** 8B BF16 weights + grads + AdamW state is roughly
    64 GB just for parameters. Needs ≥80 GB cards. Out of budget and out
    of scope.
  - **DPO / ORPO.** Preference tuning optimizes a chosen-vs-rejected
    margin. The extraction task has a single correct output per input,
    so there is no preference signal. SFT is the right objective.
  - **GaLore.** Full-parameter quality at LoRA memory ([Zhao et al.
    2024](https://arxiv.org/abs/2403.03507)) but ~4x LoRA compute and no
    documented win over QLoRA on 8B instruction-tuning.
- **Sources**:
  - [HF PEFT LoRA developer guide](https://huggingface.co/docs/peft/developer_guides/lora)
  - [HF Transformers bitsandbytes quantization](https://huggingface.co/docs/transformers/quantization/bitsandbytes)

### Training framework: Unsloth (paid run) + TRL `SFTTrainer` (smoke)

- **Why split**: One config schema, two backends selected at runtime.
  Unsloth's fused Triton kernels measurably beat the stock TRL + PEFT +
  bitsandbytes baseline on a single 4090: claimed 2x training throughput
  and ~70% lower VRAM ([Unsloth blog](https://unsloth.ai/blog/llama3-1)),
  with third-party reproductions showing 3.2 h vs. 5.8 h on a single A100
  for Llama 3.1 8B QLoRA ([Spheron,
  2026](https://www.spheron.network/blog/axolotl-vs-unsloth-vs-torchtune/)).
  On a tight RunPod budget, the speed and memory delta determines whether
  three epochs complete inside the envelope.
- **Why TRL for the smoke**: Unsloth currently rejects Apple Silicon
  (`NotImplementedError: Unsloth currently only works on NVIDIA, AMD and
  Intel GPUs`, [issue #4774, April
  2026](https://github.com/unslothai/unsloth/issues/4774)). Of the four
  framework candidates surveyed, TRL is the only one whose dependency
  chain (`transformers` + `peft` + `accelerate`) runs cleanly under MPS
  with `PYTORCH_ENABLE_MPS_FALLBACK=1` and no `bitsandbytes`. The smoke
  validates wiring, chat-template, and adapter save/load against Qwen 2.5
  0.5B Instruct in BF16 on M1, not numerics. QLoRA-specific behavior is
  only validated on the linux+CUDA paid run.
- **Rejected**:
  - **Axolotl.** Good YAML ergonomics, but v0.16 kernel work targets MoE
    and RL workloads, not dense 8B; the Mac path also forbids
    `bitsandbytes`, so there is no QLoRA-on-smoke advantage over TRL
    ([Axolotl Mac docs](https://docs.axolotl.ai/docs/mac.html)).
  - **LLaMA-Factory.** Broadest model zoo and a Gradio UI, neither needed
    here; no kernel-level edge; last PyPI release was December 2025.
  - **TRL alone for the paid run.** Gives up Unsloth's 2x / 70% on a
    tight budget for no offsetting gain. TRL v1's own documentation
    points to Unsloth as the integration path for that speedup.
  - **Unsloth-MLX / `mlx-tune`.** Separate codebase. Maintaining two
    training stacks for smoke parity is not worth the cost when TRL+MPS
    already works.
- **Sources**:
  - [Unsloth releases](https://github.com/unslothai/unsloth/releases) and
    [`pyproject.toml`](https://github.com/unslothai/unsloth/blob/main/pyproject.toml)
  - [TRL releases](https://github.com/huggingface/trl/releases) and
    [SFTTrainer docs](https://huggingface.co/docs/trl/en/sft_trainer)
  - [HuggingFace blog: Fine-tune Llama 3.1 with Unsloth](https://huggingface.co/blog/mlabonne/sft-llama3)
  - [MarkTechPost: TRL v1.0](https://www.marktechpost.com/2026/04/01/hugging-face-releases-trl-v1-0-a-unified-post-training-stack-for-sft-reward-modeling-dpo-and-grpo-workflows/)
  - [Accelerate MPS guide](https://huggingface.co/docs/accelerate/en/usage_guides/mps)

### Dataset synthesis stack: GPT-4o primary + Claude Sonnet diversity

- **Primary generator**: `gpt-4o-2024-08-06`. The August 2024 snapshot is
  the canonical structured-outputs model. With `response_format={"type":
  "json_schema", "strict": true}` it produces schema-conformant JSON
  reliably enough to drop schema-violation retries from the synthesis
  loop ([OpenAI structured
  outputs](https://platform.openai.com/docs/guides/structured-outputs);
  [OpenAI launch
  post](https://openai.com/index/introducing-structured-outputs-in-the-api/)).
  GPT-5 (Aug 2025) and GPT-4.1 are both in the lineup. GPT-4.1 is ~20%
  cheaper on input tokens but has documented `response_format=json_schema`
  gaps in chat completions ([OpenAI dev
  forum](https://community.openai.com/t/clarity-on-gpt-4-1-and-o4-mini-structured-output-support/1230973)),
  and the resulting ~$0.40 of synthesis savings does not justify the
  schema-conformance regression risk. Re-evaluate only if GPT-4o
  stylistic mode collapse appears in dataset diversity audits.
- **Diversity injector**: Claude Sonnet 4.6 at ~5% of synthesis (~200
  samples). Single-model synthesis revisits high-probability regions of
  one model's output distribution; cross-family mixing is the cheapest
  mitigation that needs no training-side hook. Claude outputs are
  re-validated against the same JSON schema post-hoc.
- **Real-world test slice**: CUAD v1 from the Atticus Project. CC BY 4.0,
  ~510 real commercial contracts, ~13k expert-annotated clause spans
  across 41 categories. Clause-span labels flatten into Anvil's
  extraction schema. Reachable via the official site and an HF mirror.
- **API pricing (May 2026)**:

  | Model | Input $/1M | Output $/1M | Role |
  |---|---|---|---|
  | `gpt-4o-2024-08-06` | $2.50 | $10.00 | Primary synthesis, eval baseline |
  | `gpt-4.1` | $2.00 | $8.00 | Considered, not used |
  | Claude Sonnet 4.6 | $3.00 | $15.00 | Diversity injector |
  | Claude Haiku 4.5 | $1.00 | $5.00 | Primary judge (Phase 5) |

- **Sources**:
  - [OpenAI API Pricing](https://openai.com/api/pricing/)
  - [Anthropic API Pricing](https://www.anthropic.com/pricing)
  - [CUAD on Hugging Face](https://huggingface.co/datasets/theatticusproject/cuad-qa)
  - [CUAD paper](https://arxiv.org/pdf/2103.06268)

### Eval methodology

- **Primary metrics (deterministic)**:
  - **Field-level precision / recall / F1** across the extraction schema.
    Per-field breakdowns prevent a single average from masking failure on
    a rare-but-critical field. F1 is the standard for span-level
    information extraction.
  - **JSON-validity rate**: the fraction of outputs that parse against
    the schema in `strict: true` mode. This is the gate before F1 is
    meaningful. A model that produces invalid JSON 30% of the time is
    unusable regardless of field accuracy on the parseable subset.
  - **Exact-match on critical fields**: parties (with string
    normalization), effective date, governing law, term length. Partial
    credit is misleading on these. "California" vs. "California, USA"
    can both be right, but "Delaware" vs. "California" is a binary
    failure that should not be smoothed into an F1 average.
- **Secondary metric (qualitative, sanity-check only)**: LLM-as-judge for
  clause-fidelity using a versioned rubric (rubric hash in eval logs).
  N=3 judge passes per sample with inter-judge variance reported. Treated
  as supplemental signal, never as a release gate. Documented
  unreliability of judges across cosmetic format changes, paraphrasing,
  and verbosity ([Chehbouni et al.,
  2025](https://arxiv.org/pdf/2510.27106); [Yan et al.,
  2025](https://arxiv.org/pdf/2512.16041)) makes a single judge score
  indefensible on its own.
- **Test composition**:
  - **Held-out synthetic split**: ~500 samples (~12.5% of synthesis),
    stratified by contract type, frozen before any training.
  - **CUAD hand-curated slice**: ~75 samples. Real distribution, no
    generator-class overlap. This is the headline number reported
    externally.
- **Anti-contamination guard**: CUAD contract text never appears in any
  synthesis prompt. Enforced via a banlist check in
  `anvil/data/synthesis.py` (Phase 3). All three splits hash their
  normalized contract text (SHA-256). Any non-empty intersection between
  train / val / test aborts the run.
- **Sources**:
  - [Rating Roulette: Self-Inconsistency in LLM-As-A-Judge Frameworks (2025)](https://arxiv.org/pdf/2510.27106)
  - [Are We on the Right Way to Assessing LLM-as-a-Judge? (2025)](https://arxiv.org/pdf/2512.16041)
  - [Invoice Information Extraction: field-level F1 framing](https://arxiv.org/html/2510.15727v1)

### GPU cloud: RunPod RTX 4090 24 GB Community

- **Why**: Parity with Forge ($0.34/hr; ~$1–$3 wall cost for the 3-epoch
  paid run with Unsloth). Same `pricing.py` GPU tier in the vendored cost
  module (Phase 6), so the comparison story is consistent across the two
  repos. Community tier is acceptable because the training run is short,
  the artifact is the LoRA adapter (not the GPU session), and the run is
  re-runnable.
- **Rejected** (documented for honesty, not used):
  - **Modal**, **Lambda**, **Vast.ai**: equivalent or better DX, no cost
    advantage at this scale, no learning amortization vs. the RunPod
    path already exercised by Forge.
  - **A100 40 GB on RunPod**: ~$1.19/hr (3.5x cost). Llama 3.1 8B at
    QLoRA does not need the extra VRAM at our seq-len.
- **Source**: [RunPod RTX 4090 page (Community $0.34/hr)](https://www.runpod.io/gpu-models/rtx-4090)

### Budget envelope

Worked numbers using May 2026 pricing. Synthesis assumes ~200 input + ~1300
output tokens per sample. Eval baseline assumes ~2000 input (contract
chunk) + ~800 output (extracted JSON). Judge assumes ~2500 input + ~300
output. All caps in the table are hard: code short-circuits the run if the
spend tracker crosses the cap.

| Line item | Calculation | Computed | Hard cap |
|---|---|---|---|
| Synthesis (GPT-4o, 4k samples) | 0.8M in × $2.50 + 5.2M out × $10.00 | $54.00 | $55 |
| Diversity (Claude Sonnet 4.6, 200) | 0.04M in × $3 + 0.26M out × $15 | $4.02 | $5 |
| QLoRA training (4090 Community) | $0.34/hr × ~4 hr (Unsloth on 4k × 2048 × 3 epochs) | $1.36 | $5 |
| Eval baseline (GPT-4o, 500) | 1.0M in × $2.50 + 0.4M out × $10 | $6.50 | $20 |
| Judge (Claude Haiku 4.5, 500, N=3) | 3.75M in × $1 + 0.45M out × $5 | $6.00 | $10 |
| **Subtotal** | | **$71.88** | **$95** |
| Contingency 25% (of caps) | | | $23.75 |
| **Total envelope** | | | **~$120** |

**Abort triggers (during training)**:

- Training loss diverges (NaN or sustained increase over 100 steps).
- Validation loss plateaus or worsens before completing the first epoch.
- JSON-validity on the validation set drops below 50% at any checkpoint.
- Spend tracker exceeds 80% of envelope ($96) before training begins,
  which signals synthesis overrun.

**Abort triggers (post-training)**:

- CUAD slice F1 falls below baseline GPT-4o F1 minus 5 points. The
  fine-tune is then actively worse than the model that generated the
  data, a sign of synthesis mode collapse or schema overfitting.

### Vendor-from-Forge policy

Anvil treats Forge as a sibling repo and vendors specific Forge modules
rather than path-depending on Forge or publishing Forge to PyPI. Vendored
modules carry a header comment `# Vendored from forge@<commit-sha>` that
pins the source revision.

Planned vendor scope (lands in the phase noted):

- `forge/cost/model.py` + `forge/cost/pricing.py` → `anvil/cost/inference_cost.py` (Phase 6)
- `forge/plots/style.py` → `anvil/plots/style.py` (Phase 9)
- Inference cost-model tests carry over alongside the vendored module.

Path-deps break on fresh clones of either repo; PyPI publish for Forge is
out of scope. Vendor + cite is the lowest-friction durable choice.

### Forge-numbers reference policy

Forge's current benchmark numbers are illustrative pending its Phase 7
paid run. Anvil's README and `DECISIONS.md` never quote Forge throughput,
cost, or latency numbers until that run lands. When referring to the
serving sibling, Anvil uses capability phrasing — for example, "served by
a vLLM + AWQ stack like Forge's; throughput estimate Y tok/s based on
[published source]" — with the estimate anchored in a *published* Llama
3.1 8B + 4090 source cited in this file.

## Pinned versions

### Dev tooling (`uv.lock`, validated 2026-05-27)

| Package | Version | Source |
|---|---|---|
| python | 3.12.x | uv-managed |
| ruff | 0.15.14 | `[dependency-groups] dev` |
| mypy | 2.1.0 | `[dependency-groups] dev` |
| pytest | 9.0.3 | `[dependency-groups] dev` |
| pytest-cov | 7.1.0 | `[dependency-groups] dev` |
| pre-commit | 4.6.0 | `[dependency-groups] dev` |

### Training stack (`constraints/train.txt`, planned; validated in Phase 4 rehearsal)

| Package | Planned pin | Notes |
|---|---|---|
| unsloth | `2026.5.x` | Required only on linux + CUDA path |
| trl | `0.24.0` | Unsloth caps at `<=0.24.0`; excludes `0.19.0` |
| transformers | `4.57.x` (4.x line) | Pin away from 5.x while Unsloth excludes specific 5.x patches |
| peft | `0.19.x` | Unsloth requires `>=0.18.0` and excludes `0.11.0` |
| bitsandbytes | `0.48.1` or `0.49.x` | Unsloth excludes `0.46.0` and `0.48.0` |
| accelerate | `1.13.x` | Required `>=0.34.1` |
| datasets | `3.6.x` (avoid 4.0.x, 4.1.0, `>=4.4`) | Per Unsloth pyproject exclusions |
| torch | per Unsloth wheel | Pinned by the chosen Unsloth wheel (`cu126-torch...`) |
| flash-attn | `2.8.3` | Prebuilt wheels for Python 3.12, torch 2.x, CUDA 12.6, Ada Lovelace |

### Eval stack (`constraints/eval.txt`, planned; validated in Phase 5)

| Package | Planned pin | Notes |
|---|---|---|
| openai | latest stable, May 2026 | GPT-4o synthesis primary + eval baseline |
| anthropic | latest stable, May 2026 | Claude Sonnet diversity + Haiku judge |
| pydantic | `>=2.x` | JSON-schema validation in metrics |

Concrete pins land when the install is first exercised. The exclusion
ranges above come directly from Unsloth's
[`pyproject.toml`](https://github.com/unslothai/unsloth/blob/main/pyproject.toml),
so the matrix is upstream-validated even before our local install.

## Compatibility matrix (validated)

The matrix has two execution profiles. The linux + CUDA paid-run profile
is the authoritative one. The darwin + MPS smoke profile is a structural
check: same wiring, same data, same callbacks, but LoRA in BF16 instead
of QLoRA in NF4. QLoRA-specific behavior is only validated on the paid
run.

| Component | darwin arm64 smoke (TRL + Qwen 2.5 0.5B) | linux x86_64 paid (Unsloth + Llama 3.1 8B, RTX 4090, CUDA 12.6) | Source |
|---|---|---|---|
| Python | 3.12.x | 3.12.x | [PyTorch 2.7 release notes](https://pytorch.org/blog/pytorch-2-7/) |
| torch | per pyproject; MPS, no CUDA | per Unsloth wheel (`cu126`) | [Unsloth pyproject](https://github.com/unslothai/unsloth/blob/main/pyproject.toml) |
| CUDA toolkit | n/a (MPS) | 12.6 | [Unsloth wheels](https://github.com/unslothai/unsloth/blob/main/pyproject.toml) |
| transformers | 4.57.x (4.x line) | 4.57.x (4.x line, avoid 5.x) | [Transformers releases](https://github.com/huggingface/transformers/releases); [Migration to 5.x](https://medium.com/@vici0549/migrating-to-transformers-5-guide-1e90058a7633) |
| peft | 0.19.x | 0.19.x | [PEFT releases](https://github.com/huggingface/peft/releases) |
| trl | 1.x acceptable (smoke passes base model + LoraConfig, not pre-wrapped PeftModel) | 0.24.0 (Unsloth-capped) | [TRL releases](https://github.com/huggingface/trl/releases); [TRL #3926](https://github.com/huggingface/trl/issues/3926) |
| bitsandbytes | not used; smoke is LoRA, not QLoRA | 0.48.1 or 0.49.x; avoid 0.46.0 and 0.48.0 | [bitsandbytes releases](https://github.com/bitsandbytes-foundation/bitsandbytes/releases); [Unsloth pyproject](https://github.com/unslothai/unsloth/blob/main/pyproject.toml) |
| accelerate | 1.13.x | 1.13.x | [Accelerate releases](https://github.com/huggingface/accelerate/releases) |
| datasets | 3.6.x | 3.6.x | [Unsloth pyproject](https://github.com/unslothai/unsloth/blob/main/pyproject.toml) |
| unsloth | not used (Apple Silicon unsupported) | 2026.5.x | [Unsloth #4774](https://github.com/unslothai/unsloth/issues/4774); [Unsloth releases](https://github.com/unslothai/unsloth/releases) |
| flash-attn | not used | 2.8.3 (prebuilt for torch 2.x + CUDA 12.6 + Ada Lovelace) | [flash-attn PyPI](https://pypi.org/project/flash-attn/) |

**Known traps:**

- **transformers 5.x is partially landmined for Unsloth in May 2026.**
  Unsloth's pyproject excludes specific 5.0.0 / 5.1.0 builds but accepts
  `<=5.5.0` with caveats. PEFT < 0.18 is incompatible with transformers
  5. The conservative path is staying on the 4.x line (4.57.x) until
  Unsloth lifts the exclusions. Direct analog to Forge's vLLM-vs-
  transformers-5.x trap.
- **bitsandbytes excludes 0.46.0 and 0.48.0** per Unsloth's pyproject.
  Use 0.48.1 / 0.48.2 / 0.49.x. NF4 + double-quantization is stable from
  0.42 onward.
- **TRL on the paid run is capped at 0.24.0.** The smoke can use a newer
  TRL because it bypasses Unsloth. Both paths instantiate `SFTTrainer`
  with a base model + `LoraConfig`, never a pre-wrapped `PeftModel`, to
  avoid the adapter-freeze regression ([TRL
  #3926](https://github.com/huggingface/trl/issues/3926)).
- **Mac smoke runs non-quantized LoRA, not QLoRA.** A `bitsandbytes` MPS
  backend exists experimentally ([PR
  #1853](https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1853))
  but is not part of the supported Anvil path. The smoke validates
  config, data, and callbacks; numerics are paid-run only.
- **Unsloth install order matters.** RunPod images vary, and the
  recurring Unsloth ABI mismatch class (#221, #879, #1026, #1358) traces
  to `xformers` / `bitsandbytes` / CUDA / torch wheel mismatches. The
  documented install (`uv pip install unsloth --torch-backend=auto`) is
  the only reliable path. Phase 7 rehearsal validates the exact commands
  on a fresh RunPod image before the paid run.
