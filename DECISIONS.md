# Decisions

Forward-looking record of choices and rationale for Anvil. Mirrors the format
of [`forge/DECISIONS.md`](../forge/DECISIONS.md). Status tokens are plain text
(`planned` / `in progress` / `partial` / `complete` / `awaiting GPU`) per the
user's global instructions.

## Project status

| Phase | Goal | Status |
|---|---|---|
| 0 | Repo bootstrap (public repo, GitHub governance) | in progress |
| 1 | Tooling foundation (uv, Ruff, mypy, pytest, pre-commit, CI, Makefile) | in progress |
| 2 | Research + `DECISIONS.md` (framework, technique, compat matrix, budget envelope) | planned |
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
- A serving / inference optimization piece — that is Forge.
- A multi-task fine-tuning framework — one task, depth over breadth.
- RLHF / preference tuning, unless Phase 2 research shows a clear win on the
  chosen task.

## Stack overview

Filled in Phase 2 with researched, justified choices. Placeholders below for
the load-bearing slots:

| Layer | Choice | Status |
|---|---|---|
| Language | Python 3.12 | locked |
| Dependency manager | uv (+ `uv.lock`) | locked |
| Lint / format | Ruff (line 100) | locked |
| Type checker | mypy strict | locked |
| Tests | pytest + pytest-cov | locked |
| CI | GitHub Actions (no GPU) | locked |
| Pre-commit | pre-commit + commitlint | locked |
| Base model | Llama 3.1 8B Instruct (Phase 2 confirm) | working assumption |
| PEFT technique | QLoRA (NF4 + LoRA adapters) | working assumption |
| Training framework | Unsloth (paid) + TRL `SFTTrainer` (smoke) | Phase 2 |
| Dataset synthesis | OpenAI GPT-4o + 5% Anthropic Claude diversity | Phase 2 |
| Experiment tracking | Weights & Biases | Phase 2 |
| GPU cloud | RunPod RTX 4090 Community | Phase 2 |
| Publishing | Hugging Face Hub | Phase 2 |
| Task | Structured extraction from legal contracts (NDAs / MSAs) | working assumption |

## Decisions (Phase 2 will expand this section)

Phase 1 ships the scaffold only. Phase 2 lands per-decision subsections with
rationale, rejected alternatives, and source links, mirroring Forge's
`DECISIONS.md` structure.

### Vendor-from-Forge policy

Anvil treats Forge as a sibling repo and **vendors** specific Forge modules
(rather than path-depending or publishing them to PyPI) when the same code
serves both repos. Vendored modules carry a header comment
`# Vendored from forge@<commit-sha>` that pins the source.

Planned vendor scope (lands in the phase noted):

- `forge/cost/model.py` and `forge/cost/pricing.py` → `anvil/cost/inference_cost.py` (Phase 6)
- `forge/plots/style.py` → `anvil/plots/style.py` (Phase 9)
- Inference cost-model tests carry over alongside the vendored module.

Rationale: path-deps break on fresh clones of either repo; PyPI publish for
Forge is out of scope. Vendor + cite is the lowest-friction durable choice.

### Forge-numbers reference policy

Forge's current benchmark numbers are illustrative pending its Phase 7 paid
run. Anvil's README **never quotes Forge throughput / cost / latency
numbers** until that lands. When referring to the serving sibling, Anvil uses
capability phrasing only — for example, "served by a vLLM + AWQ stack like
Forge's; throughput estimate Y tok/s based on [published source]" — with the
estimate anchored in a *published* Llama 3.1 8B / 4090 source cited in this
file.

## Pinned versions

Filled after the first successful `uv sync --group dev` in Phase 1, and
extended in Phase 2 after the training stack is installed against
`constraints/train.txt`.

## Compatibility matrix (validated)

Filled in Phase 2 after the training stack is chosen and the matrix is
validated on both `darwin/arm64` (smoke) and `linux/x86_64` (paid GPU).
