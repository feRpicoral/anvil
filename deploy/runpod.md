# Running on RunPod

The paid run is fully orchestrated by `deploy/runpod-train.sh --full`. The
script is designed to run end-to-end on a fresh pod with zero manual
intervention; rehearse it on M1 first with `--rehearsal` (or
`make rehearse`).

## Pod spec

- **Template**: PyTorch 2.x base image (any recent NVIDIA / CUDA 12.x).
- **GPU**: RunPod RTX 4090 24 GB (Community tier; about $0.69/hr as of
  2026-05-27).
- **Disk**: >=40 GB volume (Llama 3.1 8B base + adapter + dataset + W&B
  cache).
- **Network**: outbound to `huggingface.co`, `api.openai.com`,
  `api.anthropic.com`, `api.wandb.ai`.

## Env vars (required)

| Variable | Why |
|---|---|
| `OPENAI_API_KEY` | Synthesis + eval baseline. |
| `HF_TOKEN` | Pull gated Llama 3.1 8B; push the trained adapter. |
| `WANDB_API_KEY` | Training-run telemetry (loss curves, eval_loss, lr). |
| `CONFIRM_PAID` | Set to `1` to acknowledge paid spend before triggering. |

Optional:

| Variable | Why |
|---|---|
| `ANTHROPIC_API_KEY` | Claude diversity slice during synthesis. |

## Reproducing

```bash
# 1. Fresh pod with PyTorch image; clone + install.
git clone https://github.com/feRpicoral/anvil.git
cd anvil
uv sync --group dev
uv pip install -c constraints/train.txt unsloth trl peft transformers \
    accelerate datasets bitsandbytes

# 2. Configure env. .env file or `export`s - pick one.
export OPENAI_API_KEY=sk-...
export HF_TOKEN=hf_...
export WANDB_API_KEY=...
export CONFIRM_PAID=1

# 3. Orchestrate end-to-end (synth -> curate -> split -> train -> eval -> cost).
./deploy/runpod-train.sh --full
```

## Artifact retrieval

After the run completes, copy these off the pod (gitignored locally):

- `outputs/full/` - LoRA adapter + tokenizer.
- `results/eval/full/` - per-variant scores + comparison.
- `results/cost/full.json` - cost report.

## Rehearsal

```bash
make rehearse              # bash deploy/runpod-train.sh --rehearsal
```

Validates the entire flow on M1 with zero API spend. Run it before every
paid run so a typo in the orchestrator doesn't burn budget.
