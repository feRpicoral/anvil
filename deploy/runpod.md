# RunPod deployment guide

This document is the reproduction recipe for the Anvil paid run and the
pre-flight checklist that protects the API and GPU budget. The full run is
orchestrated by `deploy/runpod-train.sh --full`; run the rehearsal locally
before starting the pod.

## Pod spec

| Setting | Value | Why |
|---|---|---|
| GPU | **RTX A5000 24 GB** | Existing 24 GB pod; enough for Llama 3.1 8B QLoRA. |
| Image | PyTorch 2.x base image with CUDA 12.x+ | Keeps CUDA, Python, and PyTorch aligned before installing Unsloth. |
| Container disk | **20 GB** | Keep model weights, uv cache, W&B, temp files, and outputs off this disk. |
| Volume | **100 GB mounted at `/workspace`** | Room for Llama cache, synthesized data, checkpoints, final adapter, W&B, logs, and transfer archives. |
| Network | outbound to `huggingface.co`, `api.openai.com`, `api.wandb.ai` | Required for model access, synthesis/eval, and telemetry. |
| Idle shutdown | 15 min | Guard against forgotten billing after the run. |

If reusing the Forge pod, delete `/workspace/forge` only after Forge results
are already copied locally and committed. Treat `/workspace/anvil` as the only
project directory for this run.

## Pod environment

Set these before starting the pod. Changing pod-level env vars after startup
can restart the pod.

Secrets:

| Variable | Why |
|---|---|
| `OPENAI_API_KEY` | Paid GPT-4o synthesis and GPT-4o eval baseline. |
| `HF_TOKEN` | Pull gated Llama 3.1 8B weights and publish the adapter. Needs read access to the model and write access to the target repo. |
| `WANDB_API_KEY` | Training telemetry, loss curves, eval loss, and learning rate. |

Optional secret:

| Variable | Why |
|---|---|
| `ANTHROPIC_API_KEY` | Only needed if a synthesis config uses the Anthropic backend. Current full config does not. |

Plain env vars:

| Variable | Value |
|---|---|
| `CONFIRM_PAID` | `1` |
| `HF_HOME` | `/workspace/.cache/huggingface` |
| `XDG_CACHE_HOME` | `/workspace/.cache` |
| `UV_CACHE_DIR` | `/workspace/.cache/uv` |
| `UV_LINK_MODE` | `copy` |
| `WANDB_DIR` | `/workspace/wandb` |
| `TMPDIR` | `/workspace/tmp` |
| `TOKENIZERS_PARALLELISM` | `false` |

`deploy/runpod-train.sh` sets workspace-backed defaults for the cache and temp
dirs when `/workspace` exists, but pod-level env vars are still the safer path:
the background training process inherits them, and model/cache downloads do not
fall back to `/root`.

## Local pre-flight

Run this on the Mac before starting the paid pod:

```bash
cd /Users/fpicoral/dev/anvil
git status --short
uv sync --frozen --group dev
make check
bash deploy/runpod-train.sh --rehearsal
```

The rehearsal runs `data-smoke`, a dry training config check, `eval-smoke`, and
`cost` without API spend or GPU use.

## On-pod setup

After SSH-ing into the pod:

```bash
cd /workspace
mkdir -p /workspace/.cache/huggingface /workspace/.cache/uv /workspace/wandb /workspace/tmp

git clone https://github.com/feRpicoral/anvil.git
cd anvil

uv sync --frozen --group dev
uv pip install -c constraints/train.txt \
  unsloth trl peft transformers accelerate datasets bitsandbytes
```

Verify the dependency state before spending money:

```bash
uv run python - <<'PY'
import numpy
import unsloth
import bitsandbytes
import datasets
import peft
import torch
import transformers
import trl

print("numpy", numpy.__version__)
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("transformers", transformers.__version__)
print("trl", trl.__version__)
print("peft", peft.__version__)
print("datasets", datasets.__version__)
print("bitsandbytes", bitsandbytes.__version__)
print("unsloth", getattr(unsloth, "__version__", "installed"))
PY
```

Expected: NumPy is below `2.3`, CUDA is available, and the GPU is the RTX
A5000. If NumPy is `2.3+`, rerun the install command above before starting the
paid run.

Verify Hugging Face access without printing the token:

```bash
uv run python - <<'PY'
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
print("user:", api.whoami()["name"])
print("model:", api.model_info("meta-llama/Llama-3.1-8B-Instruct").id)
PY
```

Run preflight:

```bash
uv run python -m anvil.preflight --mode full
```

## Paid run

Run detached from SSH:

```bash
cd /workspace/anvil
mkdir -p logs
nohup bash deploy/runpod-train.sh --full > logs/runpod-full.log 2>&1 &
tail -f logs/runpod-full.log
```

Once the log shows the run has started, you can disconnect. The pod must keep
running, but the SSH session does not.

Useful status checks:

```bash
ps -eo pid,etime,cmd | grep -E 'runpod-train|scripts\.(synth|train|eval|cost)' | grep -v grep
nvidia-smi
df -h / /workspace
du -sh /root/.cache /workspace/.cache /workspace/anvil /workspace/wandb 2>/dev/null
tail -120 logs/runpod-full.log
```

If `/root/.cache` grows while `/workspace/.cache` does not, stop and fix the
environment before continuing.

## Artifact export

After the full run completes, verify the expected files exist:

```bash
cd /workspace/anvil
test -d outputs/full/final
test -s results/eval/full/comparison.json
test -s results/cost/full.json
find data/full outputs/full results/eval/full results/cost logs -maxdepth 3 -type f | sort | head -80
```

Create a single archive on the pod:

```bash
cd /workspace/anvil
tar -czf /workspace/anvil-results.tgz \
  -C /workspace/anvil \
  data/full \
  outputs/full \
  results/eval/full \
  results/cost/full.json \
  logs \
  -C /workspace \
  wandb

tar -tzf /workspace/anvil-results.tgz | head -80
sha256sum /workspace/anvil-results.tgz
```

Copy from the Mac using the direct IP/port shown by RunPod. Prefer direct SCP
over `ssh.runpod.io`; the proxy can inject banner text and corrupt binary
streams.

```bash
cd /Users/fpicoral/dev/anvil
scp -O -P <port> -i ~/.ssh/id_ed25519_runpod \
  root@<ip>:/workspace/anvil-results.tgz \
  anvil-results.tgz

shasum -a 256 anvil-results.tgz
tar -tzf anvil-results.tgz | head -80
tar -xzf anvil-results.tgz
```

If SCP fails, use RunPod's file browser for `/workspace/anvil-results.tgz`.
Avoid raw `ssh cat` for the archive unless stdout is guaranteed clean.

After the archive is copied and checksums match, stop/delete the pod.

## Local post-run

Back on the Mac:

```bash
cd /Users/fpicoral/dev/anvil
uv run python -m scripts.cost --config configs/cost-full.toml
uv run python -m scripts.chart \
  --eval-comparison results/eval/full/comparison.json \
  --cost-report results/cost/full.json \
  --output-dir docs/img
make check
```

Then update the README headline and methodology from the measured artifacts
before opening the results PR. Do not commit transfer archives, W&B run
directories, local tokens, or `DECISIONS.md`.
