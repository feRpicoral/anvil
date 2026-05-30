#!/usr/bin/env bash

set -euo pipefail

REHEARSAL=0
FULL=0
if [[ -n "${PYTHON:-}" ]]; then
    read -r -a PYTHON_CMD <<< "$PYTHON"
else
    PYTHON_CMD=(uv run python)
fi

usage() {
    cat <<USAGE >&2
Usage: $0 [--rehearsal | --full]
  --rehearsal   Run the M1 smoke path: data-smoke -> train-smoke (dry) ->
                eval-smoke -> cost. Zero API spend, zero GPU.
  --full        Run the paid path: data-full -> train-full -> eval-full ->
                cost-full. Requires CONFIRM_PAID=1 and paid-run env vars.
USAGE
    exit 1
}

if [[ -d /workspace ]]; then
    export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
    export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
    export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
    export WANDB_DIR="${WANDB_DIR:-/workspace/wandb}"
    export TMPDIR="${TMPDIR:-/workspace/tmp}"
    mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$WANDB_DIR" "$TMPDIR"
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --rehearsal) REHEARSAL=1; shift ;;
        --full)      FULL=1; shift ;;
        -h|--help)   usage ;;
        *)
            echo "[runpod-train] unknown arg: $1" >&2
            usage
            ;;
    esac
done

if (( REHEARSAL == 0 && FULL == 0 )); then
    echo "[runpod-train] must pass --rehearsal or --full" >&2
    usage
fi
if (( REHEARSAL == 1 && FULL == 1 )); then
    echo "[runpod-train] cannot combine --rehearsal and --full" >&2
    usage
fi

if (( FULL == 1 )); then
    if [[ "${CONFIRM_PAID:-}" != "1" ]]; then
        echo "[runpod-train] --full needs CONFIRM_PAID=1" >&2
        exit 1
    fi
    REQUIRED_ENV=(OPENAI_API_KEY HF_TOKEN WANDB_API_KEY)
    for var in "${REQUIRED_ENV[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            echo "[runpod-train] full run requires \$${var}" >&2
            exit 1
        fi
    done
fi

echo "[runpod-train] preflight"
if (( FULL == 1 )); then
    "${PYTHON_CMD[@]}" -m anvil.preflight --mode full
else
    "${PYTHON_CMD[@]}" -m anvil.preflight --mode rehearsal
fi

if (( REHEARSAL == 1 )); then
    echo "[runpod-train] rehearsal: data-smoke"
    make data-smoke
    echo "[runpod-train] rehearsal: train-smoke (dry)"
    "${PYTHON_CMD[@]}" -m scripts.train --config configs/train-smoke.toml --dry-run
    echo "[runpod-train] rehearsal: eval-smoke"
    make eval-smoke
    echo "[runpod-train] rehearsal: cost"
    "${PYTHON_CMD[@]}" -m scripts.cost --config configs/cost-smoke.toml
    echo "[runpod-train] rehearsal complete"
    exit 0
fi

echo "[runpod-train] full: data-full"
make data-full CONFIRM_PAID=1
echo "[runpod-train] full: train-full"
make train-full CONFIRM_PAID=1
echo "[runpod-train] full: eval-full + cost-full"
make cost-full CONFIRM_PAID=1
echo "[runpod-train] full complete"
