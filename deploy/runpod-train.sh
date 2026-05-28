#!/usr/bin/env bash
# RunPod orchestrator: run preflight, then synth -> curate -> split -> train
# -> eval -> cost end-to-end. Designed to run on a fresh pod with zero
# manual intervention.
#
# Usage:
#   ./deploy/runpod-train.sh --rehearsal      # M1 dress rehearsal (no GPU,
#                                             # no paid APIs; fixture replay)
#   ./deploy/runpod-train.sh --full           # paid run; requires
#                                             # CONFIRM_PAID=1 + tokens

set -euo pipefail

REHEARSAL=0
FULL=0
PYTHON="${PYTHON:-uv run python}"

usage() {
    cat <<USAGE >&2
Usage: $0 [--rehearsal | --full]
  --rehearsal   Run the M1 smoke path: data-smoke -> train-smoke (dry) ->
                eval-smoke -> cost. Zero API spend, zero GPU.
  --full        Run the paid path: data-full -> train-full -> eval-full ->
                cost. Requires CONFIRM_PAID=1, OPENAI_API_KEY, HF_TOKEN.
USAGE
    exit 1
}

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
    REQUIRED_ENV=(OPENAI_API_KEY HF_TOKEN)
    for var in "${REQUIRED_ENV[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            echo "[runpod-train] full run requires \$${var}" >&2
            exit 1
        fi
    done
fi

echo "[runpod-train] preflight"
if (( FULL == 1 )); then
    $PYTHON -m anvil.preflight --mode full
else
    $PYTHON -m anvil.preflight --mode rehearsal
fi

if (( REHEARSAL == 1 )); then
    echo "[runpod-train] rehearsal: data-smoke"
    make data-smoke
    echo "[runpod-train] rehearsal: train-smoke (dry)"
    $PYTHON -m scripts.train --config configs/train-smoke.toml --dry-run
    echo "[runpod-train] rehearsal: eval-smoke"
    make eval-smoke
    echo "[runpod-train] rehearsal: cost"
    $PYTHON -m scripts.cost --config configs/cost-smoke.toml
    echo "[runpod-train] rehearsal complete"
    exit 0
fi

# --full path below.
echo "[runpod-train] full: data-full"
make data-full CONFIRM_PAID=1
echo "[runpod-train] full: train-full"
make train-full CONFIRM_PAID=1
echo "[runpod-train] full: eval (using same config + smoke fixture predictors as placeholder)"
echo "[runpod-train] (real eval-full config arrives once a paid eval slice is wired)"
make eval-smoke
echo "[runpod-train] full: cost"
$PYTHON -m scripts.cost --config configs/cost-smoke.toml
echo "[runpod-train] full complete"
