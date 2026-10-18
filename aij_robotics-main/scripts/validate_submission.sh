#!/usr/bin/env bash
# Load submission weights and run a tiny VLM + action-expert forward pass.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="${VLA_HACKATHON_ENV:-/home/jovyan/.mlspace/envs/vla-hackathon}"
if [[ -d "${ENV}/bin" ]]; then
  export PATH="${ENV}/bin:${PATH}"
  export CUDA_HOME="${CUDA_HOME:-${ENV}}"
fi
export PYTHONPATH="$ROOT/third_party/scaling-lawyer/src${PYTHONPATH:+:$PYTHONPATH}"

SUBMISSION="${1:-}"
if [[ -z "${SUBMISSION}" ]]; then
  echo "Usage: $0 runs/<run_name>/submission" >&2
  exit 2
fi

exec python3 "$ROOT/scripts/validate_submission.py" "$SUBMISSION" "${@:2}"
