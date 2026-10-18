#!/usr/bin/env bash
# One-command hackathon training:
#   1) SmolVLM2-500M SFT via LLaMA-Factory
#   2) SmolVLA action expert via scaling-lawyer
#
# Usage:
#   ./scripts/run_pipeline.sh configs/participant.yaml
#   ./scripts/run_pipeline.sh configs/participant.yaml --skip-vlm
#   ./scripts/run_pipeline.sh configs/participant.yaml --vlm-checkpoint /path/to/vlm

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-}"
shift || true

if [[ -z "${CONFIG}" ]]; then
  echo "Usage: $0 configs/participant.yaml [--skip-vlm|--skip-vla|--vlm-checkpoint PATH]" >&2
  exit 2
fi

if [[ ! -f "${CONFIG}" ]]; then
  # allow relative paths from repo root
  if [[ -f "${ROOT}/${CONFIG}" ]]; then
    CONFIG="${ROOT}/${CONFIG}"
  else
    echo "Config not found: ${CONFIG}" >&2
    exit 2
  fi
fi

PYTHON_BIN="${PIPELINE_PYTHON:-python3}"
exec "${PYTHON_BIN}" "${ROOT}/scripts/run_pipeline.py" --config "${CONFIG}" "$@"
