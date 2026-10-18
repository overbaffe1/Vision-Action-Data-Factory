#!/usr/bin/env bash
# End-to-end smoke as a participant:
#   L1 5 samples → VLM 1 epoch → LIBERO 5 episodes action expert → validate weights
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="${VLA_HACKATHON_ENV:-/home/jovyan/.mlspace/envs/vla-hackathon}"
if [[ -d "${ENV}/bin" ]]; then
  export PATH="${ENV}/bin:${PATH}"
  export CUDA_HOME="${CUDA_HOME:-${ENV}}"
fi
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_vla_hackathon}"
mkdir -p "$TRITON_CACHE_DIR"

CFG="${1:-$ROOT/configs/smoke.yaml}"
RUN_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CFG'))['run_name'])")
rm -rf "$ROOT/runs/$RUN_NAME"

echo "[smoke] full pipeline"
"$ROOT/scripts/run_pipeline.sh" "$CFG" --smoke

echo "[smoke] DONE → $ROOT/runs/$RUN_NAME/submission"
ls -la "$ROOT/runs/$RUN_NAME/submission"
echo "[smoke] validating submission weights"
"$ROOT/scripts/validate_submission.sh" "$ROOT/runs/$RUN_NAME/submission"
