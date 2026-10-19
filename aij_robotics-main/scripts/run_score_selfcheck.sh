#!/usr/bin/env bash
# Participant Score self-check: 8× lmms-eval + libero_spatial → leaderboard_score.txt
#
# Usage:
#   ./scripts/run_score_selfcheck.sh --submission runs/<run>/submission
#   ./scripts/run_score_selfcheck.sh --submission runs/<run>/submission --smoke
#   ./scripts/run_score_selfcheck.sh --submission ... --config configs/selfcheck_score.example.yaml
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Prefer repo-local Score deps, then organizer envs.
for candidate in \
  "${ROOT}/third_party/lmms-eval/.venv/bin" \
  /home/jovyan/.mlspace/envs/lmms_eval/bin \
  /home/jovyan/.mlspace/envs/vla-hackathon/bin \
  /home/jovyan/.mlspace/envs/scaling-lawyer-core/bin
do
  if [[ -d "${candidate}" ]]; then
    export PATH="${candidate}:${PATH}"
  fi
done
if [[ -d "${ROOT}/data/erqa_converted" && -z "${ERQA_DATA_DIR:-}" ]]; then
  export ERQA_DATA_DIR="${ROOT}/data/erqa_converted"
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export API_TYPE="${API_TYPE:-dummy}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_aij_selfcheck}"
# Avoid ~/.local site-packages shadowing env pins (e.g. numpy 2.5 vs numba).
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
mkdir -p "${TRITON_CACHE_DIR}"

exec python3 "${ROOT}/scripts/run_score_selfcheck.py" "$@"
