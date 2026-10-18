#!/usr/bin/env bash
# LIBERO sim self-check for a submission/action_expert checkpoint.
# Default suite: libero_spatial (participant validation suite).
#
# Usage:
#   ./scripts/run_libero_selfcheck.sh runs/<run>/submission
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="${VLA_HACKATHON_ENV:-/home/jovyan/.mlspace/envs/vla-hackathon}"
if [[ -d "${ENV}/bin" ]]; then
  export PATH="${ENV}/bin:${PATH}"
  export CUDA_HOME="${CUDA_HOME:-${ENV}}"
fi

SUBMISSION="${1:-}"
if [[ -z "${SUBMISSION}" ]]; then
  echo "Usage: $0 <path-to-submission-or-action_expert>" >&2
  exit 2
fi
SUBMISSION="$(cd "${SUBMISSION}" && pwd)"
if [[ -d "${SUBMISSION}/action_expert" ]]; then
  POLICY="${SUBMISSION}/action_expert"
elif [[ -f "${SUBMISSION}/config.json" ]]; then
  POLICY="${SUBMISSION}"
else
  echo "Expected submission/ with action_expert/ or a policy checkpoint dir: ${SUBMISSION}" >&2
  exit 2
fi

SUITE="${SUITE:-libero_spatial}"
TASK_IDS="${TASK_IDS:-[0]}"
N_EPISODES="${N_EPISODES:-1}"
SEED="${SEED:-1000}"
GPU_INDEX="${GPU_INDEX:-0}"
if [[ -z "${OUT_DIR:-}" ]]; then
  if [[ "$(basename "${SUBMISSION}")" == "submission" ]]; then
    OUT_DIR="$(dirname "${SUBMISSION}")/libero_selfcheck"
  else
    OUT_DIR="${SUBMISSION}/libero_selfcheck"
  fi
fi
mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

# hf-libero prompts on first import if ~/.libero/config.yaml is missing (breaks Docker/CI).
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${HOME}/.libero}"
export LIBERO_CONFIG_PATH
mkdir -p "${LIBERO_CONFIG_PATH}"
if [[ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  python3 - <<'PY'
import pathlib
import yaml

pkg = pathlib.Path(__import__("libero").__file__).resolve().parent / "libero"
cfg_dir = pathlib.Path(__import__("os").environ["LIBERO_CONFIG_PATH"])
cfg = {
    "benchmark_root": str(pkg),
    "bddl_files": str(pkg / "bddl_files"),
    "init_states": str(pkg / "init_files"),
    "datasets": str(pkg.parent / "datasets"),
    "assets": str(pkg / "assets"),
}
(cfg_dir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
print(f"[selfcheck] wrote {cfg_dir / 'config.yaml'}")
PY
fi

if ! command -v lerobot-eval >/dev/null 2>&1; then
  echo "lerobot-eval not on PATH (activate vla-hackathon / rebuild Docker image)" >&2
  exit 1
fi
python3 -c "import libero" 2>/dev/null || {
  echo "Python package 'libero' missing. Re-run ./scripts/setup_env.sh (installs hf-libero)." >&2
  exit 1
}

echo "[selfcheck] policy=${POLICY}"
echo "[selfcheck] suite=${SUITE} task_ids=${TASK_IDS} n_episodes=${N_EPISODES}"
echo "[selfcheck] output=${OUT_DIR}"

# Quote task_ids so zsh does not glob the brackets.
lerobot-eval \
  --policy.path="${POLICY}" \
  --policy.device=cuda \
  --policy.use_amp=true \
  --env.type=libero \
  --env.task="${SUITE}" \
  --env.task_ids="${TASK_IDS}" \
  --env.control_mode=relative \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 \
  --eval.n_episodes="${N_EPISODES}" \
  --seed="${SEED}" \
  --output_dir="${OUT_DIR}"

echo "[selfcheck] metrics → ${OUT_DIR}/eval_info.json"
python3 - <<PY
import json
from pathlib import Path
p = Path("${OUT_DIR}") / "eval_info.json"
data = json.loads(p.read_text())
print(json.dumps(data.get("overall", data), indent=2))
PY
