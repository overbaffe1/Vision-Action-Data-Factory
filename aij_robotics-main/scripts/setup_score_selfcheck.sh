#!/usr/bin/env bash
# Install Score self-check deps into this repo (no private GitLab required):
#   1) public EvolvingLMMs-Lab/lmms-eval @ pinned SHA
#   2) apply third_party/lmms_eval_overlay (smolvlm2, run_many_tasks, erqa, msmu)
#   3) FlagEval/ERQA → data/erqa_converted
#
# Usage (from repo root):
#   ./scripts/setup_score_selfcheck.sh
#   ./scripts/setup_score_selfcheck.sh --skip-erqa
#   ./scripts/setup_score_selfcheck.sh --force-erqa
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

OVERLAY="${ROOT}/third_party/lmms_eval_overlay"
BASE_REF="${OVERLAY}/BASE_REF.txt"
LMMS_DIR="${LMMS_EVAL_DIR:-${ROOT}/third_party/lmms-eval}"
ERQA_OUT="${ERQA_DATA_DIR:-${ROOT}/data/erqa_converted}"
SKIP_ERQA=0
SKIP_LMMS=0
FORCE_ERQA=0

# Defaults from BASE_REF.txt (overridable via env)
LMMS_URL="${LMMS_EVAL_URL:-https://github.com/EvolvingLMMs-Lab/lmms-eval.git}"
LMMS_SHA="${LMMS_EVAL_SHA:-}"
if [[ -z "${LMMS_SHA}" && -f "${BASE_REF}" ]]; then
  LMMS_SHA="$(awk -F= '/^upstream_sha=/{print $2}' "${BASE_REF}" | tr -d '[:space:]')"
fi
if [[ -z "${LMMS_SHA}" ]]; then
  LMMS_SHA="75571cfe0fa36ca5ec2e6aa2bb268f9d38295af3"
fi

for arg in "$@"; do
  case "${arg}" in
    --skip-erqa) SKIP_ERQA=1 ;;
    --skip-lmms) SKIP_LMMS=1 ;;
    --force-erqa) FORCE_ERQA=1 ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: ${arg}" >&2
      exit 2
      ;;
  esac
done

PY="${PYTHON:-}"
if [[ -z "${PY}" ]]; then
  if command -v python3.13 >/dev/null 2>&1; then
    PY="$(command -v python3.13)"
  elif command -v python3.12 >/dev/null 2>&1; then
    PY="$(command -v python3.12)"
  else
    PY="$(command -v python3)"
  fi
fi

echo "[setup] repo=${ROOT}"
echo "[setup] python=${PY}"
echo "[setup] lmms-eval ${LMMS_URL} @ ${LMMS_SHA}"

# --- lmms-eval (public) + overlay ---
if [[ "${SKIP_LMMS}" -eq 0 ]]; then
  if [[ ! -d "${OVERLAY}/scripts" ]]; then
    echo "ERROR: missing overlay at ${OVERLAY}" >&2
    exit 1
  fi

  if [[ ! -d "${LMMS_DIR}/.git" ]]; then
    echo "[setup] cloning public lmms-eval @ ${LMMS_SHA} → ${LMMS_DIR}"
    mkdir -p "$(dirname "${LMMS_DIR}")"
    rm -rf "${LMMS_DIR}"
    mkdir -p "${LMMS_DIR}"
    (
      cd "${LMMS_DIR}"
      git init
      git remote add origin "${LMMS_URL}"
      git fetch --depth 1 origin "${LMMS_SHA}"
      git checkout --force FETCH_HEAD
    )
  else
    echo "[setup] updating existing checkout ${LMMS_DIR} → ${LMMS_SHA}"
    (
      cd "${LMMS_DIR}"
      git remote set-url origin "${LMMS_URL}" 2>/dev/null || git remote add origin "${LMMS_URL}"
      git fetch --depth 1 origin "${LMMS_SHA}"
      git checkout --force FETCH_HEAD
    )
  fi

  bash "${ROOT}/scripts/apply_lmms_eval_overlay.sh" "${LMMS_DIR}"

  if [[ ! -f "${LMMS_DIR}/scripts/run_many_tasks.py" ]]; then
    echo "ERROR: run_many_tasks.py missing after overlay" >&2
    exit 1
  fi

  VENV="${LMMS_DIR}/.venv"
  if [[ ! -x "${VENV}/bin/python" ]]; then
    echo "[setup] creating venv ${VENV}"
    if command -v uv >/dev/null 2>&1; then
      # Prefer uv: system python -m venv often lacks ensurepip on shared hosts.
      uv venv "${VENV}" --python "${PY}"
    else
      "${PY}" -m venv "${VENV}" || "${PY}" -m venv --without-pip "${VENV}"
    fi
  fi
  if command -v uv >/dev/null 2>&1; then
    echo "[setup] using uv for installs"
    uv pip install --python "${VENV}/bin/python" --break-system-packages -U pip setuptools wheel
    uv pip install --python "${VENV}/bin/python" --break-system-packages -e "${LMMS_DIR}"
    # SmolVLM2 processor + common eval extras (not always pulled by lmms-eval meta).
    uv pip install --python "${VENV}/bin/python" --break-system-packages \
      num2words sentencepiece protobuf
    # Match host driver (CUDA 12.x): default torch from PyPI may be cu130 and disable GPU.
    uv pip install --python "${VENV}/bin/python" --break-system-packages \
      --index-url https://download.pytorch.org/whl/cu128 \
      "torch==2.10.0" "torchvision==0.25.0" || true
  else
    # shellcheck disable=SC1091
    source "${VENV}/bin/activate"
    python -m pip install --break-system-packages -U pip setuptools wheel
    python -m pip install --break-system-packages -e "${LMMS_DIR}"
    python -m pip install --break-system-packages num2words sentencepiece protobuf
    python -m pip install --break-system-packages \
      --index-url https://download.pytorch.org/whl/cu128 \
      "torch==2.10.0" "torchvision==0.25.0" || true
    deactivate || true
  fi
  echo "[setup] lmms python: ${VENV}/bin/python"
else
  echo "[setup] skip lmms-eval"
fi

# --- ERQA ---
if [[ "${SKIP_ERQA}" -eq 0 ]]; then
  echo "[setup] preparing ERQA → ${ERQA_OUT}"
  ERQA_PY="${LMMS_DIR}/.venv/bin/python"
  if [[ ! -x "${ERQA_PY}" ]]; then
    ERQA_PY="${PY}"
  fi
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "${ERQA_PY}" --break-system-packages datasets pillow huggingface_hub
  else
    "${ERQA_PY}" -m pip install --break-system-packages datasets pillow huggingface_hub
  fi
  FORCE_FLAG=()
  if [[ "${FORCE_ERQA}" -eq 1 ]]; then
    FORCE_FLAG=(--force)
  fi
  "${ERQA_PY}" "${ROOT}/scripts/prepare_erqa.py" --output-dir "${ERQA_OUT}" "${FORCE_FLAG[@]}"
else
  echo "[setup] skip ERQA"
fi

cat <<EOF

[setup] done (public lmms-eval + in-repo overlay; no private GitLab).

Default config paths (configs/selfcheck_score.example.yaml):
  paths.lmms_eval:     third_party/lmms-eval
  paths.lmms_python:   third_party/lmms-eval/.venv/bin/python
  paths.erqa_data_dir: data/erqa_converted

Run Score self-check:
  export API_TYPE=dummy MUJOCO_GL=egl
  ./scripts/run_score_selfcheck.sh \\
    --submission runs/<run>/submission \\
    --config configs/selfcheck_score.example.yaml

EOF
