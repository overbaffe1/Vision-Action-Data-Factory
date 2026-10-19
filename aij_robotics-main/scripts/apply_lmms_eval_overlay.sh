#!/usr/bin/env bash
# Apply third_party/lmms_eval_overlay onto a public lmms-eval checkout.
#
# Usage:
#   ./scripts/apply_lmms_eval_overlay.sh [path-to-lmms-eval]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OVERLAY="${ROOT}/third_party/lmms_eval_overlay"
TARGET="${1:-${ROOT}/third_party/lmms-eval}"

if [[ ! -d "${OVERLAY}/scripts" ]]; then
  echo "ERROR: overlay missing: ${OVERLAY}" >&2
  exit 1
fi
if [[ ! -d "${TARGET}/lmms_eval" ]]; then
  echo "ERROR: not an lmms-eval tree: ${TARGET}" >&2
  exit 1
fi

echo "[overlay] → ${TARGET}"

mkdir -p "${TARGET}/scripts" \
  "${TARGET}/lmms_eval/models/simple" \
  "${TARGET}/lmms_eval/tasks"

cp -a "${OVERLAY}/scripts/run_many_tasks.py" "${TARGET}/scripts/"
cp -a "${OVERLAY}/lmms_eval/models/simple/smolvlm2.py" "${TARGET}/lmms_eval/models/simple/"
rm -rf "${TARGET}/lmms_eval/tasks/erqa" "${TARGET}/lmms_eval/tasks/msmu"
cp -a "${OVERLAY}/lmms_eval/tasks/erqa" "${TARGET}/lmms_eval/tasks/"
cp -a "${OVERLAY}/lmms_eval/tasks/msmu" "${TARGET}/lmms_eval/tasks/"

# Register SmolVLM2 in AVAILABLE_SIMPLE_MODELS (idempotent).
python3 - <<PY
from pathlib import Path

init_path = Path("${TARGET}") / "lmms_eval" / "models" / "__init__.py"
text = init_path.read_text(encoding="utf-8")
needle = '"smolvlm2": "SmolVLM2"'
if needle in text:
    print("[overlay] smolvlm2 already registered")
else:
    marker = '"idefics2": "Idefics2",'
    if marker not in text:
        raise SystemExit(f"Cannot find insertion marker in {init_path}")
    text = text.replace(marker, marker + "\n    " + needle + ",", 1)
    init_path.write_text(text, encoding="utf-8")
    print("[overlay] registered smolvlm2 in models/__init__.py")
PY

# Sanity
test -f "${TARGET}/scripts/run_many_tasks.py"
test -f "${TARGET}/lmms_eval/models/simple/smolvlm2.py"
test -f "${TARGET}/lmms_eval/tasks/erqa/erqa.yaml"
test -f "${TARGET}/lmms_eval/tasks/msmu/msmu.yaml"
echo "[overlay] OK"
