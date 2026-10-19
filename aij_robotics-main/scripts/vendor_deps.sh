#!/usr/bin/env bash
# Wrapper around vendor_deps.py
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "${ROOT}/scripts/vendor_deps.py" "$@"
