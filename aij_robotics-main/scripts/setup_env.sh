#!/usr/bin/env bash
# Install hackathon deps into conda env vla-hackathon (CUDA 12.6 + torch cu126 already present).
# Inside Docker the same steps run with system Python (see Dockerfile).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-vla-hackathon}"
ENV_PREFIX="${CONDA_PREFIX_OVERRIDE:-/home/jovyan/.mlspace/envs/${ENV_NAME}}"

if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
  PY="${ENV_PREFIX}/bin/python"
  PIP="${ENV_PREFIX}/bin/pip"
  export CUDA_HOME="${CUDA_HOME:-${ENV_PREFIX}}"
  export PATH="${ENV_PREFIX}/bin:${PATH}"
else
  PY="${PIPELINE_PYTHON:-$(command -v python3)}"
  PIP="${PIPELINE_PIP:-$(command -v pip3 || command -v pip)}"
fi

if [[ ! -x "${PY}" ]]; then
  echo "Python not found: ${PY}" >&2
  exit 1
fi

export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_vla_hackathon}"
mkdir -p "${TRITON_CACHE_DIR}"

echo "[0/4] Vendor third_party sources"
python3 "${ROOT}/scripts/vendor_deps.py"

echo "[1/4] Ensure torch stack stays cu126"
"${PIP}" install --break-system-packages \
  "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" \
  --index-url https://download.pytorch.org/whl/cu126

echo "[2/4] Install LLaMA-Factory (editable)"
"${PIP}" install --break-system-packages -e "${ROOT}/third_party/LLaMA-Factory"

echo "[3/4] Install scaling-lawyer + lerobot/smolvla stack"
# Keep torch pinned: install deps without letting pip upgrade torch.
"${PIP}" install --break-system-packages \
  "transformers==5.3.0" \
  "datasets>=2.19.0,<=4.0.0" \
  "accelerate>=1.3.0,<=1.11.0" \
  "peft>=0.18.0,<=0.18.1" \
  "trl>=0.18.0,<=0.24.0" \
  "huggingface-hub>=0.34" \
  "pyarrow>=18" \
  "deepspeed" \
  "einops" \
  "safetensors" \
  "sentencepiece" \
  "omegaconf" \
  "tyro<0.9.0" \
  "av<16.0.0" \
  "deepdiff" "torchcodec" "cmake" "diffusers" "gymnasium" \
  "imageio" "imageio-ffmpeg" "pynput" "pyserial" "rerun-sdk" \
  "termcolor" "draccus" "opencv-python-headless" "jsonlines" \
  "num2words"

"${PIP}" install --break-system-packages --no-deps "lerobot[smolvla]==0.5.1" || \
  "${PIP}" install --break-system-packages --no-deps "lerobot==0.5.1"

"${PIP}" install --break-system-packages \
  "draccus" "gymnasium" "imageio" "jsonlines" "rerun-sdk" "diffusers" || true

"${PIP}" install --break-system-packages -e "${ROOT}/third_party/scaling-lawyer"

# LIBERO sim self-check (lerobot-eval --env.type=libero). Matches lerobot[libero] pin.
# egl-probe needs system cmake (<4): pip's cmake 4.x rejects its CMakeLists.
echo "[3b/4] Install hf-libero for libero_spatial self-check"
(
  export PATH="/usr/bin:${PATH}"
  "${PIP}" install --break-system-packages --no-build-isolation \
    "hf-egl-probe==1.0.2" "egl-probe==1.0.2"
  "${PIP}" install --break-system-packages --no-build-isolation "hf-libero==0.1.4"
)

echo "[4/4] Re-pin torch after dependency installs"
"${PIP}" install --break-system-packages \
  "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" \
  --index-url https://download.pytorch.org/whl/cu126

echo "=== import check ==="
"${PY}" - <<'PY'
mods = [
    "torch", "torchaudio", "torchvision",
    "llamafactory", "scaling_lawyer", "deepspeed",
    "libero",
]
for name in mods:
    mod = __import__(name)
    ver = getattr(mod, "__version__", getattr(mod, "__file__", "?"))
    print(f"{name}: OK ({ver})")
import torch
print("cuda", torch.version.cuda, "available", torch.cuda.is_available())
from libero.libero import benchmark
assert "libero_spatial" in benchmark.get_benchmark_dict()
print("libero_spatial: OK")
PY

command -v llamafactory-cli || command -v "${ENV_PREFIX}/bin/llamafactory-cli" || true
echo "DONE"
