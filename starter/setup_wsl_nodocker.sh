#!/usr/bin/env bash
# Установка стека aij_robotics в WSL Ubuntu БЕЗ Docker.
# Зеркало Dockerfile (torch cu128, LLaMA-Factory, scaling-lawyer, lerobot, hf-libero),
# БЕЗ vendor_deps.py (он требует файлов с машин организаторов; third_party уже в kit).
#
# Запуск из КОРНЯ kit'а:  bash setup_wsl_nodocker.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIP="python3 -m pip"

echo "[0/6] python: $(python3 --version 2>&1 || true)"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  git curl ffmpeg cmake build-essential zip \
  libgl1 libglib2.0-0 libegl1 libgles2 \
  python3-pip python3-venv

$PIP install -q --break-system-packages -U pip

echo "[1/6] torch cu128 (CUDA 12.8 — подходит и для RTX 50xx)"
$PIP install --break-system-packages \
  "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" \
  --index-url https://download.pytorch.org/whl/cu128

echo "[2/6] LLaMA-Factory (editable)"
$PIP install --break-system-packages -e "$ROOT/third_party/LLaMA-Factory"

echo "[3/6] pinned deps (как в Dockerfile)"
$PIP install --break-system-packages \
  "transformers==5.3.0" "datasets==4.0.0" "accelerate==1.11.0" \
  "peft==0.18.1" "trl==0.24.0" "huggingface-hub==1.27.0" \
  "pyarrow==25.0.0" "einops==0.8.2" "safetensors==0.8.0" \
  "sentencepiece==0.2.2" "omegaconf==2.3.1" "tyro==0.8.14" \
  "av==15.1.0" "deepdiff==9.1.0" "torchcodec==0.15.0" "cmake==4.4.2" \
  "diffusers==0.39.0" "gymnasium==1.3.0" "imageio==2.37.4" \
  "imageio-ffmpeg==0.6.0" "pynput==1.8.2" "pyserial==3.5" \
  "rerun-sdk==0.35.0" "termcolor==3.3.0" "draccus==0.11.6" \
  "opencv-python-headless==5.0.0.93" "jsonlines==4.0.0" \
  "num2words==0.5.14" "PyYAML==6.0.3"

echo "[3b/6] deepspeed (нужен только для VLM-этапа; на этапе 1 можно без него)"
$PIP install --break-system-packages "deepspeed==0.19.4" \
  || echo "[warn] deepspeed не установился — этап 1 (VLA) работает без него; для этапа 2 починим"

echo "[4/6] lerobot + scaling-lawyer"
$PIP install --break-system-packages --no-deps "lerobot[smolvla]==0.5.1" \
  || $PIP install --break-system-packages --no-deps "lerobot==0.5.1"
$PIP install --break-system-packages -e "$ROOT/third_party/scaling-lawyer"

echo "[5/6] LIBERO self-check (egl-probe со СИСТЕМНЫМ cmake <4)"
PATH="/usr/bin:$PATH" $PIP install --break-system-packages --no-build-isolation \
  "hf-egl-probe==1.0.2" "egl-probe==1.0.2"
PATH="/usr/bin:$PATH" $PIP install --break-system-packages --no-build-isolation "hf-libero==0.1.4"

echo "[6/6] re-pin torch cu128"
$PIP install --break-system-packages \
  "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" \
  --index-url https://download.pytorch.org/whl/cu128

mkdir -p "${TRITON_CACHE_DIR:-/tmp/triton_cache_vla_hackathon}" "$ROOT/runs"
chmod +x "$ROOT"/scripts/*.sh

echo "=== import check ==="
python3 - <<'PY'
mods = ["torch", "llamafactory", "scaling_lawyer", "lerobot"]
for name in mods:
    m = __import__(name)
    print(f"{name}: OK")
import torch
print("cuda available:", torch.cuda.is_available())
from libero.libero import benchmark
assert "libero_spatial" in benchmark.get_benchmark_dict()
print("libero_spatial: OK")
PY
command -v llamafactory-cli >/dev/null && echo "llamafactory-cli: OK"
command -v lerobot-eval >/dev/null && echo "lerobot-eval: OK"
echo "DONE — стек установлен без Docker"
