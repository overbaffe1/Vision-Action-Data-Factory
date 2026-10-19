# Participant training image: SmolVLM2 SFT + SmolVLA action expert.
# Requires NVIDIA Container Toolkit (--gpus all).

# cu128 required for RTX 5080 / Blackwell (sm_120); cu126 lacks kernels.
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    TRITON_CACHE_DIR=/tmp/triton_cache_vla_hackathon \
    HF_HOME=/root/.cache/huggingface \
    WORKSPACE=/workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      ffmpeg \
      cmake \
      build-essential \
      libgl1 \
      libglib2.0-0 \
      libegl1 \
      libgles2 \
      software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
      python3.12 \
      python3.12-dev \
      python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/local/bin/python3 \
    && ln -sf /usr/local/bin/pip3.12 /usr/local/bin/pip \
    && ln -sf /usr/local/bin/pip3.12 /usr/local/bin/pip3

WORKDIR ${WORKSPACE}

# Copy repo (third_party included; lmms-eval is not on main).
COPY . ${WORKSPACE}

# Install stack (torch cu128 + LF + SL + lerobot). Skip re-vendor: sources already in image.
RUN ln -sf /usr/local/cuda/bin/nvcc /usr/local/bin/nvcc \
    && pip install --break-system-packages \
      "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" \
      --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --break-system-packages -e ${WORKSPACE}/third_party/LLaMA-Factory \
    && pip install --break-system-packages \
      "transformers==5.3.0" \
      "datasets==4.0.0" \
      "accelerate==1.11.0" \
      "peft==0.18.1" \
      "trl==0.24.0" \
      "huggingface-hub==1.27.0" \
      "pyarrow==25.0.0" \
      "deepspeed==0.19.4" \
      "einops==0.8.2" \
      "safetensors==0.8.0" \
      "sentencepiece==0.2.2" \
      "omegaconf==2.3.1" \
      "tyro==0.8.14" \
      "av==15.1.0" \
      "deepdiff==9.1.0" \
      "torchcodec==0.15.0" \
      "cmake==4.4.2" \
      "diffusers==0.39.0" \
      "gymnasium==1.3.0" \
      "imageio==2.37.4" \
      "imageio-ffmpeg==0.6.0" \
      "pynput==1.8.2" \
      "pyserial==3.5" \
      "rerun-sdk==0.35.0" \
      "termcolor==3.3.0" \
      "draccus==0.11.6" \
      "opencv-python-headless==5.0.0.93" \
      "jsonlines==4.0.0" \
      "num2words==0.5.14" \
      "PyYAML==6.0.3" \
    && (pip install --break-system-packages --no-deps "lerobot[smolvla]==0.5.1" \
        || pip install --break-system-packages --no-deps "lerobot==0.5.1") \
    && pip install --break-system-packages -e ${WORKSPACE}/third_party/scaling-lawyer \
    && PATH="/usr/bin:$PATH" pip install --break-system-packages --no-build-isolation \
      "hf-egl-probe==1.0.2" "egl-probe==1.0.2" \
    && PATH="/usr/bin:$PATH" pip install --break-system-packages --no-build-isolation \
      "hf-libero==0.1.4" \
    && pip install --break-system-packages \
      "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" \
      --index-url https://download.pytorch.org/whl/cu128 \
    && mkdir -p ${TRITON_CACHE_DIR} ${WORKSPACE}/runs \
    && chmod +x ${WORKSPACE}/scripts/*.sh

# Default participant config path (mount your own over this file).
ENV PATH="/usr/local/cuda/bin:/usr/local/bin:${PATH}"

CMD ["./scripts/run_pipeline.sh", "configs/participant.yaml"]
