# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional SmolVLM2 hang diagnostics. Enable with ``LLAMAFACTORY_SMOLVLM2_DEBUG=1``."""

from __future__ import annotations

import os
import time
from typing import Any

import torch

from . import logging


logger = logging.get_logger(__name__)

_ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})


def smolvlm2_debug_enabled() -> bool:
    return os.environ.get("LLAMAFACTORY_SMOLVLM2_DEBUG", "").strip().lower() in _ENABLED_VALUES


def smolvlm2_debug(msg: str, **kwargs: Any) -> None:
    if not smolvlm2_debug_enabled():
        return

    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    pid = os.getpid()
    if torch.cuda.is_available():
        device = f"cuda:{torch.cuda.current_device()}"
        mem = (
            f"mem_alloc={torch.cuda.memory_allocated() / 1e6:.0f}MB"
            f" mem_reserved={torch.cuda.memory_reserved() / 1e6:.0f}MB"
        )
    else:
        device = "cpu"
        mem = "mem_alloc=0MB mem_reserved=0MB"

    extras = " ".join(f"{key}={value}" for key, value in kwargs.items())
    logger.info_rank0(
        f"[SMOLVLM2_DEBUG] t={time.time():.3f} rank={rank} pid={pid} device={device} {mem} {msg} {extras}".strip()
    )


def smolvlm2_tensor_summary(batch: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in batch.items():
        if torch.is_tensor(value):
            parts.append(f"{key}:shape={tuple(value.shape)} dtype={value.dtype} device={value.device}")
        elif isinstance(value, list) and value and torch.is_tensor(value[0]):
            parts.append(f"{key}:list[{len(value)}] shape0={tuple(value[0].shape)}")
        else:
            parts.append(f"{key}:type={type(value).__name__}")
    return "; ".join(parts)
