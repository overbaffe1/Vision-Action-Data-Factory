from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    config = yaml.safe_load(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def require(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing config key: {section}.{key}")
    return mapping[key]


def path_value(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path string.")
    return Path(os.path.expandvars(value)).expanduser().resolve()


def require_file(path: str | Path, label: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise FileNotFoundError(f"{label} not found: {value}")
    return value


def require_dir(path: str | Path, label: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_dir():
        raise FileNotFoundError(f"{label} not found: {value}")
    return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def configured_environment(
    *,
    gpu_index: int | None = None,
    gpu_indices: list[int] | None = None,
    cache_dir: Path,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    if gpu_indices is None:
        if gpu_index is None:
            raise ValueError("gpu_index or gpu_indices is required.")
        gpu_indices = [gpu_index]
    if not gpu_indices:
        raise ValueError("gpu_indices must not be empty.")
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": ",".join(str(index) for index in gpu_indices),
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HOME": str(cache_dir),
            "HF_DATASETS_CACHE": str(cache_dir / "datasets"),
            "HUGGINGFACE_HUB_CACHE": str(cache_dir / "hub"),
        }
    )
    if extra:
        environment.update(extra)
    return environment


def run_logged(
    command: Iterable[str],
    *,
    log_file: str | Path,
    environment: dict[str, str] | None = None,
) -> None:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command_list = [str(item) for item in command]
    print("Command:", " ".join(command_list))
    with log_path.open("a", buffering=1) as log:
        process = subprocess.Popen(
            command_list,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        return_code = process.wait()
    if return_code:
        raise SystemExit(return_code)


class TeeStream:
    def __init__(self, primary, log) -> None:
        self.primary = primary
        self.log = log

    def write(self, value: str) -> int:
        self.primary.write(value)
        self.log.write(value)
        return len(value)

    def flush(self) -> None:
        self.primary.flush()
        self.log.flush()

    def __getattr__(self, name: str):
        return getattr(self.primary, name)


def enable_tee(log_file: str | Path):
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", buffering=1)
    sys.stdout = TeeStream(sys.stdout, handle)
    sys.stderr = TeeStream(sys.stderr, handle)
    return handle
