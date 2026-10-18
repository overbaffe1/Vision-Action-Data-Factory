#!/usr/bin/env python3
"""Hackathon training pipeline: LLaMA-Factory SmolVLM2 SFT → SmolVLA action expert.

Participants edit the participant YAML (ShareGPT + VLA dataset paths) and may freely
change training templates under configs/ (lr, steps, LoRA, batch size, …).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
# Prefer vendored real copies under third_party/ (no symlinks). Fallback to siblings.
DEFAULT_LLAMA_FACTORY = REPO_ROOT / "third_party" / "LLaMA-Factory"
DEFAULT_SCALING_LAWYER = REPO_ROOT / "third_party" / "scaling-lawyer"
FALLBACK_LLAMA_FACTORY = Path("/home/jovyan/people/murtazin/LLaMA-Factory")
FALLBACK_SCALING_LAWYER = Path("/home/jovyan/people/murtazin/scaling-lawyer")
# Host conda env (optional). In Docker / PATH installs we fall back to which()/sys.executable.
DEFAULT_ENV_BIN = Path("/home/jovyan/.mlspace/envs/vla-hackathon/bin")
BUNDLED_DEEPSPEED = REPO_ROOT / "configs" / "ds_z2_config.json"


def _resolve_executable(name: str, preferred: Path) -> Path:
    """Prefer an explicit path if it exists; otherwise use PATH / current interpreter."""
    if preferred.is_file():
        return preferred.resolve()
    if name == "python":
        return Path(sys.executable).resolve()
    found = shutil.which(name)
    if found:
        return Path(found).resolve()
    return preferred.expanduser().resolve()

SHAREGPT_ENTRY = {
    "formatting": "sharegpt",
    "columns": {"messages": "messages", "images": "images"},
    "tags": {
        "role_tag": "role",
        "content_tag": "content",
        "user_tag": "user",
        "assistant_tag": "assistant",
    },
}

VLA_SUITE_TEMPLATES = {
    "libero": "vla_libero.yaml",
}

VLA_SUITE_SMOKE_TEMPLATES = {
    "libero": "vla_libero.smoke.yaml",
}

# VLM training templates: smoke (--smoke) | lora | full (default).
VLM_PROTOCOL_TEMPLATES = {
    "full": "vlm_sft.yaml",
    "lora": "vlm_sft.lora.yaml",
}

# Expected LeRobot feature layout per suite (participant dataset must match).
VLA_SUITE_FEATURES: dict[str, dict[str, Any]] = {
    "libero": {
        "images": ["observation.images.image", "observation.images.image2"],
        "state_dim": 8,
        "action_dim": 7,
    },
}

CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def require_dir(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def validate_vla_dataset_for_suite(dataset_dir: Path, suite: str) -> dict[str, Any]:
    """Fail fast if local LeRobot meta does not match the selected VLA suite."""
    if suite not in VLA_SUITE_FEATURES:
        raise ValueError(f"Unknown vla.suite={suite!r}. Expected one of {sorted(VLA_SUITE_FEATURES)}")
    info_path = dataset_dir / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Expected LeRobot meta/info.json under {dataset_dir}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"Expected LeRobot dataset v3.0 under {dataset_dir}, "
            f"found codebase_version={info.get('codebase_version')!r}."
        )
    features = info.get("features") or {}
    expected = VLA_SUITE_FEATURES[suite]
    for key in expected["images"]:
        feat = features.get(key)
        if not isinstance(feat, dict):
            raise ValueError(
                f"vla.suite={suite!r} requires feature {key!r} in {info_path}. "
                f"Found keys={[k for k in features if str(k).startswith('observation.images.')]}."
            )
        dtype = feat.get("dtype")
        if dtype not in {"image", "video"}:
            raise ValueError(f"{key} dtype must be image/video, got {dtype!r} in {info_path}")

    state = features.get("observation.state") or {}
    action = features.get("action") or {}
    state_shape = list(state.get("shape") or [])
    action_shape = list(action.get("shape") or [])
    if not state_shape or int(state_shape[0]) != int(expected["state_dim"]):
        raise ValueError(
            f"vla.suite={suite!r} expects observation.state dim "
            f"{expected['state_dim']}, got shape={state_shape} in {info_path}"
        )
    if not action_shape or int(action_shape[0]) != int(expected["action_dim"]):
        raise ValueError(
            f"vla.suite={suite!r} expects action dim {expected['action_dim']}, "
            f"got shape={action_shape} in {info_path}"
        )

    # HuggingFaceVLA/libero advertises video_path even though visuals are dtype=image
    # and Hub has no videos/. Warn early if MP4s are missing.
    video_path = info.get("video_path")
    if video_path and not (dataset_dir / "videos").is_dir():
        image_dtypes = [
            (features.get(k) or {}).get("dtype") for k in expected["images"]
        ]
        if all(d == "image" for d in image_dtypes):
            print(
                "[warn] meta/info.json has video_path but no videos/ directory; "
                "visual features are dtype=image (OK for HuggingFaceVLA/libero). "
                "If LeRobot later looks for MP4s, null video_path in a local copy "
                "or use scripts/make_libero_hf_slice.py."
            )
    print(f"[ok] VLA dataset matches suite={suite}: images={expected['images']}")
    return info


def _pick_repo(preferred: Path, fallback: Path, label: str) -> Path:
    """Prefer a real (non-symlink) vendored tree; otherwise use fallback."""
    if preferred.exists():
        resolved = preferred.resolve()
        if preferred.is_symlink():
            raise RuntimeError(
                f"{label} at {preferred} is a symlink. "
                "Run ./scripts/vendor_deps.sh to create a real copy under third_party/, "
                "or set paths.llama_factory / paths.scaling_lawyer in the participant config."
            )
        return resolved
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(
        f"{label} not found at {preferred} or {fallback}. "
        "Run ./scripts/vendor_deps.sh or set an absolute path in the participant config."
    )


def resolve_tooling(participant: dict[str, Any]) -> dict[str, Path]:
    overrides = participant.get("paths") or {}
    if "llama_factory" in overrides:
        llama_factory = Path(overrides["llama_factory"]).expanduser().resolve()
    else:
        llama_factory = _pick_repo(DEFAULT_LLAMA_FACTORY, FALLBACK_LLAMA_FACTORY, "LLaMA-Factory")
    if "scaling_lawyer" in overrides:
        scaling_lawyer = Path(overrides["scaling_lawyer"]).expanduser().resolve()
    else:
        scaling_lawyer = _pick_repo(DEFAULT_SCALING_LAWYER, FALLBACK_SCALING_LAWYER, "scaling-lawyer")

    if "llamafactory_cli" in overrides:
        llamafactory_cli = Path(overrides["llamafactory_cli"]).expanduser().resolve()
    else:
        llamafactory_cli = _resolve_executable("llamafactory-cli", DEFAULT_ENV_BIN / "llamafactory-cli")
    if "scaling_lawyer_python" in overrides:
        scaling_lawyer_python = Path(overrides["scaling_lawyer_python"]).expanduser().resolve()
    else:
        scaling_lawyer_python = _resolve_executable("python", DEFAULT_ENV_BIN / "python")

    require_dir(llama_factory, "LLaMA-Factory")
    require_dir(scaling_lawyer, "scaling-lawyer")
    require_file(llamafactory_cli, "llamafactory-cli")
    require_file(scaling_lawyer_python, "scaling-lawyer python")
    return {
        "llama_factory": llama_factory,
        "scaling_lawyer": scaling_lawyer,
        "llamafactory_cli": llamafactory_cli,
        "scaling_lawyer_python": scaling_lawyer_python,
    }


def resolve_vlm_media_dir(participant: dict[str, Any], train_path: Path) -> Path:
    """Root for ShareGPT images/videos. Relative paths in JSONL are resolved against it."""
    raw = (participant.get("vlm") or {}).get("media_dir")
    if raw:
        return require_dir(Path(raw), "VLM media_dir")
    # Default: directory that contains the JSONL (so images/foo.jpg next to the file works).
    return train_path.expanduser().resolve().parent


def resolve_media_file(raw: str, media_dir: Path) -> Path | None:
    """Find a media file: as-is, under media_dir, or by basename under media_dir."""
    path = Path(raw).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append((media_dir / path).resolve())
    candidates.append((media_dir / path.name).resolve())
    # Also try stripping leading junk like datasets/.../images/name.jpg under media_dir
    parts = path.parts
    if "images" in parts:
        idx = parts.index("images")
        rel = Path(*parts[idx:])
        candidates.append((media_dir / rel).resolve())
        candidates.append((media_dir / Path(*parts[idx + 1 :])).resolve())
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def rewrite_sharegpt_jsonl(
    src: Path,
    dst: Path,
    *,
    media_dir: Path,
) -> Path:
    """Rewrite image/video paths so they resolve under media_dir (portable ShareGPT).

    Absolute foreign paths (e.g. /home/jovyan/.../0001.jpg) are remapped by basename
    when the file exists in media_dir. Relative paths are kept relative to media_dir.
    """
    media_dir = media_dir.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    rewritten = 0
    missing = 0
    total = 0
    with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for key in ("images", "videos", "audios"):
                items = obj.get(key)
                if not items:
                    continue
                if not isinstance(items, list):
                    items = [items]
                new_items: list[str] = []
                for raw in items:
                    total += 1
                    found = resolve_media_file(str(raw), media_dir)
                    if found is None:
                        missing += 1
                        new_items.append(str(raw))
                        continue
                    try:
                        new_items.append(str(found.relative_to(media_dir)))
                    except ValueError:
                        new_items.append(str(found))
                    if str(found) != str(Path(raw).expanduser()):
                        rewritten += 1
                obj[key] = new_items
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(
        f"[media] {src.name}: remapped={rewritten}, missing={missing}, total={total}, "
        f"media_dir={media_dir}"
    )
    if missing:
        raise FileNotFoundError(
            f"{missing} media file(s) from {src} were not found under media_dir={media_dir}. "
            "Put images next to the JSONL (or under media_dir) and use relative paths "
            "like 'images/0001.jpg', or set vlm.media_dir explicitly."
        )
    return dst


def register_sharegpt_dataset(
    registry_dir: Path,
    *,
    train_path: Path,
    eval_path: Path | None,
    llama_factory: Path,
) -> tuple[str, str | None]:
    """Write a per-run dataset_info.json. Never touches LLaMA-Factory/data/dataset_info.json."""
    registry_dir = registry_dir.resolve()
    lf_data = (llama_factory / "data").resolve()
    if registry_dir == lf_data or lf_data in registry_dir.parents:
        raise RuntimeError(
            "Refusing to write dataset registry inside LLaMA-Factory/data/. "
            "Participants must not edit the shared dataset_info.json."
        )

    registry_dir.mkdir(parents=True, exist_ok=True)
    info: dict[str, Any] = {
        "hackathon_vlm_train": {
            "file_name": str(train_path),
            **SHAREGPT_ENTRY,
        }
    }
    eval_name = None
    if eval_path is not None:
        eval_name = "hackathon_vlm_eval"
        info[eval_name] = {
            "file_name": str(eval_path),
            **SHAREGPT_ENTRY,
        }
    (registry_dir / "dataset_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return "hackathon_vlm_train", eval_name


def build_vlm_train_yaml(
    *,
    template_path: Path,
    tooling: dict[str, Path],
    run_dir: Path,
    registry_dir: Path,
    train_name: str,
    eval_name: str | None,
    media_dir: Path,
) -> Path:
    cfg = load_yaml(template_path)
    # Smoke/local configs may set deepspeed: false to avoid ZeRO on single GPU.
    if cfg.get("deepspeed") is False:
        cfg.pop("deepspeed", None)
    else:
        deepspeed = require_file(BUNDLED_DEEPSPEED, "Bundled DeepSpeed ZeRO-2 config")
        cfg["deepspeed"] = str(deepspeed)
    cfg["dataset"] = train_name
    # Per-run registry — participants never edit LLaMA-Factory/data/dataset_info.json
    cfg["dataset_dir"] = str(registry_dir)
    # LLaMA-Factory joins media_dir + relative image paths from ShareGPT JSONL.
    cfg["media_dir"] = str(media_dir)
    cfg["output_dir"] = str(run_dir / "vlm" / "checkpoints")
    if eval_name:
        cfg["do_eval"] = True
        cfg["eval_dataset"] = eval_name
        cfg["per_device_eval_batch_size"] = int(cfg.get("per_device_train_batch_size", 1))
        cfg["eval_strategy"] = "steps"
        # Keep eval aligned with checkpointing so short smoke runs still evaluate.
        cfg["eval_steps"] = int(cfg.get("save_steps", 500))
        cfg["load_best_model_at_end"] = True
        cfg["greater_is_better"] = False
        cfg["metric_for_best_model"] = "eval_loss"
    else:
        cfg["do_eval"] = False
        cfg.pop("eval_dataset", None)

    out = run_dir / "generated" / "vlm_sft.yaml"
    dump_yaml(out, cfg)
    return out


def build_vla_train_yaml(
    *,
    template_path: Path,
    suite: str,
    participant: dict[str, Any],
    run_dir: Path,
    vlm_checkpoint: Path,
    dataset_dir: Path,
) -> Path:
    if suite not in VLA_SUITE_TEMPLATES:
        raise ValueError(f"Unknown vla.suite={suite!r}. Expected one of {sorted(VLA_SUITE_TEMPLATES)}")

    cfg = load_yaml(template_path)
    runtime = participant["runtime"]
    gpu_indices = [int(x) for x in runtime["gpu_indices"]]

    cfg["runtime"] = {
        "gpu_indices": gpu_indices,
        "cache_dir": str(Path(runtime["cache_dir"]).expanduser().resolve()),
        "lerobot_home": str(Path(runtime["lerobot_home"]).expanduser().resolve()),
        "log_file": str(run_dir / "logs" / "vla.log"),
    }
    cfg["paths"] = {
        "vlm_checkpoint": str(vlm_checkpoint),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(run_dir / "action_expert" / "checkpoints"),
    }

    out = run_dir / "generated" / f"vla_{suite}.yaml"
    dump_yaml(out, cfg)
    return out


def is_lora_adapter(path: Path) -> bool:
    return path.is_dir() and (path / "adapter_config.json").is_file() and (
        (path / "adapter_model.safetensors").is_file()
        or (path / "adapter_model.bin").is_file()
        or any(path.glob("adapter_model*.safetensors"))
    )


def is_full_hf_checkpoint(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    return (
        any(path.glob("model*.safetensors"))
        or any(path.glob("pytorch_model*.bin"))
        or (path / "model.safetensors.index.json").is_file()
    )


def is_hf_checkpoint(path: Path) -> bool:
    return is_full_hf_checkpoint(path) or is_lora_adapter(path)


def merge_lora_vlm_checkpoint(adapter_dir: Path, merged_dir: Path) -> Path:
    """Merge PEFT LoRA adapter into a full HF folder for SmolVLA load_vlm_weights."""
    if is_full_hf_checkpoint(merged_dir):
        print(f"[ok] reusing merged VLM: {merged_dir}")
        return merged_dir

    import json

    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    base_name = adapter_cfg.get("base_model_name_or_path")
    if not base_name:
        raise FileNotFoundError(f"No base_model_name_or_path in {adapter_dir / 'adapter_config.json'}")

    print(f"[merge] base={base_name} adapter={adapter_dir} → {merged_dir}")
    merged_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(adapter_dir, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        base_name,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()
    model.save_pretrained(merged_dir)
    processor.save_pretrained(merged_dir)
    print(f"[ok] merged VLM written to {merged_dir}")
    return merged_dir


def resolve_vlm_checkpoint(vlm_output_dir: Path) -> Path:
    """Pick a HuggingFace-format checkpoint produced by LLaMA-Factory."""
    if is_hf_checkpoint(vlm_output_dir):
        return vlm_output_dir

    checkpoints = sorted(
        (p for p in vlm_output_dir.glob("checkpoint-*") if p.is_dir() and is_hf_checkpoint(p)),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if checkpoints:
        return checkpoints[-1]

    raise FileNotFoundError(
        f"No HuggingFace VLM checkpoint found under {vlm_output_dir}. "
        "Expected config.json + model weights, or adapter_config.json + adapter_model*."
    )


def prepare_vlm_for_vla(vlm_checkpoint: Path, run_dir: Path) -> Path:
    """SmolVLA needs a full HF VLM folder; merge LoRA adapters when needed."""
    if is_lora_adapter(vlm_checkpoint) and not is_full_hf_checkpoint(vlm_checkpoint):
        return merge_lora_vlm_checkpoint(vlm_checkpoint, run_dir / "vlm" / "merged")
    return vlm_checkpoint


def resolve_action_expert_checkpoint(vla_output_dir: Path) -> Path:
    """Pick the latest LeRobot pretrained_model directory."""
    numeric: list[tuple[int, Path]] = []
    for path in vla_output_dir.glob("checkpoints/*/pretrained_model"):
        if not path.is_dir():
            continue
        name = path.parent.name
        if name.isdigit():
            numeric.append((int(name), path))
    if numeric:
        numeric.sort(key=lambda item: item[0])
        return numeric[-1][1]

    last = vla_output_dir / "checkpoints" / "last" / "pretrained_model"
    if last.is_dir():
        return last
    raise FileNotFoundError(f"No action-expert checkpoint found under {vla_output_dir}")


def _env_library_path(env_root: Path, existing: str | None = None) -> str:
    """Prefer pip nvidia cudart + conda env libs over a mismatched system CUDA."""
    parts: list[str] = []
    nvidia_cudart = (
        env_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
        / "cuda_runtime"
        / "lib"
    )
    # Also try glob for site-packages path via known layout.
    candidates = [
        nvidia_cudart,
        env_root / "lib",
        env_root / "lib64",
    ]
    # Discover nvidia/*/lib under site-packages
    sp = env_root / "lib"
    for py_dir in sp.glob("python*/site-packages/nvidia/*/lib"):
        candidates.append(py_dir)
    for path in candidates:
        if path.is_dir():
            parts.append(str(path))
    if existing:
        parts.append(existing)
    # dedupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return os.pathsep.join(out)


def _training_env(
    *,
    python_or_cli: Path,
    gpu_indices: list[int],
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    env = os.environ.copy()
    env_root = python_or_cli.parent.parent
    bin_dir = python_or_cli.parent
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    # Prefer a real CUDA toolkit (nvcc). Conda-style installs keep CUDA under env_root.
    cuda_home = Path(env.get("CUDA_HOME", "/usr/local/cuda"))
    if not (cuda_home / "bin" / "nvcc").is_file():
        if (env_root / "bin" / "nvcc").is_file():
            cuda_home = env_root
        elif Path("/usr/local/cuda/bin/nvcc").is_file():
            cuda_home = Path("/usr/local/cuda")
    env["CUDA_HOME"] = str(cuda_home)
    env["LD_LIBRARY_PATH"] = _env_library_path(env_root, env.get("LD_LIBRARY_PATH"))
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_indices)
    env.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache_vla_hackathon")
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("COMET_MODE", "disabled")
    env.setdefault("MASTER_ADDR", "127.0.0.1")
    if extra:
        env.update(extra)
    return env


def run_command(command: list[str], *, cwd: Path, env: dict[str, str], log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] cwd={cwd}")
    print(f"[run] {' '.join(command)}")
    print(f"[run] log={log_file}")
    with log_file.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}: {' '.join(command)}")


def _ensure_action_expert_config_type(action_dir: Path) -> None:
    """Ensure LeRobot config.json has policy type (needed by libero_eval / from_pretrained)."""
    config_path = action_dir / "config.json"
    if not config_path.is_file():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("type"):
        return
    policy_type = "smolvla"
    train_path = action_dir / "train_config.json"
    if train_path.is_file():
        train_config = json.loads(train_path.read_text(encoding="utf-8"))
        nested = train_config.get("policy") if isinstance(train_config, dict) else None
        if isinstance(nested, dict) and nested.get("type"):
            policy_type = str(nested["type"])
    config["type"] = policy_type
    config_path.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
    print(f"[collect] patched missing config.type={policy_type!r} in {config_path}")


def collect_submission(
    *,
    run_dir: Path,
    vlm_checkpoint: Path,
    action_checkpoint: Path,
    participant: dict[str, Any],
    participant_config: Path,
) -> Path:
    submission = run_dir / "submission"
    vlm_dst = submission / "smolvlm2"
    action_dst = submission / "action_expert"

    if submission.exists():
        shutil.rmtree(submission)
    submission.mkdir(parents=True)

    print(f"[collect] copying VLM → {vlm_dst}")
    def _ignore_trainer_checkpoints(directory: str, names: list[str]) -> set[str]:
        # Drop huge Trainer/DeepSpeed step dirs; keep the exported HF weights.
        return {n for n in names if n.startswith("checkpoint-") or n.startswith("global_step")}

    shutil.copytree(vlm_checkpoint, vlm_dst, ignore=_ignore_trainer_checkpoints)
    print(f"[collect] copying action expert → {action_dst}")
    shutil.copytree(action_checkpoint, action_dst)
    _ensure_action_expert_config_type(action_dst)

    meta = {
        "run_name": participant["run_name"],
        "participant_config": str(participant_config.resolve()),
        "base_vlm": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "vlm_data": participant["vlm"]["data_path"],
        "vla_suite": participant["vla"]["suite"],
        "vla_dataset_dir": participant["vla"]["dataset_dir"],
        "artifacts": {
            "smolvlm2": str(vlm_dst),
            "action_expert": str(action_dst),
            "vlm_source_checkpoint": str(vlm_checkpoint),
            "action_source_checkpoint": str(action_checkpoint),
        },
    }
    (submission / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(participant_config, submission / "participant.yaml")
    return submission


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Participant YAML (paths to ShareGPT data + LIBERO dataset).",
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Reuse an existing VLM checkpoint under the run directory.",
    )
    parser.add_argument(
        "--skip-vla",
        action="store_true",
        help="Only run VLM SFT (no action expert).",
    )
    parser.add_argument(
        "--vlm-checkpoint",
        type=Path,
        default=None,
        help="Optional existing VLM checkpoint (implies --skip-vlm).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and write generated configs without training.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use configs/*.smoke.yaml (short VLM/VLA run for local checks).",
    )
    parser.add_argument(
        "--vlm-protocol",
        choices=sorted(VLM_PROTOCOL_TEMPLATES),
        default=None,
        help="VLM training template: full (default) or lora. Overridden by --smoke. "
        "Also settable via participant vlm.protocol. Templates live in configs/.",
    )
    args = parser.parse_args()

    participant_config = require_file(args.config, "Participant config")
    participant = load_yaml(participant_config)

    run_name = str(participant["run_name"]).strip()
    if not run_name:
        raise ValueError("run_name must be a non-empty string")

    runtime = participant["runtime"]
    output_root = Path(runtime["output_root"]).expanduser().resolve()
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "generated").mkdir(exist_ok=True)

    tooling = resolve_tooling(participant)
    dump_yaml(run_dir / "generated" / "participant.resolved.yaml", participant)

    train_path = require_file(Path(participant["vlm"]["data_path"]), "VLM ShareGPT data")
    eval_raw = participant["vlm"].get("eval_data_path")
    eval_path = require_file(Path(eval_raw), "VLM eval data") if eval_raw else None
    media_dir = resolve_vlm_media_dir(participant, train_path)

    # Portable ShareGPT: remap absolute foreign image paths via media_dir / basename.
    sharegpt_dir = run_dir / "generated" / "sharegpt"
    train_path = rewrite_sharegpt_jsonl(
        train_path,
        sharegpt_dir / "train.jsonl",
        media_dir=media_dir,
    )
    if eval_path is not None:
        eval_path = rewrite_sharegpt_jsonl(
            eval_path,
            sharegpt_dir / "eval.jsonl",
            media_dir=media_dir,
        )

    suite = str(participant["vla"]["suite"]).strip()
    vla_dataset = require_dir(Path(participant["vla"]["dataset_dir"]), "VLA LeRobot dataset")
    validate_vla_dataset_for_suite(vla_dataset, suite)

    # --- Stage 1: VLM ---
    registry_dir = run_dir / "generated" / "lf_dataset_registry"
    train_name, eval_name = register_sharegpt_dataset(
        registry_dir,
        train_path=train_path,
        eval_path=eval_path,
        llama_factory=tooling["llama_factory"],
    )
    if args.smoke:
        vlm_template = CONFIG_DIR / "vlm_sft.smoke.yaml"
    else:
        protocol = str(
            args.vlm_protocol
            or (participant.get("vlm") or {}).get("protocol")
            or "full"
        ).strip().lower()
        if protocol not in VLM_PROTOCOL_TEMPLATES:
            raise ValueError(
                f"Unknown vlm.protocol={protocol!r}. Expected one of "
                f"{sorted(VLM_PROTOCOL_TEMPLATES)} (or pass --smoke)."
            )
        vlm_template = CONFIG_DIR / VLM_PROTOCOL_TEMPLATES[protocol]
    vla_template = CONFIG_DIR / (
        VLA_SUITE_SMOKE_TEMPLATES[suite] if args.smoke else VLA_SUITE_TEMPLATES[suite]
    )
    if args.smoke:
        print("[smoke] using configs:", vlm_template.name, vla_template.name)
    else:
        print("[vlm] protocol config:", vlm_template.name)

    vlm_yaml = build_vlm_train_yaml(
        template_path=vlm_template,
        tooling=tooling,
        run_dir=run_dir,
        registry_dir=registry_dir,
        train_name=train_name,
        eval_name=eval_name,
        media_dir=media_dir,
    )

    if args.vlm_checkpoint is not None:
        vlm_checkpoint = require_dir(args.vlm_checkpoint, "Provided VLM checkpoint")
        args.skip_vlm = True
    elif args.skip_vlm:
        vlm_checkpoint = resolve_vlm_checkpoint(run_dir / "vlm" / "checkpoints")
    elif args.dry_run:
        vlm_checkpoint = run_dir / "vlm" / "checkpoints"
        print(f"[dry-run] would train VLM with {vlm_yaml}")
    else:
        gpu_indices = [int(i) for i in runtime["gpu_indices"]]
        # Single-GPU: avoid torchrun/elastic (more stable on WSL + Blackwell).
        vlm_extra = {"FORCE_TORCHRUN": "1"} if len(gpu_indices) > 1 else {}
        env = _training_env(
            python_or_cli=tooling["llamafactory_cli"],
            gpu_indices=gpu_indices,
            extra=vlm_extra,
        )
        run_command(
            [str(tooling["llamafactory_cli"]), "train", str(vlm_yaml)],
            cwd=tooling["llama_factory"],
            env=env,
            log_file=run_dir / "logs" / "vlm.log",
        )
        vlm_checkpoint = resolve_vlm_checkpoint(run_dir / "vlm" / "checkpoints")

    print(f"[ok] VLM checkpoint: {vlm_checkpoint}")

    if args.skip_vla:
        print("[ok] --skip-vla set; stopping after VLM.")
        return 0

    # --- Stage 2: action expert ---
    if not args.dry_run:
        vlm_checkpoint = prepare_vlm_for_vla(vlm_checkpoint, run_dir)
        print(f"[ok] VLM for VLA: {vlm_checkpoint}")

    vla_yaml = build_vla_train_yaml(
        template_path=vla_template,
        suite=suite,
        participant=participant,
        run_dir=run_dir,
        vlm_checkpoint=vlm_checkpoint,
        dataset_dir=vla_dataset,
    )

    if args.dry_run:
        print(f"[dry-run] would train action expert with {vla_yaml}")
        print(f"[dry-run] generated configs under {run_dir / 'generated'}")
        return 0

    env = _training_env(
        python_or_cli=tooling["scaling_lawyer_python"],
        gpu_indices=[int(i) for i in runtime["gpu_indices"]],
        extra={
            "PYTHONPATH": str(tooling["scaling_lawyer"] / "src")
            + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
        },
    )
    run_command(
        [str(tooling["scaling_lawyer_python"]), "-m", "scaling_lawyer.vla", str(vla_yaml)],
        cwd=tooling["scaling_lawyer"],
        env=env,
        log_file=run_dir / "logs" / "vla.log",
    )

    action_checkpoint = resolve_action_expert_checkpoint(run_dir / "action_expert" / "checkpoints")
    print(f"[ok] Action expert checkpoint: {action_checkpoint}")

    submission = collect_submission(
        run_dir=run_dir,
        vlm_checkpoint=vlm_checkpoint,
        action_checkpoint=action_checkpoint,
        participant=participant,
        participant_config=participant_config,
    )
    print(f"[done] Submission artifacts: {submission}")
    print(f"  - smolvlm2:      {submission / 'smolvlm2'}")
    print(f"  - action_expert: {submission / 'action_expert'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 — top-level CLI
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
