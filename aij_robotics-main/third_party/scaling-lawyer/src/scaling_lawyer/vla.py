from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .common import (
    configured_environment,
    load_yaml,
    path_value,
    require_dir,
    require_file,
    run_logged,
    write_json,
)
from .subsets import select_task_balanced_episodes


def resolve_runtime_gpus(runtime: dict[str, Any]) -> tuple[list[int], int]:
    if "gpu_indices" in runtime:
        gpu_indices = [int(value) for value in runtime["gpu_indices"]]
        if not gpu_indices:
            raise ValueError("runtime.gpu_indices must not be empty.")
        return gpu_indices, len(gpu_indices)

    gpu_index = int(runtime.get("gpu_index", 0))
    num_gpus = int(runtime.get("num_gpus", 1))
    if num_gpus < 1:
        raise ValueError("runtime.num_gpus must be positive.")
    return list(range(gpu_index, gpu_index + num_gpus)), num_gpus


def resolve_train_entrypoint(command: list[str]) -> list[str]:
    if not command or command[0] != "lerobot-train":
        return command
    entrypoint = shutil.which("lerobot-train")
    if entrypoint is None:
        raise RuntimeError("lerobot-train is not installed.")
    return [entrypoint, *command[1:]]


def wrap_train_command(command: list[str], num_gpus: int) -> list[str]:
    command = resolve_train_entrypoint(command)
    if num_gpus <= 1:
        return command
    if shutil.which("accelerate") is None:
        raise RuntimeError("accelerate is not installed.")
    accelerate = shutil.which("accelerate")
    assert accelerate is not None
    return [
        accelerate,
        "launch",
        "--num_processes",
        str(num_gpus),
        "--multi_gpu",
        "--mixed_precision",
        "bf16",
        *command,
    ]


def task_names(value: Any) -> list[str]:
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def resolve_dataset_dir(paths_config: dict[str, Any]) -> Path:
    if "dataset_dir" in paths_config:
        return path_value(paths_config["dataset_dir"], "paths.dataset_dir")
    if "libero_dir" in paths_config:
        return path_value(paths_config["libero_dir"], "paths.libero_dir")
    raise KeyError("paths must include dataset_dir (or legacy libero_dir).")


def load_dataset_info(dataset_dir: Path) -> dict[str, Any]:
    info_file = dataset_dir / "meta" / "info.json"
    if not info_file.is_file():
        raise FileNotFoundError(f"LeRobot dataset metadata not found: {info_file}")
    info = json.loads(info_file.read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError(
            f"Expected LeRobot dataset v3.0, found {info.get('codebase_version')!r}."
        )
    return info


def validate_libero_like_features(info: dict[str, Any], *, dataset_dir: Path) -> None:
    """Soft checks shared by LIBERO protocol datasets (image/image2, state 8, action 7)."""
    features = info.get("features") or {}
    for key in ("observation.images.image", "observation.images.image2"):
        if key not in features:
            return
    state = features.get("observation.state") or {}
    action = features.get("action") or {}
    state_dim = int((state.get("shape") or [0])[0])
    action_dim = int((action.get("shape") or [0])[0])
    if state_dim and state_dim != 8:
        raise ValueError(
            f"LIBERO-like dataset at {dataset_dir} has observation.state dim {state_dim}, expected 8."
        )
    if action_dim and action_dim != 7:
        raise ValueError(
            f"LIBERO-like dataset at {dataset_dir} has action dim {action_dim}, expected 7."
        )
    if info.get("video_path") and not (dataset_dir / "videos").is_dir():
        print(
            "[warn] dataset info has video_path but videos/ is missing; "
            "HuggingFaceVLA/libero stores PNG images in parquet (dtype=image)."
        )


def load_episode_tasks(
    dataset_dir: Path,
    *,
    expected_episodes: int | None = None,
) -> list[tuple[int, str]]:
    episode_files = sorted((dataset_dir / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise FileNotFoundError(f"Episode metadata not found under {dataset_dir}")
    episodes = pd.concat((pd.read_parquet(path) for path in episode_files), ignore_index=True)
    result = []
    for row in episodes.to_dict("records"):
        names = task_names(row["tasks"])
        if not names:
            raise ValueError(f"Episode {row['episode_index']} has no task.")
        result.append((int(row["episode_index"]), names[0]))
    if expected_episodes is not None and len(result) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} episodes, found {len(result)}.")
    return result


def build_train_config(
    config: dict[str, Any],
    paths: dict[str, Path],
    episodes: list[int],
    *,
    total_episodes: int | None = None,
) -> dict[str, Any]:
    dataset = config["dataset"]
    policy = config["policy"]
    optimizer = config["optimizer"]
    training = config["training"]
    # Passing an explicit list of all episode ids forces LeRobot to filter every
    # parquet shard and can hang multi-GPU startup past the NCCL barrier timeout.
    # When training on the full dataset, omit `episodes` (load everything).
    use_full_dataset = total_episodes is not None and len(episodes) == int(total_episodes)
    dataset_cfg: dict[str, Any] = {
        "repo_id": dataset["repo_id"],
        "root": str(paths["dataset_dir"]),
        "revision": dataset["revision"],
        "use_imagenet_stats": dataset["use_imagenet_stats"],
        "video_backend": dataset.get("video_backend", "pyav"),
    }
    if not use_full_dataset:
        dataset_cfg["episodes"] = episodes
    return {
        "dataset": dataset_cfg,
        "policy": {
            "type": "smolvla",
            "vlm_model_name": str(paths["vlm_checkpoint"]),
            "load_vlm_weights": policy["load_vlm_weights"],
            "train_expert_only": policy["train_expert_only"],
            "freeze_vision_encoder": policy["freeze_vision_encoder"],
            "train_state_proj": policy["train_state_proj"],
            "chunk_size": policy["chunk_size"],
            "n_action_steps": policy["n_action_steps"],
            "num_vlm_layers": policy["num_vlm_layers"],
            "num_expert_layers": policy["num_expert_layers"],
            "expert_width_multiplier": policy["expert_width_multiplier"],
            "attention_mode": policy["attention_mode"],
            "self_attn_every_n_layers": policy["self_attn_every_n_layers"],
            "device": policy["device"],
            "push_to_hub": False,
            "optimizer_lr": optimizer["learning_rate"],
            "optimizer_betas": optimizer["betas"],
            "optimizer_eps": optimizer["epsilon"],
            "optimizer_weight_decay": optimizer["weight_decay"],
            "optimizer_grad_clip_norm": optimizer["grad_clip_norm"],
            "scheduler_warmup_steps": optimizer["warmup_steps"],
            "scheduler_decay_steps": optimizer["decay_steps"],
            "scheduler_decay_lr": optimizer["decay_learning_rate"],
        },
        "output_dir": str(paths["output_dir"]),
        "job_name": training["job_name"],
        "steps": training["steps"],
        "batch_size": training["batch_size"],
        "num_workers": training["num_workers"],
        "seed": training["seed"],
        "log_freq": training["log_freq"],
        "save_checkpoint": True,
        "save_freq": training["save_freq"],
        "eval_freq": 0,
        "wandb": {"enable": False},
        "resume": False,
    }


def stage_train_config(
    output_dir: Path,
    train_config: dict[str, Any],
    *,
    resume: bool,
) -> Path:
    staged_config = output_dir.with_name(f"{output_dir.name}.train_config.json")
    if staged_config.is_file():
        existing_config = json.loads(staged_config.read_text())
        if existing_config != train_config:
            raise RuntimeError(
                f"Existing train config differs from this run: {staged_config}. "
                "Use the original settings or a new output directory."
            )
    if resume and not staged_config.is_file() and not (output_dir / "train_config.json").is_file():
        raise FileNotFoundError(
            f"Cannot resume because train config is missing for output directory: {output_dir}"
        )
    write_json(staged_config, train_config)
    return staged_config


def fresh_command(train_config_path: Path) -> list[str]:
    return ["lerobot-train", f"--config_path={train_config_path}"]


def stage_episode_manifest(
    output_dir: Path,
    manifest: dict[str, Any],
    *,
    resume: bool,
) -> tuple[Path, Path]:
    run_manifest = output_dir / "episode_manifest.json"
    staged_manifest = output_dir.with_name(f"{output_dir.name}.episode_manifest.json")

    for manifest_path in (run_manifest, staged_manifest):
        if manifest_path.is_file():
            existing_manifest = json.loads(manifest_path.read_text())
            if existing_manifest != manifest:
                raise RuntimeError(
                    f"Existing episode manifest differs from this config: {manifest_path}. "
                    "Use the original subset settings or a new output directory."
                )

    if not resume and output_dir.exists():
        contents = list(output_dir.iterdir()) if output_dir.is_dir() else []
        if output_dir.is_dir() and contents == [run_manifest]:
            # Older scaling-lawyer versions created this file before invoking
            # LeRobot, which then rejected the otherwise empty output directory.
            run_manifest.unlink()
            output_dir.rmdir()
        else:
            raise FileExistsError(
                f"Output directory already exists and resume is false: {output_dir}"
            )

    if resume and not output_dir.is_dir():
        raise FileNotFoundError(f"Cannot resume because output directory is missing: {output_dir}")

    write_json(staged_manifest, manifest)
    return staged_manifest, run_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = load_yaml(args.config)
    runtime = config["runtime"]
    dataset_dir = resolve_dataset_dir(config["paths"])
    paths = {
        "vlm_checkpoint": path_value(config["paths"]["vlm_checkpoint"], "paths.vlm_checkpoint"),
        "dataset_dir": dataset_dir,
        "output_dir": path_value(config["paths"]["output_dir"], "paths.output_dir"),
    }
    require_dir(paths["vlm_checkpoint"], "VLM checkpoint")
    require_dir(paths["dataset_dir"], "LeRobot dataset")
    if shutil.which("lerobot-train") is None:
        raise RuntimeError("lerobot-train is not installed.")

    dataset_config = config["dataset"]
    dataset_info = load_dataset_info(paths["dataset_dir"])
    validate_libero_like_features(dataset_info, dataset_dir=paths["dataset_dir"])
    expected_episodes = dataset_info.get("total_episodes")
    count = int(dataset_config["episode_count"])
    subset_seed = int(dataset_config["subset_seed"])
    episode_tasks = load_episode_tasks(
        paths["dataset_dir"],
        expected_episodes=expected_episodes,
    )
    if count > len(episode_tasks):
        raise ValueError(
            f"dataset.episode_count={count} exceeds local episodes={len(episode_tasks)} "
            f"under {paths['dataset_dir']}. For HuggingFaceVLA/libero the full protocol "
            "expects all 1693 episodes downloaded (LeRobot v3.0)."
        )
    episodes = select_task_balanced_episodes(episode_tasks, count, subset_seed)
    task_by_episode = dict(episode_tasks)
    manifest = {
        "dataset_repo_id": dataset_config["repo_id"],
        "dataset_dir": str(paths["dataset_dir"]),
        "episode_count": count,
        "subset_seed": subset_seed,
        "episodes": [
            {"episode_index": episode, "task": task_by_episode[episode]}
            for episode in episodes
        ],
    }

    training = config["training"]
    staged_manifest, run_manifest = stage_episode_manifest(
        paths["output_dir"],
        manifest,
        resume=bool(training["resume"]),
    )
    staged_train_config: Path | None = None
    if training["resume"]:
        resume_config = require_file(
            path_value(training["resume_config_path"], "training.resume_config_path"),
            "Resume train_config.json",
        )
        command = ["lerobot-train", f"--config_path={resume_config}", "--resume=true"]
    else:
        train_config = build_train_config(
            config,
            paths,
            episodes,
            total_episodes=expected_episodes,
        )
        staged_train_config = stage_train_config(
            paths["output_dir"],
            train_config,
            resume=False,
        )
        command = fresh_command(staged_train_config)

    gpu_indices, num_gpus = resolve_runtime_gpus(runtime)
    command = wrap_train_command(command, num_gpus)

    cache_dir = path_value(runtime["cache_dir"], "runtime.cache_dir")
    lerobot_home = path_value(runtime["lerobot_home"], "runtime.lerobot_home")
    environment = configured_environment(
        gpu_indices=gpu_indices,
        cache_dir=cache_dir,
        extra={
            "HF_LEROBOT_HOME": str(lerobot_home),
            **(
                {
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": "29500",
                }
                if num_gpus > 1
                else {}
            ),
        },
    )
    print(
        f"Selected {len(episodes)} nested, task-balanced episodes from "
        f"{dataset_config['repo_id']} ({expected_episodes} total)."
    )
    if num_gpus > 1:
        per_device_batch = int(config["training"]["batch_size"])
        print(
            f"Launching distributed training on {num_gpus} GPUs "
            f"(effective batch size {per_device_batch * num_gpus})."
        )
    run_logged(
        command,
        log_file=path_value(runtime["log_file"], "runtime.log_file"),
        environment=environment,
    )
    write_json(run_manifest, manifest)
    staged_manifest.unlink(missing_ok=True)
    if staged_train_config is not None:
        staged_train_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
