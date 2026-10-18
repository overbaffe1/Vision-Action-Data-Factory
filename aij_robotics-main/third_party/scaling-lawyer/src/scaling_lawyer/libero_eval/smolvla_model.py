"""SmolVLA policy wrapper for LIBERO observation layout (LeRobot v3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

LIBERO_IMAGE_KEYS = ["observation.images.image", "observation.images.image2"]


def _image_keys(policy_cfg: PreTrainedConfig) -> list[str]:
    keys = [
        key
        for key, feature in policy_cfg.input_features.items()
        if feature.type is FeatureType.VISUAL
    ]
    return sorted(keys)


def _state_dim(policy_cfg: PreTrainedConfig) -> int:
    state = policy_cfg.input_features.get("observation.state")
    if state is None:
        raise KeyError("Checkpoint has no observation.state feature.")
    return int(state.shape[0])


def libero_obs_to_batch(
    obs: dict[str, Any],
    task: str,
    *,
    image_keys: list[str],
    state_dim: int,
    agentview_key: str = "agentview_image",
    wrist_key: str = "robot0_eye_in_hand_image",
) -> dict[str, Any]:
    """Map LIBERO-style observations to LeRobot SmolVLA batch keys."""
    if sorted(image_keys) != LIBERO_IMAGE_KEYS:
        raise ValueError(f"Unsupported image feature set for LIBERO eval: {image_keys}")

    # Accept either LIBERO env keys or a flat rgb_obs dict used in validate_submission.
    if "rgb_obs" in obs:
        rgb = obs["rgb_obs"]
        agent = rgb.get(agentview_key, rgb.get("image", rgb.get("rgb_static")))
        wrist = rgb.get(wrist_key, rgb.get("image2", rgb.get("rgb_gripper")))
        if agent is None or wrist is None:
            raise KeyError(
                "LIBERO obs rgb_obs must include agentview/image and wrist/image2 "
                f"(or rgb_static/rgb_gripper). Got keys={sorted(rgb)}"
            )
        agent = np.asarray(agent, dtype=np.uint8)
        wrist = np.asarray(wrist, dtype=np.uint8)
        robot_obs = np.asarray(obs["robot_obs"], dtype=np.float32)
    else:
        agent = np.asarray(obs.get(agentview_key, obs.get("observation.images.image")), dtype=np.uint8)
        wrist = np.asarray(
            obs.get(wrist_key, obs.get("observation.images.image2")), dtype=np.uint8
        )
        if agent.ndim != 3 or wrist.ndim != 3:
            raise KeyError("LIBERO obs missing agentview and wrist images")
        robot_obs = np.asarray(
            obs.get("robot_obs", obs.get("observation.state")), dtype=np.float32
        )

    if robot_obs.shape[0] < state_dim:
        state = np.zeros(state_dim, dtype=np.float32)
        state[: robot_obs.shape[0]] = robot_obs
    else:
        state = robot_obs[:state_dim].copy()

    return {
        "observation.images.image": agent,
        "observation.images.image2": wrist,
        "observation.state": state,
        "task": task if task.endswith("\n") else f"{task}\n",
    }


def _ensure_policy_config_type(checkpoint_dir: Path) -> None:
    """LeRobot sometimes omits `type` in config.json; recover from train_config.json."""
    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("type"):
        return

    policy_type = "smolvla"
    train_config_path = checkpoint_dir / "train_config.json"
    if train_config_path.is_file():
        train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
        nested = train_config.get("policy") if isinstance(train_config, dict) else None
        if isinstance(nested, dict) and nested.get("type"):
            policy_type = str(nested["type"])

    config["type"] = policy_type
    config_path.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")


class SmolVLALiberoModel:
    """LIBERO-compatible policy using a LeRobot SmolVLA checkpoint."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        device: str | torch.device = "cuda:0",
    ) -> None:
        checkpoint_dir = Path(checkpoint_dir)
        _ensure_policy_config_type(checkpoint_dir)
        policy_cfg = PreTrainedConfig.from_pretrained(checkpoint_dir)
        policy_cls = get_policy_class(policy_cfg.type)
        self.device = torch.device(device)
        self.policy = policy_cls.from_pretrained(checkpoint_dir, config=policy_cfg)
        self.policy.to(self.device)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg, str(checkpoint_dir)
        )
        self.image_keys = _image_keys(policy_cfg)
        self.state_dim = _state_dim(policy_cfg)
        self._task: str | None = None

    def reset(self) -> None:
        self.policy.reset()
        self._task = None

    def step(self, obs: dict[str, Any], goal: str) -> np.ndarray:
        if goal != self._task:
            self._task = goal
            self.policy.reset()

        batch = libero_obs_to_batch(
            obs,
            goal,
            image_keys=self.image_keys,
            state_dim=self.state_dim,
        )
        task = batch.pop("task")
        batch = prepare_observation_for_inference(batch, self.device, task=task)
        batch = self.preprocessor(batch)
        with torch.inference_mode():
            action = self.policy.select_action(batch)
        action = self.postprocessor(action)
        # Keep raw gripper channel (LIBERO continuous / binary-ish, not remapped).
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32)
