#!/usr/bin/env python3
"""Sanity-check hackathon submission weights (VLM + action expert).

Validates artifact layout, loads checkpoints, and runs a tiny forward pass.
Action expert must use the LIBERO layout (image/image2).
Exits 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _submission_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "smolvlm2").is_dir() and (path / "action_expert").is_dir():
        return path
    candidate = path / "submission"
    if candidate.is_dir():
        return candidate.resolve()
    raise FileNotFoundError(f"Expected submission/ with smolvlm2/ and action_expert/: {path}")


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"{label} is empty: {path}")
    return path


def _require_weights(dir_path: Path, label: str) -> None:
    has_safetensors = any(dir_path.glob("*.safetensors"))
    has_bin = any(dir_path.glob("pytorch_model*.bin"))
    if not has_safetensors and not has_bin:
        raise FileNotFoundError(f"{label}: no model weights under {dir_path}")


def check_layout(submission: Path) -> None:
    vlm_dir = submission / "smolvlm2"
    action_dir = submission / "action_expert"
    for name, directory in (("smolvlm2", vlm_dir), ("action_expert", action_dir)):
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing {name}/ under {submission}")
        _require_file(directory / "config.json", f"{name} config.json")
        _require_weights(directory, name)

    meta_path = submission / "metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("base_vlm") != "HuggingFaceTB/SmolVLM2-500M-Video-Instruct":
            print(f"[warn] unexpected base_vlm in metadata: {meta.get('base_vlm')}")
        suite = meta.get("vla_suite")
        if suite and suite != "libero":
            print(f"[warn] unexpected vla_suite in metadata: {suite!r} (expected 'libero')")
    print("[ok] layout")


def _visual_feature_keys(action_dir: Path) -> list[str]:
    config = json.loads((action_dir / "config.json").read_text(encoding="utf-8"))
    features = config.get("input_features") or {}
    keys: list[str] = []
    for name, feat in features.items():
        if not isinstance(feat, dict):
            continue
        ftype = str(feat.get("type", "")).upper()
        if "VISUAL" in ftype or name.startswith("observation.images."):
            keys.append(name)
    return sorted(keys)


def assert_libero_action_suite(action_dir: Path) -> None:
    """Require LIBERO image/image2 layout on the action-expert checkpoint."""
    keys = _visual_feature_keys(action_dir)
    expected = ["observation.images.image", "observation.images.image2"]
    if keys != expected:
        raise ValueError(
            "Action expert must use LIBERO layout "
            f"{expected}, got input_features {keys}."
        )


def check_vlm(vlm_dir: Path, *, device: str, dtype: str) -> None:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    torch_dtype = getattr(torch, dtype)
    processor = AutoProcessor.from_pretrained(vlm_dir, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        vlm_dir,
        local_files_only=True,
        dtype=torch_dtype,
        device_map=device if device != "cpu" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    image = Image.new("RGB", (224, 224), color=(32, 64, 128))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in one word."},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], text_kwargs={"return_tensors": "pt"})
    model_device = next(model.parameters()).device
    for key, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            continue
        if value.is_floating_point():
            inputs[key] = value.to(model_device, dtype=torch_dtype)
        else:
            inputs[key] = value.to(model_device)

    with torch.inference_mode():
        logits = model(**inputs).logits
        if not torch.isfinite(logits).all():
            raise RuntimeError("VLM forward produced non-finite logits")
        output_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)

    answer = processor.decode(output_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    print(f"[ok] VLM forward + generate (sample: {answer.strip()!r})")


def check_action_expert(action_dir: Path, *, device: str) -> None:
    """Load LIBERO action expert and run a dummy forward."""
    sl_src = REPO_ROOT / "third_party" / "scaling-lawyer" / "src"
    if str(sl_src) not in sys.path:
        sys.path.insert(0, str(sl_src))

    assert_libero_action_suite(action_dir)
    print("[info] action-expert suite=libero")

    from scaling_lawyer.libero_eval import SmolVLALiberoModel

    model: Any = SmolVLALiberoModel(action_dir, device=device)
    obs = {
        "rgb_obs": {
            "image": np.zeros((256, 256, 3), dtype=np.uint8),
            "image2": np.zeros((256, 256, 3), dtype=np.uint8),
        },
        "robot_obs": np.zeros(model.state_dim, dtype=np.float32),
    }

    action = model.step(obs, goal="push the button")
    if action.shape[-1] != 7:
        raise RuntimeError(f"Expected 7-dim action, got shape {action.shape}")
    if not np.isfinite(action).all():
        raise RuntimeError(f"Action expert produced non-finite action: {action}")
    print(f"[ok] libero_eval SmolVLALiberoModel forward (action: {np.round(action, 4).tolist()})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=Path,
        help="Path to runs/<run>/submission or runs/<run>/",
    )
    parser.add_argument("--device", default="cuda:0", help="cuda:0 or cpu")
    parser.add_argument("--dtype", default="bfloat16", help="VLM torch dtype (bfloat16/float16/float32)")
    parser.add_argument("--skip-vlm", action="store_true")
    parser.add_argument("--skip-action", action="store_true")
    args = parser.parse_args()

    submission = _submission_dir(args.path)
    print(f"[validate] {submission}")

    check_layout(submission)
    if not args.skip_vlm:
        check_vlm(submission / "smolvlm2", device=args.device, dtype=args.dtype)
    if not args.skip_action:
        check_action_expert(submission / "action_expert", device=args.device)

    print("[done] submission weights look loadable")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 — CLI entry
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
