"""Unit tests for config-driven model initialization (Stage 2a).

Verifies that ``runner.init_models`` reads the ``models`` config section and
forwards device / dtype / batch_size / empty_cache to the wrappers, and that the
shipped ``configs/config.yaml`` exposes those keys.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.vlm.prompts_processing import load_yaml

pytestmark = pytest.mark.unit

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def test_shipped_config_has_models_section():
    cfg = load_yaml(CONFIG_PATH)
    assert "models" in cfg
    models = cfg["models"]
    for key in ("device", "sam2_dtype", "yolo_batch_size", "vlm_workers", "empty_cache"):
        assert key in models, f"missing models.{key}"


def test_init_models_forwards_config():
    import runner

    captured = {}

    def fake_segmenter(**kwargs):
        captured["sam2"] = kwargs
        return object()

    def fake_yolo(**kwargs):
        captured["yolo"] = kwargs
        return object()

    with patch.object(runner, "Sam2Segmenter", side_effect=fake_segmenter), patch.object(
        runner, "YoloV8", side_effect=fake_yolo
    ):
        runner.init_models(
            {
                "device": "cuda:1",
                "sam2_dtype": "bfloat16",
                "yolo_model": "yolov8x.pt",
                "yolo_batch_size": 8,
                "empty_cache": True,
            }
        )

    assert captured["sam2"] == {
        "device": "cuda:1",
        "dtype": "bfloat16",
        "empty_cache": True,
    }
    assert captured["yolo"] == {
        "model_path": "yolov8x.pt",
        "device": "cuda:1",
        "batch_size": 8,
        "empty_cache": True,
    }


def test_init_models_defaults_when_empty():
    import runner

    captured = {}
    with patch.object(
        runner, "Sam2Segmenter", side_effect=lambda **k: captured.setdefault("sam2", k)
    ), patch.object(runner, "YoloV8", side_effect=lambda **k: captured.setdefault("yolo", k)):
        runner.init_models({})

    assert captured["sam2"]["device"] == "cuda"
    assert captured["sam2"]["dtype"] is None
    assert captured["yolo"]["batch_size"] == 1
    assert captured["yolo"]["empty_cache"] is False
