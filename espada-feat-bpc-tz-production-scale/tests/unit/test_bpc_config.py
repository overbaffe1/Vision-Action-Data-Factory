"""Tests for src/bpc/config.py — BpcConfig loading and merging."""
import pytest
from pathlib import Path

from src.bpc.config import load_bpc_config, BpcConfig, BboxValidationConfig


def test_load_bpc_config_defaults(tmp_path):
    cfg_file = tmp_path / "bpc.yaml"
    cfg_file.write_text("input:\n  max_side: 512\n")
    cfg = load_bpc_config(cfg_file)
    assert cfg.input.max_side == 512
    assert cfg.input.require_prepared_images is True


def test_load_bpc_config_full_file():
    """Smoke-test loading the real bpc.yaml."""
    cfg = load_bpc_config("configs/bpc.yaml")
    assert cfg.yolo.batch_size > 0
    assert cfg.vlm.workers > 0
    assert cfg.validation.bbox.out_of_bounds_policy in ("reject", "clip")
    assert isinstance(cfg.validation.captions.forbidden_phrases, list)


def test_bpc_config_defaults():
    cfg = BpcConfig()
    assert cfg.retry.max_attempts == 3
    assert cfg.storage.save_raw_vlm_responses is False
    assert cfg.storage.save_visualizations is True
    assert cfg.judge.enabled is False
    assert "bboxing" in cfg.judge.tasks


def test_bbox_validation_config_defaults():
    v = BboxValidationConfig()
    assert v.min_box_area_px == 4
    assert v.max_boxes_per_image == 100
    assert v.out_of_bounds_policy == "reject"


def test_load_config_judge_section(tmp_path):
    cfg_file = tmp_path / "bpc.yaml"
    cfg_file.write_text(
        "judge:\n  enabled: true\n  tasks:\n    - bboxing\n  workers: 2\n"
    )
    cfg = load_bpc_config(cfg_file)
    assert cfg.judge.enabled is True
    assert cfg.judge.workers == 2
    assert cfg.judge.tasks == ["bboxing"]


def test_load_config_explicit_vlm_targets(tmp_path):
    cfg_file = tmp_path / "bpc.yaml"
    cfg_file.write_text(
        "vlm:\n"
        "  targets:\n"
        "    - base_url: https://gateway.example.com\n"
        "      model: qwen-h100\n"
        "      max_concurrency: 12\n"
        "    - base_url: https://gateway.example.com\n"
        "      model: qwen-a100\n"
        "      max_concurrency: 6\n"
    )
    cfg = load_bpc_config(cfg_file)
    assert cfg.vlm.targets == [
        {"base_url": "https://gateway.example.com", "model": "qwen-h100", "max_concurrency": 12},
        {"base_url": "https://gateway.example.com", "model": "qwen-a100", "max_concurrency": 6},
    ]


def test_load_config_retry(tmp_path):
    cfg_file = tmp_path / "bpc.yaml"
    cfg_file.write_text("retry:\n  max_attempts: 5\n  backoff_seconds: 10\n")
    cfg = load_bpc_config(cfg_file)
    assert cfg.retry.max_attempts == 5
    assert cfg.retry.backoff_seconds == 10


def test_load_config_validation_section(tmp_path):
    cfg_file = tmp_path / "bpc.yaml"
    cfg_file.write_text(
        "validation:\n"
        "  bbox:\n"
        "    out_of_bounds_policy: clip\n"
        "    max_boxes_per_image: 50\n"
        "  captions:\n"
        "    min_length: 10\n"
        "    forbidden_phrases:\n"
        "      - 'as an AI'\n"
    )
    cfg = load_bpc_config(cfg_file)
    assert cfg.validation.bbox.out_of_bounds_policy == "clip"
    assert cfg.validation.bbox.max_boxes_per_image == 50
    assert cfg.validation.captions.min_length == 10
    assert "as an AI" in cfg.validation.captions.forbidden_phrases
