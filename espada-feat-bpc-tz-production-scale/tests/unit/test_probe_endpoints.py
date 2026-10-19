"""Tests for scripts/probe_endpoints.py helper logic."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "probe_endpoints.py"
_SPEC = importlib.util.spec_from_file_location("probe_endpoints", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
probe_endpoints = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe_endpoints
_SPEC.loader.exec_module(probe_endpoints)


def test_resolve_api_token_allows_empty_when_enabled(monkeypatch):
    monkeypatch.delenv("API_TOKEN", raising=False)
    assert probe_endpoints._resolve_api_token(allow_no_auth=True) == ""


def test_resolve_api_token_requires_env_by_default(monkeypatch):
    monkeypatch.delenv("API_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="API_TOKEN"):
        probe_endpoints._resolve_api_token(allow_no_auth=False)


def test_build_probe_payload_light_mode_uses_one_image():
    payload = probe_endpoints._build_probe_payload(mode="light")
    assert payload.model_task == "light"
    assert len(payload.images) == 1
    assert "STRICT JSON" in payload.system_text


def test_build_probe_payload_bboxing_uses_two_images_and_real_prompt():
    payload = probe_endpoints._build_probe_payload(mode="bboxing")
    assert payload.model_task == "bboxing"
    assert len(payload.images) == 2
    assert "Locate" in payload.system_text
    assert "BBox dictionary" in payload.user_text


def test_build_probe_payload_pointing_uses_two_images_and_real_prompt():
    payload = probe_endpoints._build_probe_payload(mode="pointing")
    assert payload.model_task == "pointing"
    assert len(payload.images) == 2
    assert "Point" in payload.system_text
    assert "Points dictionary" in payload.user_text


def test_build_probe_payload_captions_uses_one_image_and_real_prompt():
    payload = probe_endpoints._build_probe_payload(mode="captions")
    assert payload.model_task == "captions"
    assert len(payload.images) == 1
    assert "robot action planner" in payload.user_text
