"""Tests for src/bpc/preflight.py — image preflight validation."""
import pytest
import numpy as np

pytest.importorskip("cv2")

from pathlib import Path
from src.bpc.preflight import preflight_image, run_preflight, PreflightResult
from src.bpc.config import InputConfig


def _default_cfg(**kwargs) -> InputConfig:
    defaults = dict(
        require_prepared_images=True,
        allowed_extensions=[".jpg", ".jpeg"],
        require_rgb=True,
        max_side=768,
    )
    defaults.update(kwargs)
    return InputConfig(**defaults)


def _write_jpeg(path: Path, w: int = 100, h: int = 80) -> Path:
    """Creates a minimal valid JPEG using PIL."""
    from PIL import Image
    img = Image.new("RGB", (w, h), color=(128, 64, 32))
    img.save(path, format="JPEG")
    return path


class TestPreflightImage:
    def test_valid_image_passes(self, tmp_path):
        img = _write_jpeg(tmp_path / "img.jpg")
        result = preflight_image(img, _default_cfg())
        assert result.valid is True
        assert result.reason is None
        assert result.image_id == "img"

    def test_wrong_extension_rejected(self, tmp_path):
        p = tmp_path / "img.png"
        _write_jpeg(p)
        result = preflight_image(p, _default_cfg())
        assert result.valid is False
        assert "Extension" in result.reason

    def test_png_allowed_when_configured(self, tmp_path):
        p = tmp_path / "img.png"
        _write_jpeg(p)
        result = preflight_image(p, _default_cfg(allowed_extensions=[".jpg", ".jpeg", ".png"]))
        assert result.valid is True

    def test_unreadable_file_rejected(self, tmp_path):
        p = tmp_path / "bad.jpg"
        p.write_bytes(b"not an image")
        result = preflight_image(p, _default_cfg())
        assert result.valid is False
        assert "read" in result.reason.lower()

    def test_image_id_is_stem(self, tmp_path):
        p = _write_jpeg(tmp_path / "frame_0042.jpg")
        result = preflight_image(p, _default_cfg())
        assert result.image_id == "frame_0042"


class TestRunPreflight:
    def test_all_valid(self, tmp_path):
        imgs = [_write_jpeg(tmp_path / f"img{i}.jpg") for i in range(3)]
        valid, results = run_preflight(imgs, _default_cfg())
        assert len(valid) == 3
        assert all(r.valid for r in results)

    def test_mixed_valid_invalid(self, tmp_path):
        good = _write_jpeg(tmp_path / "good.jpg")
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"corrupted")
        valid, results = run_preflight([good, bad], _default_cfg())
        assert len(valid) == 1
        assert valid[0] == good
        assert len(results) == 2

    def test_writes_preflight_log(self, tmp_path):
        imgs = [_write_jpeg(tmp_path / f"img{i}.jpg") for i in range(2)]
        log = tmp_path / "preflight.log"
        run_preflight(imgs, _default_cfg(), preflight_log=log)
        assert log.exists()
        import json
        lines = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        assert len(lines) == 2
        assert all(l["valid"] for l in lines)

    def test_empty_list(self, tmp_path):
        valid, results = run_preflight([], _default_cfg())
        assert valid == []
        assert results == []
