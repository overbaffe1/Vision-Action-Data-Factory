"""Shared pytest fixtures for the espada test suite.

The repository's heavy dependencies (cv2, pandas, torch, ultralytics,
transformers) are only installed in the full ``vvinokurov-markup-env`` venv.
Tests that need them use ``pytest.importorskip`` so the pure-logic suite stays
green in a minimal environment.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _ensure_stub(name: str, attrs: dict | None = None) -> None:
    """Register a lightweight stand-in module if the real one is unavailable.

    Heavy deps (torch, transformers, ultralytics) live only in the full
    ``vvinokurov-markup-env`` venv. The unit/smoke tests never exercise real
    model math — they patch or fake the models — so a stub that merely lets the
    imports succeed is enough. If the real package is installed it wins and the
    stub is never used.
    """
    if name in sys.modules:
        return
    try:
        __import__(name)
        return
    except ImportError:
        pass
    module = types.ModuleType(name)
    for key, value in (attrs or {}).items():
        setattr(module, key, value)
    sys.modules[name] = module


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False

    @staticmethod
    def empty_cache() -> None:
        return None


# Installed at collection time so any test module can import the wrappers.
_ensure_stub(
    "torch",
    {"cuda": _FakeCuda(), "device": (lambda x: x), "no_grad": None, "float16": "float16"},
)
_ensure_stub("transformers", {"Sam2Model": object, "Sam2Processor": object})
_ensure_stub("ultralytics", {"YOLO": object})


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random generator for reproducible synthetic data."""
    return np.random.default_rng(1234)


@pytest.fixture
def synthetic_rgb(rng: np.random.Generator) -> np.ndarray:
    """A small synthetic RGB uint8 image (H=64, W=96, 3)."""
    return rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)


@pytest.fixture
def raw_images_dir(tmp_path: Path, rng: np.random.Generator) -> Path:
    """Directory with a few valid JPEG/PNG images written via Pillow.

    Avoids cv2 so it works in the minimal environment.
    """
    src = tmp_path / "raw_images"
    src.mkdir()
    for i in range(3):
        arr = rng.integers(0, 256, size=(40, 60, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        ext = ".jpg" if i % 2 == 0 else ".png"
        img.save(src / f"frame_{i:03d}{ext}")
    # An unsupported file that must be ignored by collectors.
    (src / "notes.txt").write_text("ignore me", encoding="utf-8")
    return src


@pytest.fixture
def single_image_file(tmp_path: Path, rng: np.random.Generator) -> Path:
    """A single valid JPEG on disk."""
    arr = rng.integers(0, 256, size=(50, 70, 3), dtype=np.uint8)
    path = tmp_path / "one.jpg"
    Image.fromarray(arr, mode="RGB").save(path)
    return path


def make_synthetic_dataset(
    directory: Path,
    count: int,
    *,
    size: tuple[int, int] = (48, 64),
    seed: int = 0,
    ext: str = ".jpg",
) -> list[Path]:
    """Write ``count`` deterministic synthetic RGB images into ``directory``.

    Returns the sorted list of created paths. Shared by fixtures and the
    standalone data-generation script so tests and manual runs use identical data.
    """
    directory.mkdir(parents=True, exist_ok=True)
    gen = np.random.default_rng(seed)
    h, w = size
    paths: list[Path] = []
    width = max(4, len(str(count)))
    for i in range(count):
        arr = gen.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        p = directory / f"frame_{i:0{width}d}{ext}"
        Image.fromarray(arr, mode="RGB").save(p)
        paths.append(p)
    return sorted(paths)


@pytest.fixture
def dataset_factory(tmp_path: Path):
    """Factory fixture: ``dataset_factory(n)`` -> list of n synthetic image paths."""

    def _make(count: int, *, subdir: str = "ds", **kwargs) -> list[Path]:
        return make_synthetic_dataset(tmp_path / subdir, count, **kwargs)

    return _make


@pytest.fixture
def bbox_detections() -> dict[str, list]:
    """A realistic ``build_extended_detections``-shaped dict."""
    return {
        "id": [1, 2],
        "bbox": [[10, 10, 30, 40], [5, 5, 58, 60]],
        "class": ["person", "chair"],
        "conf": [0.91, 0.55],
    }
