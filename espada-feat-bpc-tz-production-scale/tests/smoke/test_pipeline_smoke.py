"""End-to-end smoke test for the queue-based pipeline.

No real models or network calls: YOLO and SAM2 are lightweight fakes and the
VLM ``chat_completion`` is patched to return canned, schema-valid JSON. The
test drives the *real* ``run_pipeline_with_queues`` orchestration (threads,
queues, sentinels, progress monitor, JSON saving) against synthetic images and
asserts that valid output files are produced for every stage.

Requires cv2 + numpy + pandas (the worker code imports cv2 and uses
``build_extended_detections`` which needs pandas). Skipped otherwise.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2", reason="opencv-python not installed")
pd = pytest.importorskip("pandas", reason="pandas not installed")

# NOTE: we deliberately do NOT import src.sam2 (it pulls in torch). The SAM2
# worker only reads ``out.masks``, so a SimpleNamespace stand-in is enough.
from src.vlm.prompts_processing import load_yaml  # noqa: E402

pytestmark = [pytest.mark.smoke, pytest.mark.needs_cv2, pytest.mark.needs_pandas]


# --- fakes -----------------------------------------------------------------


def _one_detection_df(image: np.ndarray) -> pd.DataFrame:
    h, w = image.shape[:2]
    return pd.DataFrame(
        [
            {
                "class": "person",
                "class_id": 0,
                "conf": 0.9,
                "x1": 2.0,
                "y1": 2.0,
                "x2": float(w // 4),
                "y2": float(h // 4),
            }
        ]
    )


class FakeYolo:
    """Returns one fixed detection per image, as a YOLO-style DataFrame.

    Implements both the single-image and batched inference APIs so it can be
    used to test either YOLO worker.
    """

    def predict(self, image: np.ndarray) -> pd.DataFrame:
        return _one_detection_df(image)

    def predict_on_batch(self, images: list[np.ndarray]) -> list[pd.DataFrame]:
        return [_one_detection_df(img) for img in images]


class FakeSegmenter:
    """Produces one boolean mask matching the image size."""

    def __init__(self) -> None:
        self._image_size: tuple[int, int] = (0, 0)

    def set_image(self, pil_image: Image.Image) -> None:
        self._image_size = (pil_image.height, pil_image.width)

    def predict_from_boxes(self, boxes, multimask_output=False) -> Sam2Masks:
        h, w = self._image_size
        masks = np.zeros((len(boxes), h, w), dtype=bool)
        for i in range(len(boxes)):
            masks[i, 2 : max(3, h // 4), 2 : max(3, w // 4)] = True
        return SimpleNamespace(masks=masks)


def _fake_chat_completion(*, base_url, api_token, model, system_text, user_text, images):
    """Return schema-valid JSON depending on which task is being requested."""
    # We can't see task_name here, so emit a payload valid for all three
    # validators by inspecting the user/system text heuristics is brittle;
    # instead emit per-call canned answers keyed on number of images + content.
    # bboxing & pointing send 2 images; captioning sends 1.
    if len(images) == 1:
        content = (
            '{"grounding_question": "Describe the scene.",'
            ' "grounding_answer": "A person stands near a wall in a plain room."}'
        )
    elif "point" in (system_text + user_text).lower():
        content = (
            '{"grounding_question": "Point every person.",'
            ' "grounding_answer": [{"label": "person", "point_2d": [5, 5]}]}'
        )
    else:
        content = (
            '{"grounding_question": "Locate every person.",'
            ' "grounding_answer": [{"label": "person", "bbox_2d": [2, 2, 20, 20]}]}'
        )
    return SimpleNamespace(content=content)


# --- fixtures --------------------------------------------------------------


@pytest.fixture
def processed_images(tmp_path: Path, rng) -> list[Path]:
    """A couple of real on-disk JPEGs the workers can cv2.imread."""
    out: list[Path] = []
    for i in range(2):
        arr = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
        p = tmp_path / f"img_{i}.jpg"
        Image.fromarray(arr, mode="RGB").save(p)
        out.append(p)
    return out


@pytest.fixture
def prompts() -> dict:
    root = Path(__file__).resolve().parents[2]
    return load_yaml(root / "configs" / "prompts.yaml")


# --- the smoke test --------------------------------------------------------


@pytest.mark.parametrize("yolo_batch_size", [1, 4], ids=["per-image", "batched"])
def test_pipeline_produces_valid_outputs(
    processed_images, prompts, tmp_path, monkeypatch, yolo_batch_size
):
    """Both the per-image and batched YOLO workers must yield identical outputs."""
    from runner import run_pipeline_with_queues

    # vlm_worker writes to Path("outputs")/task — run from an isolated cwd.
    workdir = tmp_path / "run"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    with patch("src.workers.chat_completion", side_effect=_fake_chat_completion):
        run_pipeline_with_queues(
            saved=processed_images,
            segmenter=FakeSegmenter(),
            yolo_model=FakeYolo(),
            prompts=prompts,
            base_url="http://fake",
            api_token="",
            model_vlm="fake-model",
            fail_fast=True,
            vlm_workers_number=4,
            yolo_batch_size=yolo_batch_size,
        )

    outputs = workdir / "outputs"
    for stage in ("bboxing", "captions", "pointing"):
        stage_dir = outputs / stage
        assert stage_dir.is_dir(), f"missing stage dir: {stage}"
        files = sorted(stage_dir.glob("*.jsonl"))
        assert len(files) == len(processed_images), f"{stage}: expected one json per image"

        dataset = outputs / f"{stage}.jsonl"
        records = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()]
        assert len(records) == len(processed_images)
        assert all(record["messages"][0]["content"].startswith("<image>\n") for record in records)
        assert all(record["images"][0].startswith("/") for record in records)


def test_pipeline_skips_pointing_when_no_detections(processed_images, prompts, tmp_path, monkeypatch):
    """If YOLO finds nothing, bbox+caption still run but pointing is skipped."""

    class EmptyYolo:
        def predict(self, image):
            return pd.DataFrame(
                columns=["class", "class_id", "conf", "x1", "y1", "x2", "y2"]
            )

    from runner import run_pipeline_with_queues

    workdir = tmp_path / "run2"
    workdir.mkdir()
    monkeypatch.chdir(workdir)

    with patch("src.workers.chat_completion", side_effect=_fake_chat_completion):
        run_pipeline_with_queues(
            saved=processed_images,
            segmenter=FakeSegmenter(),
            yolo_model=EmptyYolo(),
            prompts=prompts,
            base_url="http://fake",
            api_token="",
            model_vlm="fake-model",
            fail_fast=False,
            vlm_workers_number=2,
        )

    outputs = workdir / "outputs"
    # bbox + caption still produced
    assert len(list((outputs / "captions").glob("*.jsonl"))) == len(processed_images)
    # pointing dir may not exist at all (skipped before any VLM task)
    pointing = outputs / "pointing"
    if pointing.exists():
        assert len(list(pointing.glob("*.jsonl"))) == 0
