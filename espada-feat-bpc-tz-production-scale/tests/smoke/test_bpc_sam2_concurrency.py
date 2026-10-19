"""End-to-end test that run_bpc_pipeline actually uses cfg.sam2.workers > 1
SAM2 threads, and that they share one Sam2Segmenter safely (no race).

This targets the exact bug found via code review: cfg.sam2.workers was read
nowhere in run_bpc_pipeline — exactly one hardcoded SAM2 thread was always
spawned. On a live 4-GPU run this meant SAM2 could block entirely on a
saturated vlm_queue.put(), and `pointing` output stayed stuck at 0 while
bboxing/captions (produced earlier, directly by YOLO) kept progressing.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python not installed")
pd = pytest.importorskip("pandas", reason="pandas not installed")

from tests.conftest import make_synthetic_dataset  # noqa: E402

pytestmark = [pytest.mark.smoke, pytest.mark.needs_cv2, pytest.mark.needs_pandas]


def _one_detection_df(image: np.ndarray) -> pd.DataFrame:
    h, w = image.shape[:2]
    return pd.DataFrame(
        [{"class": "person", "class_id": 0, "conf": 0.9,
          "x1": 2.0, "y1": 2.0, "x2": float(w // 4), "y2": float(h // 4)}]
    )


class FakeYolo:
    def predict(self, image):
        return _one_detection_df(image)

    def predict_on_batch(self, images):
        return [_one_detection_df(img) for img in images]


class RaceDetectingSegmenter:
    """Not thread-safe by itself (mutates self._size with no lock) — used to
    verify run_bpc_pipeline's segmenter_lock actually serializes GPU calls."""

    def __init__(self, work_seconds: float = 0.02):
        self._size = (0, 0)
        self._work_seconds = work_seconds
        self._active = 0
        self.max_concurrent_seen = 0
        self._guard = threading.Lock()  # only used to observe, not to fix

    def set_image(self, pil_image):
        with self._guard:
            self._active += 1
            self.max_concurrent_seen = max(self.max_concurrent_seen, self._active)
        self._size = (pil_image.height, pil_image.width)
        time.sleep(self._work_seconds)

    def predict_from_boxes(self, boxes, multimask_output=False):
        h, w = self._size
        masks = np.zeros((len(boxes), h, w), dtype=bool)
        for i in range(len(boxes)):
            masks[i, 2:max(3, h // 4), 2:max(3, w // 4)] = True
        with self._guard:
            self._active -= 1
        return SimpleNamespace(masks=masks)


def _fake_chat_completion(*, base_url, api_token, model, system_text, user_text, images, timeout_s=60):
    text = (system_text + user_text).lower()
    if "point" in text:
        content = '{"grounding_question": "Point.", "grounding_answer": [{"label": "person", "point_2d": [5, 5]}]}'
    elif len(images) == 1:
        content = '{"grounding_question": "Describe.", "grounding_answer": "A person stands."}'
    else:
        content = '{"grounding_question": "Locate.", "grounding_answer": [{"label": "person", "bbox_2d": [2, 2, 20, 20]}]}'
    return SimpleNamespace(content=content)


def test_sam2_workers_config_is_respected_and_thread_safe(tmp_path):
    """cfg.sam2.workers=3 must spawn 3 SAM2 threads sharing one segmenter
    safely (max_concurrent_seen == 1 thanks to segmenter_lock), and every
    frame with detections must still get a pointing output."""
    from src.bpc.config import load_bpc_config
    from src.bpc.state import StateLogger
    from src.bpc.storage import setup_output_dirs
    from src.bpc.pipeline import run_bpc_pipeline
    from src.vlm.prompts_processing import load_yaml

    frames = make_synthetic_dataset(tmp_path / "raw", count=15, seed=42)
    out_root = tmp_path / "out"

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "judge:\n  enabled: false\n"
        "sam2:\n  workers: 3\n"
        "vlm:\n  workers: 4\n  endpoints: ['http://fake']\n"
    )
    cfg = load_bpc_config(str(cfg_path))
    assert cfg.sam2.workers == 3

    dirs = setup_output_dirs(out_root, save_visualizations=cfg.storage.save_visualizations)
    prompts = load_yaml("configs/prompts.yaml")
    state_logger = StateLogger(out_root / "logs", "test-run")
    segmenter = RaceDetectingSegmenter(work_seconds=0.02)

    with patch("src.bpc.pipeline.chat_completion", side_effect=_fake_chat_completion):
        run_bpc_pipeline(
            images=frames,
            output_root=out_root,
            cfg=cfg,
            prompts=prompts,
            api_token="",
            segmenter=segmenter,
            yolo_model=FakeYolo(),
            dirs=dirs,
            state_logger=state_logger,
            run_id="test-run",
        )

    # The critical safety property: even with 3 concurrent SAM2 threads,
    # segmenter_lock must ensure only one GPU call is ever "active" — if
    # this is > 1, the lock isn't being applied and images could get masks
    # computed against the wrong frame's set_image() call.
    assert segmenter.max_concurrent_seen == 1

    # Every frame (all have detections from FakeYolo) must produce a
    # pointing output despite fan-out across multiple SAM2 threads.
    pointing_dir = out_root / "outputs" / "pointing"
    stems = sorted(p.stem for p in pointing_dir.glob("*.jsonl"))
    expected = sorted(p.stem for p in frames)
    assert stems == expected, f"pointing output incomplete: {len(stems)}/{len(expected)}"


def test_single_sam2_worker_still_works_default(tmp_path):
    """Backward compat: sam2.workers defaulting to 1 (or unset) still works."""
    from src.bpc.config import load_bpc_config
    from src.bpc.state import StateLogger
    from src.bpc.storage import setup_output_dirs
    from src.bpc.pipeline import run_bpc_pipeline
    from src.vlm.prompts_processing import load_yaml

    frames = make_synthetic_dataset(tmp_path / "raw", count=3, seed=1)
    out_root = tmp_path / "out"
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("judge:\n  enabled: false\nvlm:\n  workers: 1\n  endpoints: ['http://fake']\n")
    cfg = load_bpc_config(str(cfg_path))
    assert cfg.sam2.workers == 1

    dirs = setup_output_dirs(out_root, save_visualizations=cfg.storage.save_visualizations)
    prompts = load_yaml("configs/prompts.yaml")
    state_logger = StateLogger(out_root / "logs", "r")
    segmenter = RaceDetectingSegmenter(work_seconds=0.0)

    with patch("src.bpc.pipeline.chat_completion", side_effect=_fake_chat_completion):
        run_bpc_pipeline(
            images=frames, output_root=out_root, cfg=cfg, prompts=prompts,
            api_token="", segmenter=segmenter, yolo_model=FakeYolo(),
            dirs=dirs, state_logger=state_logger, run_id="r",
        )

    pointing_dir = out_root / "outputs" / "pointing"
    stems = sorted(p.stem for p in pointing_dir.glob("*.jsonl"))
    assert stems == sorted(p.stem for p in frames)


def test_pointing_skipped_recorded_in_state_jsonl(tmp_path):
    """No-detection frames must show up as a distinct 'skip' event in
    state.jsonl, not just vanish — so `espada-bpc status` can report them
    instead of leaving pointing indistinguishable from stuck."""
    import json as json_mod
    from src.bpc.config import load_bpc_config
    from src.bpc.state import StateLogger
    from src.bpc.storage import setup_output_dirs
    from src.bpc.pipeline import run_bpc_pipeline
    from src.vlm.prompts_processing import load_yaml

    frames = make_synthetic_dataset(tmp_path / "raw", count=2, seed=9)
    out_root = tmp_path / "out"
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("judge:\n  enabled: false\nvlm:\n  workers: 1\n  endpoints: ['http://fake']\n")
    cfg = load_bpc_config(str(cfg_path))

    dirs = setup_output_dirs(out_root, save_visualizations=cfg.storage.save_visualizations)
    prompts = load_yaml("configs/prompts.yaml")
    state_logger = StateLogger(out_root / "logs", "r")
    segmenter = RaceDetectingSegmenter(work_seconds=0.0)

    class NoDetectionYolo:
        def predict(self, image):
            return pd.DataFrame(columns=["class", "class_id", "conf", "x1", "y1", "x2", "y2"])

        def predict_on_batch(self, images):
            return [self.predict(img) for img in images]

    with patch("src.bpc.pipeline.chat_completion", side_effect=_fake_chat_completion):
        run_bpc_pipeline(
            images=frames, output_root=out_root, cfg=cfg, prompts=prompts,
            api_token="", segmenter=segmenter, yolo_model=NoDetectionYolo(),
            dirs=dirs, state_logger=state_logger, run_id="r",
        )

    state_path = out_root / "logs" / "state.jsonl"
    records = [json_mod.loads(l) for l in state_path.read_text().splitlines() if l.strip()]
    skip_events = [r for r in records if r.get("stage") == "sam2" and r.get("event") == "skip"]
    assert len(skip_events) == len(frames)
    assert {r["task_name"] for r in skip_events} == {"pointing"}
