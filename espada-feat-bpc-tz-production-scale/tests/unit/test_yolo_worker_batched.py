"""Unit tests for the batched YOLO worker (Stage 2a).

Focus on the queue/batching mechanics: that frames are grouped into a single
``predict_on_batch`` call, that ``task_done`` is balanced so ``queue.join()``
returns, and that the SENTINEL terminates the worker.
"""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("cv2", reason="opencv-python not installed")
pytest.importorskip("pandas", reason="pandas not installed")
import pandas as pd  # noqa: E402

from src import workers  # noqa: E402
from src.structs import FrameTask  # noqa: E402
from src.vlm.prompts_processing import load_yaml  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.needs_cv2, pytest.mark.needs_pandas]

_PROMPTS = load_yaml(Path(__file__).resolve().parents[2] / "configs" / "prompts.yaml")


class _RecordingYolo:
    """Records the sizes of each predict_on_batch call."""

    def __init__(self):
        self.batch_sizes: list[int] = []

    def predict_on_batch(self, images):
        self.batch_sizes.append(len(images))
        return [pd.DataFrame(columns=["class", "conf", "x1", "y1", "x2", "y2"]) for _ in images]


def _run_batched(tasks, batch_size, yolo):
    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    vlm_q: queue.Queue = queue.Queue()
    prog_q: queue.Queue = queue.Queue()
    errs: queue.Queue = queue.Queue()
    stop = threading.Event()

    for t in tasks:
        in_q.put(t)
    in_q.put(workers.SENTINEL)

    t = threading.Thread(
        target=workers.yolo_worker_batched,
        kwargs={
            "input_queue": in_q,
            "output_queue": out_q,
            "vlm_queue": vlm_q,
            "yolo_model": yolo,
            "prompts": _PROMPTS,
            "fail_fast": False,
            "stop_event": stop,
            "errors": errs,
            "progress_queue": prog_q,
            "batch_size": batch_size,
        },
        daemon=True,
    )
    t.start()
    in_q.join()  # must return -> task_done accounting is balanced
    t.join(timeout=5)
    assert not t.is_alive(), "worker did not exit on SENTINEL"
    return out_q, vlm_q


def test_batched_groups_frames(tmp_path, monkeypatch):
    # Bypass real image reading: return a tiny synthetic frame for any path.
    fake_img = np.zeros((20, 20, 3), dtype=np.uint8)
    monkeypatch.setattr(
        workers, "_read_frame_images", lambda p: (fake_img, object())
    )

    tasks = [FrameTask(image_id=f"f{i}", frame_path=Path(f"/x/f{i}.jpg")) for i in range(5)]
    yolo = _RecordingYolo()

    out_q, vlm_q = _run_batched(tasks, batch_size=2, yolo=yolo)

    # 5 frames at batch_size 2 -> groups of [2, 2, 1] (exact grouping may vary
    # with queue timing, but total processed must be 5 and no group exceeds 2).
    assert sum(yolo.batch_sizes) == 5
    assert all(b <= 2 for b in yolo.batch_sizes)
    # Empty detections -> still emits one YoloStageResult per frame downstream.
    assert out_q.qsize() == 5


def test_batched_invalid_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        workers.yolo_worker_batched(
            input_queue=queue.Queue(),
            output_queue=queue.Queue(),
            vlm_queue=queue.Queue(),
            yolo_model=_RecordingYolo(),
            prompts={},
            fail_fast=False,
            stop_event=threading.Event(),
            errors=queue.Queue(),
            progress_queue=queue.Queue(),
            batch_size=0,
        )


def test_batched_read_failure_is_reported(tmp_path, monkeypatch):
    def boom(_path):
        raise ValueError("cannot read")

    monkeypatch.setattr(workers, "_read_frame_images", boom)

    tasks = [FrameTask(image_id="f0", frame_path=Path("/x/f0.jpg"))]
    yolo = _RecordingYolo()

    in_q: queue.Queue = queue.Queue()
    out_q: queue.Queue = queue.Queue()
    vlm_q: queue.Queue = queue.Queue()
    prog_q: queue.Queue = queue.Queue()
    errs: queue.Queue = queue.Queue()
    stop = threading.Event()
    for t in tasks:
        in_q.put(t)
    in_q.put(workers.SENTINEL)

    th = threading.Thread(
        target=workers.yolo_worker_batched,
        kwargs={
            "input_queue": in_q,
            "output_queue": out_q,
            "vlm_queue": vlm_q,
            "yolo_model": yolo,
            "prompts": {},
            "fail_fast": False,
            "stop_event": stop,
            "errors": errs,
            "progress_queue": prog_q,
            "batch_size": 4,
        },
        daemon=True,
    )
    th.start()
    in_q.join()
    th.join(timeout=5)

    assert not th.is_alive()
    assert not errs.empty()  # read failure recorded
    assert yolo.batch_sizes == []  # nothing reached inference
