"""Tests for src/workers.py::sam_worker — multi-threaded SAM2 sharing, skip callback."""
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cv2")

from src.workers import sam_worker, SENTINEL
from src.structs import YoloStageResult, PipelineProgressEvent


def _make_task(image_id: str, bboxes: list, size=(20, 20)) -> YoloStageResult:
    from PIL import Image
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    return YoloStageResult(
        image_id=image_id,
        frame_path=Path(f"{image_id}.jpg"),
        original_image=img,
        pil_image=Image.fromarray(img),
        detections={"bbox": bboxes},
        vis_image_boxes=img,
    )


class FakeSegmenter:
    """Tracks concurrent access to catch race conditions without a lock."""

    def __init__(self, work_seconds: float = 0.0):
        self._work_seconds = work_seconds
        self._active = 0
        self.max_concurrent_seen = 0
        self._lock = threading.Lock()
        self._image = None

    def set_image(self, pil_image):
        with self._lock:
            self._active += 1
            self.max_concurrent_seen = max(self.max_concurrent_seen, self._active)
        self._image = pil_image
        if self._work_seconds:
            time.sleep(self._work_seconds)

    def predict_from_boxes(self, boxes, multimask_output=False):
        h, w = self._image.height, self._image.width
        masks = np.zeros((len(boxes), h, w), dtype=bool)
        for i in range(len(boxes)):
            masks[i, 1:3, 1:3] = True
        with self._lock:
            self._active -= 1
        return SimpleNamespace(masks=masks)


def _run_worker_to_completion(*, tasks, segmenter, segmenter_lock=None, on_pointing_skipped=None):
    input_queue: queue.Queue = queue.Queue()
    vlm_queue: queue.Queue = queue.Queue()
    progress_queue: queue.Queue = queue.Queue()
    errors: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    for t in tasks:
        input_queue.put(t)
    input_queue.put(SENTINEL)

    sam_worker(
        input_queue=input_queue,
        vlm_queue=vlm_queue,
        segmenter=segmenter,
        prompts={"vlm": {"pointing": {"system": "s: {{ points_dict }}", "user": "u"}}},
        fail_fast=False,
        stop_event=stop_event,
        errors=errors,
        progress_queue=progress_queue,
        segmenter_lock=segmenter_lock,
        on_pointing_skipped=on_pointing_skipped,
    )
    return vlm_queue, progress_queue, errors


class TestSamWorkerBasic:
    def test_produces_pointing_task_for_detections(self):
        task = _make_task("img1", bboxes=[[1, 1, 5, 5]])
        vlm_queue, progress_queue, errors = _run_worker_to_completion(
            tasks=[task], segmenter=FakeSegmenter(),
        )
        assert vlm_queue.qsize() == 1
        assert errors.empty()

    def test_skips_pointing_when_no_detections(self):
        task = _make_task("img1", bboxes=[])
        vlm_queue, progress_queue, errors = _run_worker_to_completion(
            tasks=[task], segmenter=FakeSegmenter(),
        )
        assert vlm_queue.empty()
        event = progress_queue.get_nowait()
        assert event.kind == "pointing_skipped"
        assert event.image_id == "img1"

    def test_on_pointing_skipped_callback_invoked(self):
        task = _make_task("img1", bboxes=[])
        skipped_tasks = []
        _run_worker_to_completion(
            tasks=[task], segmenter=FakeSegmenter(),
            on_pointing_skipped=lambda t: skipped_tasks.append(t.image_id),
        )
        assert skipped_tasks == ["img1"]

    def test_on_pointing_skipped_not_called_when_detections_present(self):
        task = _make_task("img1", bboxes=[[1, 1, 5, 5]])
        skipped_tasks = []
        _run_worker_to_completion(
            tasks=[task], segmenter=FakeSegmenter(),
            on_pointing_skipped=lambda t: skipped_tasks.append(t.image_id),
        )
        assert skipped_tasks == []

    def test_on_pointing_skipped_none_does_not_raise(self):
        task = _make_task("img1", bboxes=[])
        # on_pointing_skipped=None is the default — must not raise.
        _run_worker_to_completion(tasks=[task], segmenter=FakeSegmenter())


class TestMultipleSamWorkersSharedSegmenter:
    """Multiple sam_worker threads must not corrupt each other's masks."""

    def test_lock_serializes_gpu_calls_no_interleave(self):
        n_threads = 4
        n_tasks_per_thread = 5
        segmenter = FakeSegmenter(work_seconds=0.01)
        segmenter_lock = threading.Lock()

        def make_input_queue():
            q: queue.Queue = queue.Queue()
            for i in range(n_tasks_per_thread):
                q.put(_make_task(f"img{i}", bboxes=[[1, 1, 5, 5]]))
            q.put(SENTINEL)
            return q

        threads = []
        vlm_queues = []
        for _ in range(n_threads):
            input_q = make_input_queue()
            vlm_q: queue.Queue = queue.Queue()
            vlm_queues.append(vlm_q)
            t = threading.Thread(
                target=sam_worker,
                kwargs={
                    "input_queue": input_q,
                    "vlm_queue": vlm_q,
                    "segmenter": segmenter,
                    "prompts": {"vlm": {"pointing": {"system": "s: {{ points_dict }}", "user": "u"}}},
                    "fail_fast": False,
                    "stop_event": threading.Event(),
                    "errors": queue.Queue(),
                    "progress_queue": queue.Queue(),
                    "segmenter_lock": segmenter_lock,
                },
            )
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
            assert not t.is_alive()

        # With the lock held around set_image+predict, only one thread's
        # GPU call should ever be "active" at a time.
        assert segmenter.max_concurrent_seen == 1

        total_produced = sum(q.qsize() for q in vlm_queues)
        assert total_produced == n_threads * n_tasks_per_thread

    def test_without_lock_concurrency_is_unbounded(self):
        """Sanity check that the FakeSegmenter's race detector actually works:
        without a lock, concurrent set_image calls ARE observed overlapping."""
        n_threads = 4
        segmenter = FakeSegmenter(work_seconds=0.05)

        def make_input_queue():
            q: queue.Queue = queue.Queue()
            q.put(_make_task("img", bboxes=[[1, 1, 5, 5]]))
            q.put(SENTINEL)
            return q

        threads = [
            threading.Thread(
                target=sam_worker,
                kwargs={
                    "input_queue": make_input_queue(),
                    "vlm_queue": queue.Queue(),
                    "segmenter": segmenter,
                    "prompts": {"vlm": {"pointing": {"system": "s: {{ points_dict }}", "user": "u"}}},
                    "fail_fast": False,
                    "stop_event": threading.Event(),
                    "errors": queue.Queue(),
                    "progress_queue": queue.Queue(),
                    "segmenter_lock": None,  # no lock: race is possible
                },
            )
            for _ in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert segmenter.max_concurrent_seen > 1
