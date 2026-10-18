"""Tests for src/bpc/pipeline.py::_enqueue_judge — backpressure and empty-answer skip."""
import json
import queue

import pytest

from src.bpc.pipeline import _enqueue_judge
from src.bpc.state import StateLogger


def _logger(tmp_path):
    return StateLogger(tmp_path / "logs", "run-test")


def _item(answer):
    return {
        "task_name": "bboxing",
        "image_id": "img1",
        "relative_path": "img1.jpg",
        "original_result": {"images": ["img1.jpg"], "question": "q", "answer": answer},
        "images": [],
    }


class TestEmptyAnswerSkip:
    """Judge may only edit labels on existing items — an empty answer gives it
    nothing to do, and on live runs the 35B judge hallucinated objects into
    empty lists (always rejected by verify_judge_output). Skip entirely."""

    def test_empty_answer_not_enqueued(self, tmp_path):
        q: queue.Queue = queue.Queue(maxsize=8)
        _enqueue_judge(q, _item(answer=[]), state_logger=_logger(tmp_path))
        assert q.empty()

    def test_empty_answer_writes_skip_event(self, tmp_path):
        q: queue.Queue = queue.Queue(maxsize=8)
        lg = _logger(tmp_path)
        _enqueue_judge(q, _item(answer=[]), state_logger=lg)
        records = [
            json.loads(l)
            for l in (tmp_path / "logs" / "state.jsonl").read_text().splitlines()
            if l.strip()
        ]
        assert len(records) == 1
        assert records[0]["stage"] == "judge"
        assert records[0]["event"] == "skip"

    def test_nonempty_answer_enqueued(self, tmp_path):
        q: queue.Queue = queue.Queue(maxsize=8)
        item = _item(answer=[{"label": "car", "bbox_2d": [0, 0, 10, 10]}])
        _enqueue_judge(q, item, state_logger=_logger(tmp_path))
        assert q.qsize() == 1
        assert q.get_nowait() is item

    def test_missing_answer_key_treated_as_empty(self, tmp_path):
        q: queue.Queue = queue.Queue(maxsize=8)
        item = _item(answer=[])
        del item["original_result"]["answer"]
        _enqueue_judge(q, item, state_logger=_logger(tmp_path))
        assert q.empty()


class TestQueueFullDrop:
    def test_full_queue_drops_and_logs_failed(self, tmp_path):
        q: queue.Queue = queue.Queue(maxsize=1)
        q.put("filler")
        lg = _logger(tmp_path)
        item = _item(answer=[{"label": "car", "bbox_2d": [0, 0, 10, 10]}])
        _enqueue_judge(q, item, state_logger=lg, timeout=0.1)

        assert q.qsize() == 1  # only the filler; item was dropped
        records = [
            json.loads(l)
            for l in (tmp_path / "logs" / "state.jsonl").read_text().splitlines()
            if l.strip()
        ]
        assert records[0]["stage"] == "judge"
        assert records[0]["event"] == "failed"
        assert "full" in records[0]["error"]
