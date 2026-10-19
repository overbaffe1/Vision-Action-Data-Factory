"""Tests for src/workers.py::_put_with_backpressure_log — observable blocking put."""
import queue
import threading
import time
import logging
import pytest

from src.workers import _put_with_backpressure_log


class TestPutWithBackpressureLog:
    def test_puts_immediately_when_queue_has_room(self):
        q = queue.Queue(maxsize=5)
        _put_with_backpressure_log(q, "item", queue_name="test_queue", warn_every=10)
        assert q.qsize() == 1
        assert q.get_nowait() == "item"

    def test_blocks_until_space_available(self):
        q = queue.Queue(maxsize=1)
        q.put("filler")

        result = {"done": False}

        def producer():
            _put_with_backpressure_log(q, "item2", queue_name="test_queue", warn_every=10)
            result["done"] = True

        t = threading.Thread(target=producer, daemon=True)
        t.start()
        time.sleep(0.1)
        assert not result["done"]  # still blocked, queue full

        q.get_nowait()  # free up space
        t.join(timeout=2)
        assert result["done"]
        assert q.get_nowait() == "item2"

    def test_logs_warning_when_blocked_past_threshold(self, caplog):
        q = queue.Queue(maxsize=1)
        q.put("filler")

        def producer():
            _put_with_backpressure_log(q, "item2", queue_name="my_queue", warn_every=0.2)

        t = threading.Thread(target=producer, daemon=True)
        with caplog.at_level(logging.WARNING, logger="pipeline"):
            t.start()
            time.sleep(0.5)  # let it exceed warn_every at least once
            q.get_nowait()  # unblock
            t.join(timeout=2)

        assert any("my_queue is full" in r.message for r in caplog.records)

    def test_does_not_drop_items(self):
        """Unlike a dropping backpressure strategy, every item must eventually land."""
        q = queue.Queue(maxsize=2)
        items = [f"item{i}" for i in range(5)]
        consumed = []

        def consumer():
            while len(consumed) < len(items):
                consumed.append(q.get())
                time.sleep(0.01)

        c = threading.Thread(target=consumer, daemon=True)
        c.start()

        for item in items:
            _put_with_backpressure_log(q, item, queue_name="test_queue", warn_every=5)

        c.join(timeout=3)
        assert sorted(consumed) == sorted(items)
