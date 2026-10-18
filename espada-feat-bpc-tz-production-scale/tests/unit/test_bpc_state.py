"""Tests for src/bpc/state.py — StateLogger append-only event journaling."""
import json
import pytest
from pathlib import Path

from src.bpc.state import StateLogger


def _make_logger(tmp_path: Path, run_id: str = "test-run") -> StateLogger:
    return StateLogger(logs_dir=tmp_path / "logs", run_id=run_id)


class TestStateLogger:
    def test_event_creates_state_jsonl(self, tmp_path):
        lg = _make_logger(tmp_path)
        lg.event(image_id="img1", relative_path="img1.jpg", stage="vlm",
                 task_name="bboxing", event="done")
        state_file = tmp_path / "logs" / "state.jsonl"
        assert state_file.exists()
        lines = [json.loads(l) for l in state_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        assert lines[0]["event"] == "done"
        assert lines[0]["image_id"] == "img1"

    def test_failed_creates_failed_jsonl(self, tmp_path):
        lg = _make_logger(tmp_path)
        lg.failed(image_id="img2", relative_path="img2.jpg", stage="vlm",
                  task_name="pointing", reason="HTTP 500", attempt=3)
        failed_file = tmp_path / "logs" / "failed.jsonl"
        assert failed_file.exists()
        rec = json.loads(failed_file.read_text().splitlines()[0])
        assert rec["reason"] == "HTTP 500"
        assert rec["attempt"] == 3

    def test_multiple_events_appended(self, tmp_path):
        lg = _make_logger(tmp_path)
        for i in range(5):
            lg.event(image_id=f"img{i}", relative_path=f"img{i}.jpg",
                     stage="vlm", task_name="bboxing", event="done", attempt=1)
        lines = (tmp_path / "logs" / "state.jsonl").read_text().splitlines()
        assert len(lines) == 5

    def test_load_completed_empty(self, tmp_path):
        lg = _make_logger(tmp_path)
        assert lg.load_completed() == set()

    def test_load_completed_finds_done(self, tmp_path):
        lg = _make_logger(tmp_path)
        lg.event(image_id="img1", relative_path="img1.jpg", stage="vlm",
                 task_name="bboxing", event="done")
        lg.event(image_id="img1", relative_path="img1.jpg", stage="vlm",
                 task_name="captions", event="done")
        lg.event(image_id="img2", relative_path="img2.jpg", stage="vlm",
                 task_name="bboxing", event="started")  # not done
        done = lg.load_completed()
        assert ("img1", "bboxing") in done
        assert ("img1", "captions") in done
        assert ("img2", "bboxing") not in done

    def test_load_completed_ignores_corrupt_lines(self, tmp_path):
        lg = _make_logger(tmp_path)
        lg.event(image_id="img1", relative_path="img1.jpg", stage="vlm",
                 task_name="bboxing", event="done")
        # Manually append a corrupt line
        (tmp_path / "logs" / "state.jsonl").open("a").write("not json\n")
        done = lg.load_completed()
        assert ("img1", "bboxing") in done

    def test_event_includes_duration(self, tmp_path):
        lg = _make_logger(tmp_path)
        lg.event(image_id="img1", relative_path="img1.jpg", stage="vlm",
                 task_name="bboxing", event="done", duration_sec=3.14)
        rec = json.loads((tmp_path / "logs" / "state.jsonl").read_text().splitlines()[0])
        assert abs(rec["duration_sec"] - 3.14) < 0.01

    def test_event_includes_run_id(self, tmp_path):
        lg = _make_logger(tmp_path, run_id="my-run-123")
        lg.event(image_id="img1", relative_path="img1.jpg", stage="vlm",
                 task_name="bboxing", event="done")
        rec = json.loads((tmp_path / "logs" / "state.jsonl").read_text().splitlines()[0])
        assert rec["run_id"] == "my-run-123"

    def test_thread_safe_concurrent_writes(self, tmp_path):
        import threading
        lg = _make_logger(tmp_path)
        threads = [
            threading.Thread(target=lambda i=i: lg.event(
                image_id=f"img{i}", relative_path=f"img{i}.jpg",
                stage="vlm", task_name="bboxing", event="done"
            ))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        lines = (tmp_path / "logs" / "state.jsonl").read_text().splitlines()
        assert len(lines) == 20
