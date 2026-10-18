"""Tests for src/bpc/status.py — state.jsonl aggregation into per-bucket counts.

No coverage existed for this file before, which is exactly why the
judge-vs-base-VLM task_name collision shipped: judge events write the SAME
task_name ("bboxing"/"pointing") as the underlying VLM task -- only
``stage`` ("vlm" vs "judge") tells them apart. Grouping by task_name alone
merged judge "done" events into the base bucket, so "bboxing-judged"/
"pointing-judged" stayed permanently 0 in `espada-bpc status` even while
outputs/bboxing-judged/*.jsonl was filling up correctly on a live run.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.bpc.status import read_status, _display_name


def _write_state(tmp_path: Path, records: list[dict]) -> Path:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_path = logs_dir / "state.jsonl"
    with state_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return tmp_path


def _event(*, stage, task_name, event, image_id="img1", **extra) -> dict:
    rec = {
        "time": "2026-07-02T00:00:00+00:00", "run_id": "r", "image_id": image_id,
        "relative_path": f"{image_id}.jpg", "stage": stage, "task_name": task_name,
        "event": event, "worker_id": "", "attempt": 1, "error": None,
    }
    rec.update(extra)
    return rec


class TestDisplayName:
    def test_vlm_stage_uses_task_name_directly(self):
        assert _display_name("vlm", "bboxing") == "bboxing"
        assert _display_name("vlm", "pointing") == "pointing"
        assert _display_name("vlm", "captions") == "captions"

    def test_judge_stage_appends_judged_suffix(self):
        assert _display_name("judge", "bboxing") == "bboxing-judged"
        assert _display_name("judge", "pointing") == "pointing-judged"

    def test_sam2_stage_uses_task_name_directly(self):
        assert _display_name("sam2", "pointing") == "pointing"


class TestReadStatusJudgeVlmCollision:
    """The exact regression: same task_name, different stage."""

    def test_judge_done_does_not_leak_into_base_bboxing_bucket(self, tmp_path):
        run_dir = _write_state(tmp_path, [
            _event(stage="vlm", task_name="bboxing", event="done", image_id="img1"),
            _event(stage="judge", task_name="bboxing", event="done", image_id="img1"),
        ])
        data = read_status(run_dir)
        assert data["counts"]["bboxing"]["done"] == 1
        assert data["counts"]["bboxing-judged"]["done"] == 1

    def test_judge_and_vlm_running_tracked_independently(self, tmp_path):
        run_dir = _write_state(tmp_path, [
            _event(stage="vlm", task_name="bboxing", event="started", image_id="img1"),
            _event(stage="vlm", task_name="bboxing", event="done", image_id="img1"),
            _event(stage="judge", task_name="bboxing", event="started", image_id="img1"),
            # judge for img1 is still running -- must not be masked by the
            # base VLM "done" on the same image_id/task_name.
        ])
        data = read_status(run_dir)
        assert data["counts"]["bboxing"]["done"] == 1
        assert data["counts"]["bboxing"].get("running", 0) == 0
        assert data["counts"]["bboxing-judged"].get("running", 0) == 1
        assert data["counts"]["bboxing-judged"].get("done", 0) == 0

    def test_pointing_judge_separate_from_pointing_vlm(self, tmp_path):
        run_dir = _write_state(tmp_path, [
            _event(stage="vlm", task_name="pointing", event="done", image_id="img1"),
            _event(stage="vlm", task_name="pointing", event="done", image_id="img2"),
            _event(stage="judge", task_name="pointing", event="done", image_id="img1"),
        ])
        data = read_status(run_dir)
        assert data["counts"]["pointing"]["done"] == 2
        assert data["counts"]["pointing-judged"]["done"] == 1


class TestReadStatusBasics:
    def test_empty_state_file(self, tmp_path):
        run_dir = _write_state(tmp_path, [])
        data = read_status(run_dir)
        assert data["counts"] == {}

    def test_missing_state_file(self, tmp_path):
        (tmp_path / "logs").mkdir()
        data = read_status(tmp_path)
        assert data["counts"] == {}

    def test_counts_done_failed_skip_retry(self, tmp_path):
        run_dir = _write_state(tmp_path, [
            _event(stage="vlm", task_name="bboxing", event="done", image_id="a"),
            _event(stage="vlm", task_name="bboxing", event="failed", image_id="b"),
            _event(stage="sam2", task_name="pointing", event="skip", image_id="c"),
            _event(stage="vlm", task_name="bboxing", event="retry", image_id="d"),
        ])
        data = read_status(run_dir)
        assert data["counts"]["bboxing"]["done"] == 1
        assert data["counts"]["bboxing"]["failed"] == 1
        assert data["counts"]["pointing"]["skipped"] == 1
        assert data["counts"]["bboxing"]["retry"] == 1

    def test_started_without_done_counts_as_running(self, tmp_path):
        run_dir = _write_state(tmp_path, [
            _event(stage="vlm", task_name="captions", event="started", image_id="x"),
        ])
        data = read_status(run_dir)
        assert data["counts"]["captions"]["running"] == 1

    def test_started_then_done_not_running(self, tmp_path):
        run_dir = _write_state(tmp_path, [
            _event(stage="vlm", task_name="captions", event="started", image_id="x"),
            _event(stage="vlm", task_name="captions", event="done", image_id="x"),
        ])
        data = read_status(run_dir)
        assert data["counts"]["captions"].get("running", 0) == 0
        assert data["counts"]["captions"]["done"] == 1

    def test_ignores_corrupt_lines(self, tmp_path):
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True)
        (logs_dir / "state.jsonl").write_text(
            json.dumps(_event(stage="vlm", task_name="bboxing", event="done")) + "\n"
            + "not valid json\n"
        )
        data = read_status(tmp_path)
        assert data["counts"]["bboxing"]["done"] == 1

    def test_permanently_failed_from_failed_jsonl_respects_stage(self, tmp_path):
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True)
        (logs_dir / "failed.jsonl").write_text(
            json.dumps({
                "image_id": "img1", "task_name": "bboxing", "stage": "judge",
                "reason": "coords changed", "attempt": 3,
            }) + "\n"
        )
        data = read_status(tmp_path)
        assert data["permanently_failed"]["bboxing-judged"] == 1
        assert "bboxing" not in data["permanently_failed"]
