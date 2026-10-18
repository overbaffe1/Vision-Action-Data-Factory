"""Append-only state.jsonl logger for pipeline events and failed.jsonl tracker."""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


logger = logging.getLogger("bpc.state")


class StateLogger:
    """Thread-safe append-only writer to logs/state.jsonl and logs/failed.jsonl."""

    def __init__(self, logs_dir: Path, run_id: str) -> None:
        self._logs_dir = logs_dir
        self._run_id = run_id
        self._lock = threading.Lock()
        self._state_path = logs_dir / "state.jsonl"
        self._failed_path = logs_dir / "failed.jsonl"
        logs_dir.mkdir(parents=True, exist_ok=True)

    def _append(self, path: Path, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def event(
        self,
        *,
        image_id: str,
        relative_path: str,
        stage: str,
        task_name: str,
        event: str,
        worker_id: str = "",
        attempt: int = 1,
        duration_sec: float | None = None,
        output_path: str | None = None,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "time": self._now(),
            "run_id": self._run_id,
            "image_id": image_id,
            "relative_path": relative_path,
            "stage": stage,
            "task_name": task_name,
            "event": event,
            "worker_id": worker_id,
            "attempt": attempt,
        }
        if duration_sec is not None:
            record["duration_sec"] = round(duration_sec, 3)
        if output_path is not None:
            record["output_path"] = output_path
        record["error"] = error
        if extra:
            record.update(extra)
        self._append(self._state_path, record)

    def failed(
        self,
        *,
        image_id: str,
        relative_path: str,
        stage: str,
        task_name: str,
        reason: str,
        attempt: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "time": self._now(),
            "run_id": self._run_id,
            "image_id": image_id,
            "relative_path": relative_path,
            "stage": stage,
            "task_name": task_name,
            "reason": reason,
            "attempt": attempt,
        }
        if extra:
            record.update(extra)
        self._append(self._failed_path, record)
        logger.warning(
            "FAILED image_id=%s task=%s/%s attempt=%d: %s",
            image_id, stage, task_name, attempt, reason,
        )

    def load_completed(self) -> set[tuple[str, str]]:
        """Returns set of (image_id, task_name) that already have event='done'."""
        done: set[tuple[str, str]] = set()
        if not self._state_path.exists():
            return done
        with self._state_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("event") == "done":
                        done.add((rec["image_id"], rec["task_name"]))
                except (json.JSONDecodeError, KeyError):
                    pass
        return done
