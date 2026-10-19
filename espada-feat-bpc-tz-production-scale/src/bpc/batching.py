"""Index-based batching with checkpoint/resume for very large datasets.

For 12M+ images, scanning every output file at startup is too slow. Instead we:
  1. Build a stable, sorted list of input images (the "index space").
  2. Slice it into fixed-size numbered batches.
  3. Record the last fully-completed batch per (node, gpu) in progress.json.
  4. On restart, skip all batches up to and including the last completed one.

This mirrors the reference pipeline's START_INDEX..END_INDEX range sharding,
but generalized to per-worker checkpoints written atomically.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("bpc.batching")


@dataclass(frozen=True)
class IndexBatch:
    """A contiguous slice of the global image index space."""

    batch_id: int
    start: int           # inclusive global index
    end: int             # exclusive global index
    paths: list[Path]

    @property
    def size(self) -> int:
        return len(self.paths)


def make_index_batches(images: list[Path], batch_size: int) -> list[IndexBatch]:
    """Slices a sorted image list into numbered contiguous batches.

    The image list MUST be deterministically ordered (sorted) so that batch_id
    maps to the same images across restarts.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    batches: list[IndexBatch] = []
    for batch_id, start in enumerate(range(0, len(images), batch_size)):
        end = min(start + batch_size, len(images))
        batches.append(
            IndexBatch(batch_id=batch_id, start=start, end=end, paths=images[start:end])
        )
    return batches


class ProgressCheckpoint:
    """Tracks the last completed batch id, written atomically to progress.json.

    Layout:
        {
          "schema": 1,
          "total_images": 12000000,
          "batch_size": 1000,
          "last_completed_batch": 4217,   # global, monotonic
          "completed_batches": [...],     # optional sparse list for out-of-order
          "shards": {"node0_gpu0": 4217, ...}  # per-worker high-water marks
        }
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {
            "schema": 1,
            "total_images": 0,
            "batch_size": 0,
            "last_completed_batch": -1,
            "completed_batches": [],
            "shards": {},
        }
        if path.exists():
            self._load()

    def _load(self) -> None:
        try:
            self._data = json.loads(self._path.read_text(encoding="utf-8"))
            self._data.setdefault("completed_batches", [])
            self._data.setdefault("shards", {})
            self._data.setdefault("last_completed_batch", -1)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load progress.json (%s); starting fresh.", exc)

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def init_run(self, *, total_images: int, batch_size: int) -> None:
        with self._lock:
            self._data["total_images"] = total_images
            self._data["batch_size"] = batch_size
            self._flush()

    def mark_batch_done(self, batch_id: int, *, shard_key: str | None = None) -> None:
        """Records a batch as completed. Advances the contiguous high-water mark."""
        with self._lock:
            completed = set(self._data["completed_batches"])
            completed.add(batch_id)

            # Advance the contiguous last_completed_batch as far as possible.
            last = self._data["last_completed_batch"]
            while (last + 1) in completed:
                last += 1
            self._data["last_completed_batch"] = last

            # Keep only the sparse "ahead" batches to bound file size.
            self._data["completed_batches"] = sorted(b for b in completed if b > last)

            if shard_key is not None:
                shards = self._data["shards"]
                shards[shard_key] = max(shards.get(shard_key, -1), batch_id)

            self._flush()

    @property
    def last_completed_batch(self) -> int:
        with self._lock:
            return self._data["last_completed_batch"]

    def is_batch_done(self, batch_id: int) -> bool:
        with self._lock:
            if batch_id <= self._data["last_completed_batch"]:
                return True
            return batch_id in self._data["completed_batches"]

    def pending_batches(self, batches: list[IndexBatch]) -> list[IndexBatch]:
        """Filters out already-completed batches for fast resume."""
        with self._lock:
            last = self._data["last_completed_batch"]
            ahead = set(self._data["completed_batches"])
        return [b for b in batches if b.batch_id > last and b.batch_id not in ahead]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)
