"""Tests for src/bpc/batching.py — index batches + checkpoint/resume."""
import json
import pytest
from pathlib import Path

from src.bpc.batching import make_index_batches, ProgressCheckpoint, IndexBatch


def _paths(n: int) -> list[Path]:
    return [Path(f"/data/img_{i:06d}.jpg") for i in range(n)]


class TestMakeIndexBatches:
    def test_even_split(self):
        batches = make_index_batches(_paths(10), batch_size=5)
        assert len(batches) == 2
        assert batches[0].batch_id == 0
        assert batches[0].start == 0 and batches[0].end == 5
        assert batches[1].start == 5 and batches[1].end == 10
        assert batches[0].size == 5

    def test_uneven_split(self):
        batches = make_index_batches(_paths(7), batch_size=3)
        assert len(batches) == 3
        assert batches[-1].size == 1
        assert batches[-1].start == 6 and batches[-1].end == 7

    def test_single_batch(self):
        batches = make_index_batches(_paths(3), batch_size=10)
        assert len(batches) == 1
        assert batches[0].size == 3

    def test_empty_input(self):
        assert make_index_batches([], batch_size=5) == []

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError):
            make_index_batches(_paths(5), batch_size=0)

    def test_batches_cover_all_images_disjointly(self):
        images = _paths(100)
        batches = make_index_batches(images, batch_size=7)
        reconstructed = [p for b in batches for p in b.paths]
        assert reconstructed == images

    def test_batch_ids_contiguous(self):
        batches = make_index_batches(_paths(50), batch_size=10)
        assert [b.batch_id for b in batches] == [0, 1, 2, 3, 4]

    def test_deterministic(self):
        a = make_index_batches(_paths(20), batch_size=6)
        b = make_index_batches(_paths(20), batch_size=6)
        assert [x.paths for x in a] == [x.paths for x in b]


class TestProgressCheckpoint:
    def test_starts_empty(self, tmp_path):
        cp = ProgressCheckpoint(tmp_path / "progress.json")
        assert cp.last_completed_batch == -1

    def test_init_run_persists(self, tmp_path):
        p = tmp_path / "progress.json"
        cp = ProgressCheckpoint(p)
        cp.init_run(total_images=1000, batch_size=100)
        assert p.exists()
        data = json.loads(p.read_text())
        assert data["total_images"] == 1000
        assert data["batch_size"] == 100

    def test_mark_batch_advances_contiguous(self, tmp_path):
        cp = ProgressCheckpoint(tmp_path / "progress.json")
        cp.mark_batch_done(0)
        cp.mark_batch_done(1)
        cp.mark_batch_done(2)
        assert cp.last_completed_batch == 2

    def test_out_of_order_does_not_advance_past_gap(self, tmp_path):
        cp = ProgressCheckpoint(tmp_path / "progress.json")
        cp.mark_batch_done(0)
        cp.mark_batch_done(2)  # gap at 1
        assert cp.last_completed_batch == 0
        assert cp.is_batch_done(2) is True
        # Now fill the gap
        cp.mark_batch_done(1)
        assert cp.last_completed_batch == 2

    def test_is_batch_done(self, tmp_path):
        cp = ProgressCheckpoint(tmp_path / "progress.json")
        cp.mark_batch_done(0)
        cp.mark_batch_done(1)
        assert cp.is_batch_done(0) is True
        assert cp.is_batch_done(1) is True
        assert cp.is_batch_done(2) is False

    def test_resume_from_disk(self, tmp_path):
        p = tmp_path / "progress.json"
        cp1 = ProgressCheckpoint(p)
        cp1.mark_batch_done(0)
        cp1.mark_batch_done(1)
        # Simulate restart
        cp2 = ProgressCheckpoint(p)
        assert cp2.last_completed_batch == 1
        assert cp2.is_batch_done(1) is True

    def test_pending_batches_skips_done(self, tmp_path):
        cp = ProgressCheckpoint(tmp_path / "progress.json")
        batches = make_index_batches(_paths(50), batch_size=10)  # 5 batches
        cp.mark_batch_done(0)
        cp.mark_batch_done(1)
        pending = cp.pending_batches(batches)
        assert [b.batch_id for b in pending] == [2, 3, 4]

    def test_pending_batches_all_when_fresh(self, tmp_path):
        cp = ProgressCheckpoint(tmp_path / "progress.json")
        batches = make_index_batches(_paths(30), batch_size=10)
        pending = cp.pending_batches(batches)
        assert len(pending) == 3

    def test_shard_high_water_mark(self, tmp_path):
        cp = ProgressCheckpoint(tmp_path / "progress.json")
        cp.mark_batch_done(0, shard_key="node0")
        cp.mark_batch_done(1, shard_key="node0")
        snap = cp.snapshot()
        assert snap["shards"]["node0"] == 1

    def test_corrupt_progress_recovers(self, tmp_path):
        p = tmp_path / "progress.json"
        p.write_text("{ broken json")
        cp = ProgressCheckpoint(p)  # should not raise
        assert cp.last_completed_batch == -1

    def test_atomic_write_no_tmp_left(self, tmp_path):
        p = tmp_path / "progress.json"
        cp = ProgressCheckpoint(p)
        cp.mark_batch_done(0)
        assert list(tmp_path.glob("*.tmp")) == []

    def test_thread_safe_marking(self, tmp_path):
        import threading
        cp = ProgressCheckpoint(tmp_path / "progress.json")
        threads = [threading.Thread(target=cp.mark_batch_done, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # All 0..19 marked -> contiguous to 19
        assert cp.last_completed_batch == 19


class TestResumeScenario:
    """End-to-end resume scenario: crash mid-run, restart skips done batches."""

    def test_full_resume_cycle(self, tmp_path):
        images = _paths(1000)
        batches = make_index_batches(images, batch_size=100)  # 10 batches

        # First run: complete 4 batches then "crash"
        cp = ProgressCheckpoint(tmp_path / "progress.json")
        cp.init_run(total_images=len(images), batch_size=100)
        for b in batches[:4]:
            cp.mark_batch_done(b.batch_id)

        # Restart: load checkpoint, get pending
        cp2 = ProgressCheckpoint(tmp_path / "progress.json")
        pending = cp2.pending_batches(batches)
        assert len(pending) == 6
        assert pending[0].batch_id == 4
        # No re-scan of the 400 already-done images needed
        assert pending[0].start == 400
