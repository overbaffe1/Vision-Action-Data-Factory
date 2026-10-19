"""Smoke tests for espada_bpc CLI commands (prepare-images, status, retry-failed)."""
import json
import pytest
from pathlib import Path
from click.testing import CliRunner

pytest.importorskip("cv2")

from espada_bpc import cli


def _write_jpeg(path: Path, w: int = 200, h: int = 150) -> Path:
    from PIL import Image
    img = Image.new("RGB", (w, h), color=(100, 150, 200))
    img.save(path, format="JPEG")
    return path


class TestPrepareImages:
    def test_converts_jpeg(self, tmp_path):
        src = tmp_path / "raw"
        dst = tmp_path / "prepared"
        src.mkdir()
        _write_jpeg(src / "img001.jpg", 1200, 900)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "prepare-images",
            "--input", str(src),
            "--output", str(dst),
            "--size", "768",
        ])
        assert result.exit_code == 0, result.output
        out_files = list(dst.rglob("*.jpg"))
        assert len(out_files) == 1

    def test_downscales_large_image(self, tmp_path):
        from PIL import Image as PILImage
        src = tmp_path / "raw"
        dst = tmp_path / "prepared"
        src.mkdir()
        _write_jpeg(src / "big.jpg", 2000, 1500)

        runner = CliRunner()
        runner.invoke(cli, [
            "prepare-images", "--input", str(src), "--output", str(dst), "--size", "768"
        ])
        out = list(dst.rglob("*.jpg"))[0]
        with PILImage.open(out) as img:
            assert max(img.size) <= 768

    def test_skips_already_converted(self, tmp_path):
        src = tmp_path / "raw"
        dst = tmp_path / "prepared"
        src.mkdir()
        dst.mkdir()
        _write_jpeg(src / "img.jpg")
        # Pre-place an output file
        _write_jpeg(dst / "img.jpg")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "prepare-images", "--input", str(src), "--output", str(dst), "--size", "768"
        ])
        assert "skipped" in result.output

    def test_no_images_exits_nonzero(self, tmp_path):
        src = tmp_path / "empty"
        src.mkdir()
        runner = CliRunner()
        result = runner.invoke(cli, [
            "prepare-images", "--input", str(src), "--output", str(tmp_path / "out"), "--size", "768"
        ])
        assert result.exit_code != 0


class TestStatusCommand:
    def test_status_shows_tasks(self, tmp_path):
        # Create a minimal state.jsonl
        logs = tmp_path / "logs"
        logs.mkdir()
        state = logs / "state.jsonl"
        import json as json_mod
        records = [
            {"image_id": "img1", "task_name": "bboxing", "event": "done", "run_id": "r1",
             "relative_path": "img1.jpg", "stage": "vlm", "worker_id": "", "attempt": 1, "error": None},
            {"image_id": "img2", "task_name": "pointing", "event": "done", "run_id": "r1",
             "relative_path": "img2.jpg", "stage": "vlm", "worker_id": "", "attempt": 1, "error": None},
        ]
        state.write_text("\n".join(json_mod.dumps(r) for r in records))

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--run", str(tmp_path)])
        assert result.exit_code == 0
        assert "bboxing" in result.output
        assert "pointing" in result.output

    def test_status_empty_run(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--run", str(tmp_path)])
        assert result.exit_code == 0


class TestRetryFailedCommand:
    def _make_run(self, tmp_path: Path) -> Path:
        logs = tmp_path / "logs"
        logs.mkdir(parents=True)
        outputs = tmp_path / "outputs" / "bboxing"
        outputs.mkdir(parents=True)

        import json as json_mod
        # Write a failed entry
        failed = logs / "failed.jsonl"
        failed.write_text(json_mod.dumps({
            "image_id": "img1", "task_name": "bboxing",
            "stage": "vlm", "reason": "HTTP 500", "attempt": 3,
            "relative_path": "img1.jpg", "run_id": "r1",
        }) + "\n")

        # Write a state entry with done for img1/bboxing
        state = logs / "state.jsonl"
        state.write_text(json_mod.dumps({
            "image_id": "img1", "task_name": "bboxing", "event": "done",
            "run_id": "r1", "relative_path": "img1.jpg", "stage": "vlm",
            "worker_id": "", "attempt": 1, "error": None,
        }) + "\n")

        # Place an output file for img1
        out_file = outputs / "img1.jsonl"
        out_file.write_text('{}\n')

        return tmp_path

    def test_retry_removes_failed_entry(self, tmp_path):
        run_dir = self._make_run(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, [
            "retry-failed", "--run", str(run_dir),
            "--stage", "vlm", "--task", "bboxing",
        ])
        assert result.exit_code == 0
        # failed.jsonl should be empty after retry
        failed = (run_dir / "logs" / "failed.jsonl").read_text().strip()
        assert failed == ""

    def test_retry_removes_output_file(self, tmp_path):
        run_dir = self._make_run(tmp_path)
        out_file = run_dir / "outputs" / "bboxing" / "img1.jsonl"
        assert out_file.exists()
        runner = CliRunner()
        runner.invoke(cli, ["retry-failed", "--run", str(run_dir)])
        assert not out_file.exists()

    def test_retry_removes_done_from_state(self, tmp_path):
        import json as json_mod
        run_dir = self._make_run(tmp_path)
        runner = CliRunner()
        runner.invoke(cli, ["retry-failed", "--run", str(run_dir)])
        state = run_dir / "logs" / "state.jsonl"
        lines = [l for l in state.read_text().splitlines() if l.strip()]
        done_records = [
            json_mod.loads(l) for l in lines
            if json_mod.loads(l).get("event") == "done"
            and json_mod.loads(l).get("image_id") == "img1"
            and json_mod.loads(l).get("task_name") == "bboxing"
        ]
        assert len(done_records) == 0

    def test_no_failed_tasks_message(self, tmp_path):
        (tmp_path / "logs").mkdir()
        runner = CliRunner()
        result = runner.invoke(cli, ["retry-failed", "--run", str(tmp_path)])
        assert "Nothing to retry" in result.output or "No failed" in result.output
