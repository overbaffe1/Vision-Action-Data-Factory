"""Tests for src/bpc/logging_setup.py — per-stage file handlers, multi-process suffix."""
import logging
import pytest

from src.bpc.logging_setup import setup_bpc_logging, _STAGE_LOGGERS


def _reset_loggers():
    """Removes handlers added by setup_bpc_logging so tests don't leak state."""
    names = [f"bpc.{n}" if n != "pipeline" else "pipeline" for n in _STAGE_LOGGERS]
    names += ["bpc", "pipeline"]
    for name in names:
        lgr = logging.getLogger(name)
        for h in list(lgr.handlers):
            lgr.removeHandler(h)
            h.close()
    root = logging.getLogger()
    for h in list(root.handlers):
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            root.removeHandler(h)


class TestSetupBpcLogging:
    def teardown_method(self):
        _reset_loggers()

    def test_creates_all_stage_files(self, tmp_path):
        setup_bpc_logging(tmp_path)
        for filename in _STAGE_LOGGERS.values():
            assert (tmp_path / filename).exists(), f"missing {filename}"

    def test_creates_logs_dir_if_missing(self, tmp_path):
        logs_dir = tmp_path / "nested" / "logs"
        setup_bpc_logging(logs_dir)
        assert logs_dir.is_dir()

    def test_writes_to_correct_stage_file(self, tmp_path):
        setup_bpc_logging(tmp_path)
        logging.getLogger("bpc.yolo").info("test message for yolo")
        for h in logging.getLogger("bpc.yolo").handlers:
            h.flush()
        content = (tmp_path / "yolo.log").read_text()
        assert "test message for yolo" in content

    def test_errors_log_captures_warnings_from_any_bpc_logger(self, tmp_path):
        setup_bpc_logging(tmp_path)
        logging.getLogger("bpc.sam2").warning("sam2 warning message")
        for h in logging.getLogger("bpc.sam2").handlers + logging.getLogger("bpc").handlers:
            h.flush()
        content = (tmp_path / "errors.log").read_text()
        assert "sam2 warning message" in content

    def test_errors_log_ignores_info_level(self, tmp_path):
        setup_bpc_logging(tmp_path)
        logging.getLogger("bpc.sam2").info("info level, should not appear in errors.log")
        for h in logging.getLogger("bpc.sam2").handlers + logging.getLogger("bpc").handlers:
            h.flush()
        content = (tmp_path / "errors.log").read_text()
        assert "should not appear" not in content


class TestFileSuffix:
    """Multi-GPU: each spawned process gets its own log files via file_suffix."""

    def teardown_method(self):
        _reset_loggers()

    def test_suffix_creates_distinct_files(self, tmp_path):
        setup_bpc_logging(tmp_path, file_suffix=".gpu0")
        assert (tmp_path / "yolo.gpu0.log").exists()
        assert not (tmp_path / "yolo.log").exists()

    def test_no_suffix_uses_plain_names(self, tmp_path):
        setup_bpc_logging(tmp_path, file_suffix="")
        assert (tmp_path / "yolo.log").exists()

    def test_errors_log_also_suffixed(self, tmp_path):
        setup_bpc_logging(tmp_path, file_suffix=".gpu1")
        assert (tmp_path / "errors.gpu1.log").exists()

    def test_different_suffixes_dont_collide(self, tmp_path):
        setup_bpc_logging(tmp_path, file_suffix=".gpu0")
        _reset_loggers()
        setup_bpc_logging(tmp_path, file_suffix=".gpu1")
        for filename in _STAGE_LOGGERS.values():
            stem, _, ext = filename.rpartition(".")
            assert (tmp_path / f"{stem}.gpu0.{ext}").exists()
            assert (tmp_path / f"{stem}.gpu1.{ext}").exists()

    def test_all_stage_files_get_suffix(self, tmp_path):
        setup_bpc_logging(tmp_path, file_suffix=".gpu2")
        for filename in _STAGE_LOGGERS.values():
            stem, _, ext = filename.rpartition(".")
            assert (tmp_path / f"{stem}.gpu2.{ext}").exists(), f"missing suffixed {filename}"


class TestNoDuplicateHandlersOnReentry:
    """Calling setup_bpc_logging() twice for the same logs_dir/suffix must not
    duplicate FileHandlers — each log line would otherwise be written N times
    and the old file handles would leak."""

    def teardown_method(self):
        _reset_loggers()

    def test_second_call_does_not_duplicate_yolo_handler(self, tmp_path):
        setup_bpc_logging(tmp_path)
        setup_bpc_logging(tmp_path)
        yolo_logger = logging.getLogger("bpc.yolo")
        file_handlers = [h for h in yolo_logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1

    def test_second_call_does_not_duplicate_errors_handler(self, tmp_path):
        setup_bpc_logging(tmp_path)
        setup_bpc_logging(tmp_path)
        bpc_logger = logging.getLogger("bpc")
        file_handlers = [h for h in bpc_logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1

    def test_log_line_written_exactly_once_after_double_setup(self, tmp_path):
        setup_bpc_logging(tmp_path)
        setup_bpc_logging(tmp_path)
        logging.getLogger("bpc.yolo").info("only once please")
        for h in logging.getLogger("bpc.yolo").handlers:
            h.flush()
        content = (tmp_path / "yolo.log").read_text()
        assert content.count("only once please") == 1

    def test_different_suffix_still_gets_its_own_handler(self, tmp_path):
        # Re-calling with a DIFFERENT suffix is a different target file, so
        # it must add a new handler rather than being deduped away.
        setup_bpc_logging(tmp_path, file_suffix=".gpu0")
        setup_bpc_logging(tmp_path, file_suffix=".gpu1")
        yolo_logger = logging.getLogger("bpc.yolo")
        file_handlers = [h for h in yolo_logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 2
