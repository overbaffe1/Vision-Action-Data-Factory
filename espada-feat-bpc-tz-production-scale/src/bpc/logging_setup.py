"""Structured per-stage logging setup for BPC pipeline."""
from __future__ import annotations

import logging
from pathlib import Path


_STAGE_LOGGERS = {
    "pipeline": "pipeline.log",
    "preflight": "preflight.log",
    "yolo": "yolo.log",
    "sam2": "sam2.log",
    "vlm": "vlm.log",
    "validation": "validation.log",
    "judge": "judge.log",
    "errors": "errors.log",
}

_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_bpc_logging(
    logs_dir: Path,
    *,
    console_level: str = "INFO",
    file_level: str = "DEBUG",
    file_suffix: str = "",
) -> None:
    """Configures per-stage file handlers and a console handler.

    ``file_suffix`` (e.g. ``".gpu0"``) is inserted before the extension so
    each spawned multi-GPU process gets its own log files (yolo.gpu0.log,
    yolo.gpu1.log, ...) instead of contending for one shared file handle —
    unlike StateLogger's raw append("a") writes, logging.FileHandler does
    not guarantee atomic interleaving of concurrent writers across processes.
    Leave empty for the single-process path (yolo.log, ...).
    """
    logs_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_FMT)

    # Root / console
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, console_level))
        ch.setFormatter(formatter)
        root.addHandler(ch)

    def _with_suffix(filename: str) -> str:
        if not file_suffix:
            return filename
        stem, _, ext = filename.rpartition(".")
        return f"{stem}{file_suffix}.{ext}"

    def _add_file_handler_once(lgr: logging.Logger, target_path: Path, level: int) -> None:
        # If setup_bpc_logging() is ever called twice in the same process
        # (e.g. a future retry loop, or test harness reentry — currently
        # each multi-GPU worker calls it exactly once, so this is latent
        # rather than actively triggered), adding a handler unconditionally
        # would duplicate it on the logger: every log line gets written N
        # times and the old file handle leaks. Match by resolved target path
        # so re-calling with the same logs_dir/suffix is a no-op.
        for h in lgr.handlers:
            if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == target_path:
                return
        fh = logging.FileHandler(target_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        lgr.addHandler(fh)

    # Per-stage file handlers
    for logger_name, filename in _STAGE_LOGGERS.items():
        lgr = logging.getLogger(f"bpc.{logger_name}" if logger_name != "pipeline" else "pipeline")
        _add_file_handler_once(lgr, logs_dir / _with_suffix(filename), getattr(logging, file_level))

    # errors.log also captures WARNING+ from all bpc loggers
    errors_path = logs_dir / _with_suffix("errors.log")
    _add_file_handler_once(logging.getLogger("bpc"), errors_path, logging.WARNING)
    _add_file_handler_once(logging.getLogger("pipeline"), errors_path, logging.WARNING)
