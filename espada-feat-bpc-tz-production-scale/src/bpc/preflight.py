"""Preflight validation of input images (format, readability, RGB, size)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

from .config import InputConfig


logger = logging.getLogger("bpc.preflight")


class PreflightResult(NamedTuple):
    path: Path
    image_id: str
    valid: bool
    reason: str | None


def preflight_image(path: Path, cfg: InputConfig) -> PreflightResult:
    """Checks a single image against the input contract.

    Returns PreflightResult with valid=False and a reason string on failure.
    Does not raise — callers decide what to do with invalid images.
    """
    import cv2

    image_id = path.stem

    # Extension check
    if path.suffix.lower() not in {ext.lower() for ext in cfg.allowed_extensions}:
        return PreflightResult(path, image_id, False, f"Extension '{path.suffix}' not allowed.")

    # Readability
    img = cv2.imread(str(path))
    if img is None:
        return PreflightResult(path, image_id, False, "Image could not be read by cv2.")

    h, w = img.shape[:2]

    # RGB check — verify image has 3 channels (cv2 reads BGR, same channel count)
    if cfg.require_rgb and img.ndim != 3:
        return PreflightResult(path, image_id, False, "Image is not 3-channel (RGB/BGR).")

    # Size check
    if cfg.max_side and max(h, w) > cfg.max_side * 2:
        # Warn but don't reject — prepare-images should handle this
        logger.warning(
            "Image %s is large (%dx%d). Run prepare-images first.", path.name, w, h
        )

    return PreflightResult(path, image_id, True, None)


def run_preflight(
    images: list[Path],
    cfg: InputConfig,
    *,
    preflight_log: Path | None = None,
) -> tuple[list[Path], list[PreflightResult]]:
    """Runs preflight on all images, returns (valid_paths, all_results)."""
    results: list[PreflightResult] = []
    valid: list[Path] = []

    for img_path in images:
        result = preflight_image(img_path, cfg)
        results.append(result)
        if result.valid:
            valid.append(img_path)
        else:
            logger.warning("PREFLIGHT REJECT %s: %s", img_path.name, result.reason)

    if preflight_log:
        import json
        preflight_log.parent.mkdir(parents=True, exist_ok=True)
        with preflight_log.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(
                    json.dumps(
                        {
                            "path": str(r.path),
                            "image_id": r.image_id,
                            "valid": r.valid,
                            "reason": r.reason,
                        }
                    )
                    + "\n"
                )

    logger.info(
        "Preflight: %d total, %d valid, %d rejected",
        len(results),
        len(valid),
        len(results) - len(valid),
    )
    return valid, results
