"""Strict technical validation for BPC pipeline outputs (bbox, pointing, captions)."""
from __future__ import annotations

import math
from typing import Any

from .config import BboxValidationConfig, PointingValidationConfig, CaptionValidationConfig


class ValidationError(ValueError):
    """Raised when a VLM result fails technical validation."""


def _check_coord(value: Any, name: str) -> float:
    """Ensures coordinate is a finite number (not NaN, inf, null, string)."""
    if value is None:
        raise ValidationError(f"Coordinate '{name}' is null.")
    if not isinstance(value, (int, float)):
        raise ValidationError(f"Coordinate '{name}' must be a number, got {type(value).__name__}.")
    if math.isnan(value) or math.isinf(value):
        raise ValidationError(f"Coordinate '{name}' is NaN or inf.")
    return float(value)


def validate_bbox_result(
    result: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
    cfg: BboxValidationConfig,
) -> None:
    """Validates bounding box VLM result against image dimensions and config.

    Raises ValidationError with a descriptive message on any failure.
    """
    answer = result.get("answer")
    if not isinstance(answer, list):
        raise ValidationError("'answer' must be a list of bbox objects.")

    if len(answer) == 0:
        # empty list is valid (no objects detected by VLM)
        return

    if len(answer) > cfg.max_boxes_per_image:
        raise ValidationError(
            f"Too many boxes: {len(answer)} > max {cfg.max_boxes_per_image}."
        )

    for i, obj in enumerate(answer):
        if not isinstance(obj, dict):
            raise ValidationError(f"Annotation [{i}] must be a dict.")

        label = obj.get("label")
        if not label or not isinstance(label, str) or not label.strip():
            raise ValidationError(f"Annotation [{i}] has empty or missing 'label'.")

        bbox = obj.get("bbox_2d")
        if bbox is None:
            raise ValidationError(f"Annotation [{i}] missing 'bbox_2d'.")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValidationError(f"Annotation [{i}] 'bbox_2d' must be [x1,y1,x2,y2].")

        x1 = _check_coord(bbox[0], f"[{i}].bbox_2d[0] x1")
        y1 = _check_coord(bbox[1], f"[{i}].bbox_2d[1] y1")
        x2 = _check_coord(bbox[2], f"[{i}].bbox_2d[2] x2")
        y2 = _check_coord(bbox[3], f"[{i}].bbox_2d[3] y2")

        if x1 >= x2:
            raise ValidationError(f"Annotation [{i}] x1={x1} >= x2={x2}.")
        if y1 >= y2:
            raise ValidationError(f"Annotation [{i}] y1={y1} >= y2={y2}.")

        area = (x2 - x1) * (y2 - y1)
        if area < cfg.min_box_area_px:
            raise ValidationError(
                f"Annotation [{i}] box area {area:.1f} < min {cfg.min_box_area_px}."
            )

        if cfg.out_of_bounds_policy == "reject":
            if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
                raise ValidationError(
                    f"Annotation [{i}] bbox [{x1},{y1},{x2},{y2}] out of image "
                    f"{image_width}x{image_height}."
                )

        conf = obj.get("confidence")
        if conf is not None:
            c = _check_coord(conf, f"[{i}].confidence")
            if not (0.0 <= c <= 1.0):
                raise ValidationError(f"Annotation [{i}] confidence {c} not in [0,1].")


def validate_pointing_result(
    result: dict[str, Any],
    *,
    image_width: int,
    image_height: int,
    cfg: PointingValidationConfig,
) -> None:
    """Validates pointing VLM result against image dimensions and config."""
    answer = result.get("answer")
    if not isinstance(answer, list):
        raise ValidationError("'answer' must be a list of point objects.")

    if len(answer) == 0:
        return

    if len(answer) > cfg.max_points_per_image:
        raise ValidationError(
            f"Too many points: {len(answer)} > max {cfg.max_points_per_image}."
        )

    for i, obj in enumerate(answer):
        if not isinstance(obj, dict):
            raise ValidationError(f"Point [{i}] must be a dict.")

        label = obj.get("label")
        if not label or not isinstance(label, str) or not label.strip():
            raise ValidationError(f"Point [{i}] has empty or missing 'label'.")

        point = obj.get("point_2d")
        if point is None:
            raise ValidationError(f"Point [{i}] missing 'point_2d'.")
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValidationError(f"Point [{i}] 'point_2d' must be [x, y].")

        x = _check_coord(point[0], f"[{i}].point_2d[0] x")
        y = _check_coord(point[1], f"[{i}].point_2d[1] y")

        if cfg.out_of_bounds_policy == "reject":
            if x < 0 or y < 0 or x >= image_width or y >= image_height:
                raise ValidationError(
                    f"Point [{i}] [{x},{y}] out of image {image_width}x{image_height}."
                )

        conf = obj.get("confidence")
        if conf is not None:
            c = _check_coord(conf, f"[{i}].confidence")
            if not (0.0 <= c <= 1.0):
                raise ValidationError(f"Point [{i}] confidence {c} not in [0,1].")


def validate_caption_result(
    result: dict[str, Any],
    *,
    cfg: CaptionValidationConfig,
) -> None:
    """Validates captioning VLM result (basic technical checks only)."""
    answer = result.get("answer")
    if answer is None:
        raise ValidationError("'answer' is missing.")
    if not isinstance(answer, str):
        raise ValidationError(f"'answer' must be a string, got {type(answer).__name__}.")

    text = answer.strip()
    if not text:
        raise ValidationError("'answer' is empty.")

    if len(text) < cfg.min_length:
        raise ValidationError(
            f"Caption too short: {len(text)} < min {cfg.min_length}."
        )
    if len(text) > cfg.max_length:
        raise ValidationError(
            f"Caption too long: {len(text)} > max {cfg.max_length}."
        )

    lower = text.lower()
    for phrase in cfg.forbidden_phrases:
        if phrase.lower() in lower:
            raise ValidationError(f"Caption contains forbidden phrase: '{phrase}'.")

    # Detect unclosed markdown/JSON artifacts
    if text.startswith("```") or text.endswith("```"):
        raise ValidationError("Caption contains markdown code fences.")
