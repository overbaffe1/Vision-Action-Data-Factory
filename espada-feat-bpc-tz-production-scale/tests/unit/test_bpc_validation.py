"""Tests for src/bpc/validation.py — strict bbox/pointing/caption checks."""
import pytest

from src.bpc.validation import (
    validate_bbox_result,
    validate_pointing_result,
    validate_caption_result,
    ValidationError,
)
from src.bpc.config import (
    BboxValidationConfig,
    PointingValidationConfig,
    CaptionValidationConfig,
)


# ---- helpers ----

def _bbox_cfg(**kwargs):
    defaults = dict(out_of_bounds_policy="reject", min_box_area_px=4, max_boxes_per_image=100)
    defaults.update(kwargs)
    return BboxValidationConfig(**defaults)


def _point_cfg(**kwargs):
    defaults = dict(out_of_bounds_policy="reject", require_point_inside_related_bbox=False,
                    max_points_per_image=100)
    defaults.update(kwargs)
    return PointingValidationConfig(**defaults)


def _cap_cfg(**kwargs):
    defaults = dict(min_length=5, max_length=1000, forbidden_phrases=[])
    defaults.update(kwargs)
    return CaptionValidationConfig(**defaults)


W, H = 640, 480


# =================== BBOX ===================

class TestValidateBbox:
    def _valid(self):
        return {
            "images": ["img.jpg"],
            "answer": [{"label": "car", "bbox_2d": [10, 10, 100, 80]}],
        }

    def test_valid_bbox(self):
        validate_bbox_result(self._valid(), image_width=W, image_height=H, cfg=_bbox_cfg())

    def test_empty_answer_allowed(self):
        result = {"images": ["img.jpg"], "answer": []}
        validate_bbox_result(result, image_width=W, image_height=H, cfg=_bbox_cfg())

    def test_answer_not_list(self):
        with pytest.raises(ValidationError, match="list"):
            validate_bbox_result({"answer": "bad"}, image_width=W, image_height=H, cfg=_bbox_cfg())

    def test_missing_label(self):
        result = {"answer": [{"bbox_2d": [0, 0, 10, 10]}]}
        with pytest.raises(ValidationError, match="label"):
            validate_bbox_result(result, image_width=W, image_height=H, cfg=_bbox_cfg())

    def test_empty_label(self):
        result = {"answer": [{"label": "  ", "bbox_2d": [0, 0, 10, 10]}]}
        with pytest.raises(ValidationError, match="label"):
            validate_bbox_result(result, image_width=W, image_height=H, cfg=_bbox_cfg())

    def test_missing_bbox_2d(self):
        result = {"answer": [{"label": "car"}]}
        with pytest.raises(ValidationError, match="bbox_2d"):
            validate_bbox_result(result, image_width=W, image_height=H, cfg=_bbox_cfg())

    def test_bbox_wrong_length(self):
        result = {"answer": [{"label": "car", "bbox_2d": [0, 0, 10]}]}
        with pytest.raises(ValidationError, match="bbox_2d"):
            validate_bbox_result(result, image_width=W, image_height=H, cfg=_bbox_cfg())

    def test_x1_ge_x2(self):
        result = {"answer": [{"label": "car", "bbox_2d": [100, 10, 50, 80]}]}
        with pytest.raises(ValidationError, match="x1"):
            validate_bbox_result(result, image_width=W, image_height=H, cfg=_bbox_cfg())

    def test_y1_ge_y2(self):
        result = {"answer": [{"label": "car", "bbox_2d": [10, 80, 100, 10]}]}
        with pytest.raises(ValidationError, match="y1"):
            validate_bbox_result(result, image_width=W, image_height=H, cfg=_bbox_cfg())

    def test_area_too_small(self):
        result = {"answer": [{"label": "car", "bbox_2d": [0, 0, 1, 1]}]}
        with pytest.raises(ValidationError, match="area"):
            validate_bbox_result(result, image_width=W, image_height=H, cfg=_bbox_cfg(min_box_area_px=10))

    def test_out_of_bounds_reject(self):
        result = {"answer": [{"label": "car", "bbox_2d": [0, 0, 700, 100]}]}
        with pytest.raises(ValidationError, match="out of image"):
            validate_bbox_result(result, image_width=W, image_height=H, cfg=_bbox_cfg())

    def test_out_of_bounds_clip_policy_passes(self):
        result = {"answer": [{"label": "car", "bbox_2d": [0, 0, 700, 100]}]}
        # clip policy: does not raise for out-of-bounds
        validate_bbox_result(result, image_width=W, image_height=H,
                             cfg=_bbox_cfg(out_of_bounds_policy="clip"))

    def test_nan_coordinate(self):
        import math
        result = {"answer": [{"label": "car", "bbox_2d": [0, 0, float("nan"), 100]}]}
        with pytest.raises(ValidationError, match="NaN"):
            validate_bbox_result(result, image_width=W, image_height=H, cfg=_bbox_cfg())

    def test_inf_coordinate(self):
        result = {"answer": [{"label": "car", "bbox_2d": [0, 0, float("inf"), 100]}]}
        with pytest.raises(ValidationError, match="inf"):
            validate_bbox_result(result, image_width=W, image_height=H, cfg=_bbox_cfg())

    def test_null_coordinate(self):
        result = {"answer": [{"label": "car", "bbox_2d": [0, 0, None, 100]}]}
        with pytest.raises(ValidationError, match="null"):
            validate_bbox_result(result, image_width=W, image_height=H, cfg=_bbox_cfg())

    def test_string_coordinate(self):
        result = {"answer": [{"label": "car", "bbox_2d": ["10", 10, 100, 80]}]}
        with pytest.raises(ValidationError, match="number"):
            validate_bbox_result(result, image_width=W, image_height=H, cfg=_bbox_cfg())

    def test_too_many_boxes(self):
        boxes = [{"label": "obj", "bbox_2d": [i, i, i + 5, i + 5]} for i in range(10)]
        result = {"answer": boxes}
        with pytest.raises(ValidationError, match="Too many"):
            validate_bbox_result(result, image_width=W, image_height=H,
                                cfg=_bbox_cfg(max_boxes_per_image=5))

    def test_confidence_out_of_range(self):
        result = {"answer": [{"label": "car", "bbox_2d": [0, 0, 100, 80], "confidence": 1.5}]}
        with pytest.raises(ValidationError, match="confidence"):
            validate_bbox_result(result, image_width=W, image_height=H, cfg=_bbox_cfg())

    def test_confidence_valid(self):
        result = {"answer": [{"label": "car", "bbox_2d": [0, 0, 100, 80], "confidence": 0.9}]}
        validate_bbox_result(result, image_width=W, image_height=H, cfg=_bbox_cfg())


# =================== POINTING ===================

class TestValidatePointing:
    def _valid(self):
        return {
            "images": ["img.jpg"],
            "answer": [{"label": "wheel", "point_2d": [50, 50]}],
        }

    def test_valid_point(self):
        validate_pointing_result(self._valid(), image_width=W, image_height=H, cfg=_point_cfg())

    def test_empty_answer_allowed(self):
        validate_pointing_result({"answer": []}, image_width=W, image_height=H, cfg=_point_cfg())

    def test_answer_not_list(self):
        with pytest.raises(ValidationError):
            validate_pointing_result({"answer": "bad"}, image_width=W, image_height=H, cfg=_point_cfg())

    def test_missing_label(self):
        with pytest.raises(ValidationError, match="label"):
            validate_pointing_result(
                {"answer": [{"point_2d": [10, 10]}]},
                image_width=W, image_height=H, cfg=_point_cfg()
            )

    def test_missing_point_2d(self):
        with pytest.raises(ValidationError, match="point_2d"):
            validate_pointing_result(
                {"answer": [{"label": "obj"}]},
                image_width=W, image_height=H, cfg=_point_cfg()
            )

    def test_point_wrong_length(self):
        with pytest.raises(ValidationError, match="point_2d"):
            validate_pointing_result(
                {"answer": [{"label": "obj", "point_2d": [10]}]},
                image_width=W, image_height=H, cfg=_point_cfg()
            )

    def test_point_out_of_bounds_reject(self):
        with pytest.raises(ValidationError, match="out of image"):
            validate_pointing_result(
                {"answer": [{"label": "obj", "point_2d": [700, 50]}]},
                image_width=W, image_height=H, cfg=_point_cfg()
            )

    def test_point_out_of_bounds_clip_passes(self):
        validate_pointing_result(
            {"answer": [{"label": "obj", "point_2d": [700, 50]}]},
            image_width=W, image_height=H,
            cfg=_point_cfg(out_of_bounds_policy="clip")
        )

    def test_nan_coord(self):
        with pytest.raises(ValidationError, match="NaN"):
            validate_pointing_result(
                {"answer": [{"label": "obj", "point_2d": [float("nan"), 50]}]},
                image_width=W, image_height=H, cfg=_point_cfg()
            )

    def test_null_coord(self):
        with pytest.raises(ValidationError, match="null"):
            validate_pointing_result(
                {"answer": [{"label": "obj", "point_2d": [None, 50]}]},
                image_width=W, image_height=H, cfg=_point_cfg()
            )

    def test_too_many_points(self):
        points = [{"label": f"obj{i}", "point_2d": [i % W, i % H]} for i in range(10)]
        with pytest.raises(ValidationError, match="Too many"):
            validate_pointing_result(
                {"answer": points}, image_width=W, image_height=H,
                cfg=_point_cfg(max_points_per_image=5)
            )


# =================== CAPTIONS ===================

class TestValidateCaptions:
    def test_valid_caption(self):
        validate_caption_result(
            {"answer": "A street scene with cars and pedestrians."},
            cfg=_cap_cfg()
        )

    def test_empty_caption(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_caption_result({"answer": ""}, cfg=_cap_cfg())

    def test_whitespace_only(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_caption_result({"answer": "   "}, cfg=_cap_cfg())

    def test_not_string(self):
        with pytest.raises(ValidationError, match="string"):
            validate_caption_result({"answer": 42}, cfg=_cap_cfg())

    def test_none_answer(self):
        with pytest.raises(ValidationError, match="missing"):
            validate_caption_result({}, cfg=_cap_cfg())

    def test_too_short(self):
        with pytest.raises(ValidationError, match="short"):
            validate_caption_result({"answer": "Hi"}, cfg=_cap_cfg(min_length=5))

    def test_too_long(self):
        with pytest.raises(ValidationError, match="long"):
            validate_caption_result({"answer": "x" * 200}, cfg=_cap_cfg(max_length=10))

    def test_forbidden_phrase(self):
        with pytest.raises(ValidationError, match="forbidden"):
            validate_caption_result(
                {"answer": "I cannot see the image clearly."},
                cfg=_cap_cfg(forbidden_phrases=["I cannot see the image"])
            )

    def test_forbidden_phrase_case_insensitive(self):
        with pytest.raises(ValidationError, match="forbidden"):
            validate_caption_result(
                {"answer": "As An AI Model I cannot judge this."},
                cfg=_cap_cfg(forbidden_phrases=["as an AI model"])
            )

    def test_markdown_fence(self):
        with pytest.raises(ValidationError, match="markdown"):
            validate_caption_result({"answer": "```json\n{}\n```"}, cfg=_cap_cfg())

    def test_no_forbidden_phrases(self):
        validate_caption_result(
            {"answer": "A sunny day with children playing in the park."},
            cfg=_cap_cfg(forbidden_phrases=["I cannot see the image"])
        )
