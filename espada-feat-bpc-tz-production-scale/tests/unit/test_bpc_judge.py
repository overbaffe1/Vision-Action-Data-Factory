"""Tests for src/bpc/judge.py — VLM Judge coordinate immutability checks."""
import pytest

from src.bpc.judge import (
    verify_judge_output,
    CoordinateChangedError,
    _coords_equal,
    _verify_bbox_coords_unchanged,
    _verify_pointing_coords_unchanged,
)


# ---- _coords_equal ----

class TestCoordsEqual:
    def test_int_equal(self):
        assert _coords_equal(10, 10) is True

    def test_int_not_equal(self):
        assert _coords_equal(10, 11) is False

    def test_float_exact(self):
        assert _coords_equal(10.0, 10.0) is True

    def test_int_float_equal(self):
        assert _coords_equal(10, 10.0) is True

    def test_int_float_not_equal(self):
        assert _coords_equal(10, 10.1) is False

    def test_zero(self):
        assert _coords_equal(0, 0) is True

    def test_negative(self):
        assert _coords_equal(-5, -5) is True


# ---- bbox coordinate verification ----

class TestVerifyBboxCoords:
    def _orig(self):
        return [{"label": "car", "bbox_2d": [10, 20, 100, 200]}]

    def test_unchanged_ok(self):
        _verify_bbox_coords_unchanged(self._orig(), self._orig())

    def test_changed_x1(self):
        judged = [{"label": "vehicle", "bbox_2d": [11, 20, 100, 200]}]
        with pytest.raises(CoordinateChangedError, match="bbox_2d"):
            _verify_bbox_coords_unchanged(self._orig(), judged)

    def test_changed_y2(self):
        judged = [{"label": "car", "bbox_2d": [10, 20, 100, 201]}]
        with pytest.raises(CoordinateChangedError, match="bbox_2d"):
            _verify_bbox_coords_unchanged(self._orig(), judged)

    def test_different_count(self):
        judged = [{"label": "car", "bbox_2d": [10, 20, 100, 200]},
                  {"label": "truck", "bbox_2d": [5, 5, 50, 50]}]
        with pytest.raises(CoordinateChangedError, match="number of objects"):
            _verify_bbox_coords_unchanged(self._orig(), judged)

    def test_float_equiv_ok(self):
        judged = [{"label": "car", "bbox_2d": [10.0, 20.0, 100.0, 200.0]}]
        _verify_bbox_coords_unchanged(self._orig(), judged)

    def test_multiple_boxes_changed_second(self):
        orig = [
            {"label": "car", "bbox_2d": [10, 20, 100, 200]},
            {"label": "bike", "bbox_2d": [5, 5, 50, 50]},
        ]
        judged = [
            {"label": "vehicle", "bbox_2d": [10, 20, 100, 200]},
            {"label": "bicycle", "bbox_2d": [5, 5, 51, 50]},  # x2 changed
        ]
        with pytest.raises(CoordinateChangedError):
            _verify_bbox_coords_unchanged(orig, judged)


# ---- pointing coordinate verification ----

class TestVerifyPointingCoords:
    def _orig(self):
        return [{"label": "wheel", "point_2d": [50, 60]}]

    def test_unchanged_ok(self):
        _verify_pointing_coords_unchanged(self._orig(), self._orig())

    def test_changed_x(self):
        judged = [{"label": "tyre", "point_2d": [51, 60]}]
        with pytest.raises(CoordinateChangedError, match="point_2d"):
            _verify_pointing_coords_unchanged(self._orig(), judged)

    def test_changed_y(self):
        judged = [{"label": "wheel", "point_2d": [50, 61]}]
        with pytest.raises(CoordinateChangedError):
            _verify_pointing_coords_unchanged(self._orig(), judged)

    def test_different_count(self):
        judged = [{"label": "wheel", "point_2d": [50, 60]},
                  {"label": "extra", "point_2d": [10, 10]}]
        with pytest.raises(CoordinateChangedError, match="number of points"):
            _verify_pointing_coords_unchanged(self._orig(), judged)


# ---- verify_judge_output ----

class TestVerifyJudgeOutput:
    def test_bbox_unchanged(self):
        orig = {"answer": [{"label": "car", "bbox_2d": [0, 0, 100, 100]}]}
        judged = {"answer": [{"label": "vehicle", "bbox_2d": [0, 0, 100, 100]}]}
        verify_judge_output(orig, judged, "bboxing")

    def test_bbox_changed_raises(self):
        orig = {"answer": [{"label": "car", "bbox_2d": [0, 0, 100, 100]}]}
        judged = {"answer": [{"label": "vehicle", "bbox_2d": [1, 0, 100, 100]}]}
        with pytest.raises(CoordinateChangedError):
            verify_judge_output(orig, judged, "bboxing")

    def test_pointing_unchanged(self):
        orig = {"answer": [{"label": "wheel", "point_2d": [50, 50]}]}
        judged = {"answer": [{"label": "tyre", "point_2d": [50, 50]}]}
        verify_judge_output(orig, judged, "pointing")

    def test_pointing_changed_raises(self):
        orig = {"answer": [{"label": "wheel", "point_2d": [50, 50]}]}
        judged = {"answer": [{"label": "tyre", "point_2d": [51, 50]}]}
        with pytest.raises(CoordinateChangedError):
            verify_judge_output(orig, judged, "pointing")

    def test_unsupported_task(self):
        with pytest.raises(ValueError, match="not supported"):
            verify_judge_output({"answer": []}, {"answer": []}, "captions")

    def test_answer_not_list_raises(self):
        with pytest.raises(CoordinateChangedError):
            verify_judge_output({"answer": "bad"}, {"answer": []}, "bboxing")

    def test_empty_answers_ok(self):
        verify_judge_output({"answer": []}, {"answer": []}, "bboxing")
        verify_judge_output({"answer": []}, {"answer": []}, "pointing")
