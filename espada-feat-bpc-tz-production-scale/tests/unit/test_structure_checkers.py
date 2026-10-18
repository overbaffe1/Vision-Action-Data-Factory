"""Unit tests for VLM output JSON validators.

These validators are pure-logic and have no heavy dependencies, so they form
the fast, always-runnable core of the suite.
"""

from __future__ import annotations

import pytest

from src.structure_checkers import (
    _check_bbox_json_structure,
    _check_captioning_json_structure,
    _check_pointing_json_structure,
    _is_images_paths_empty,
)

pytestmark = pytest.mark.unit


# --- shared: images list ---------------------------------------------------


def test_empty_images_raises():
    with pytest.raises(ValueError, match="No images"):
        _is_images_paths_empty({"images": []})


def test_nonempty_images_passes():
    _is_images_paths_empty({"images": ["a.jpg"]})  # no raise


# --- bbox ------------------------------------------------------------------


def _bbox_markup(answer):
    return {"images": ["img.jpg"], "question": "q", "answer": answer}


def test_bbox_valid():
    markup = _bbox_markup([{"label": "person", "bbox_2d": [10, 20, 30, 40]}])
    _check_bbox_json_structure(markup)  # no raise


def test_bbox_empty_images():
    markup = {"images": [], "answer": [{"label": "x", "bbox_2d": [1, 2, 3, 4]}]}
    with pytest.raises(ValueError):
        _check_bbox_json_structure(markup)


def test_bbox_item_not_dict():
    with pytest.raises(TypeError):
        _check_bbox_json_structure(_bbox_markup(["not-a-dict"]))


def test_bbox_missing_label():
    with pytest.raises(KeyError):
        _check_bbox_json_structure(_bbox_markup([{"bbox_2d": [1, 2, 3, 4]}]))


def test_bbox_empty_label():
    with pytest.raises(ValueError):
        _check_bbox_json_structure(_bbox_markup([{"label": "", "bbox_2d": [1, 2, 3, 4]}]))


def test_bbox_missing_bbox():
    with pytest.raises(KeyError):
        _check_bbox_json_structure(_bbox_markup([{"label": "x"}]))


def test_bbox_wrong_length():
    with pytest.raises(ValueError, match="4 elements"):
        _check_bbox_json_structure(_bbox_markup([{"label": "x", "bbox_2d": [1, 2, 3]}]))


def test_bbox_non_int_coord():
    with pytest.raises(TypeError):
        _check_bbox_json_structure(_bbox_markup([{"label": "x", "bbox_2d": [1, 2, 3, 4.5]}]))


@pytest.mark.parametrize(
    "bbox",
    [
        [30, 10, 10, 40],  # x_min >= x_max
        [10, 40, 30, 10],  # y_min >= y_max
        [10, 10, 10, 40],  # x_min == x_max
    ],
)
def test_bbox_invalid_ordering(bbox):
    with pytest.raises(ValueError, match="less than"):
        _check_bbox_json_structure(_bbox_markup([{"label": "x", "bbox_2d": bbox}]))


def test_bbox_negative_coords():
    with pytest.raises(ValueError, match="non-negative"):
        _check_bbox_json_structure(_bbox_markup([{"label": "x", "bbox_2d": [-1, 0, 5, 5]}]))


def test_bbox_empty_answer_is_allowed():
    # No items -> loop body never runs -> valid.
    _check_bbox_json_structure(_bbox_markup([]))


# --- pointing --------------------------------------------------------------


def _point_markup(answer):
    return {"images": ["img.jpg"], "question": "q", "answer": answer}


def test_pointing_valid():
    _check_pointing_json_structure(_point_markup([{"label": "person", "point_2d": [12, 34]}]))


def test_pointing_label_not_string():
    with pytest.raises(TypeError):
        _check_pointing_json_structure(_point_markup([{"label": 5, "point_2d": [1, 2]}]))


def test_pointing_empty_label():
    with pytest.raises(ValueError):
        _check_pointing_json_structure(_point_markup([{"label": "", "point_2d": [1, 2]}]))


def test_pointing_point_not_list():
    with pytest.raises(TypeError):
        _check_pointing_json_structure(_point_markup([{"label": "x", "point_2d": "1,2"}]))


def test_pointing_wrong_length():
    with pytest.raises(ValueError, match="2 elements"):
        _check_pointing_json_structure(_point_markup([{"label": "x", "point_2d": [1, 2, 3]}]))


def test_pointing_non_int_coord():
    with pytest.raises(TypeError):
        _check_pointing_json_structure(_point_markup([{"label": "x", "point_2d": [1, 2.0]}]))


def test_pointing_negative():
    with pytest.raises(ValueError, match="non-negative"):
        _check_pointing_json_structure(_point_markup([{"label": "x", "point_2d": [-1, 2]}]))


# --- captioning ------------------------------------------------------------


def test_captioning_valid():
    _check_captioning_json_structure(
        {"images": ["img.jpg"], "question": "q", "answer": "A clear description."}
    )


def test_captioning_not_string():
    with pytest.raises(TypeError):
        _check_captioning_json_structure({"images": ["img.jpg"], "answer": ["list"]})


def test_captioning_blank():
    with pytest.raises(ValueError):
        _check_captioning_json_structure({"images": ["img.jpg"], "answer": "   "})


def test_captioning_empty_images():
    with pytest.raises(ValueError):
        _check_captioning_json_structure({"images": [], "answer": "text"})
