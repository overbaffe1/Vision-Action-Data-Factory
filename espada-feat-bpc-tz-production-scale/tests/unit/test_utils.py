"""Unit tests for src/utils.py.

The whole module imports cv2 at import time, so the entire file is skipped
when opencv is unavailable (minimal environment). Pandas is skipped
separately only for the detection-DataFrame test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python not installed")

from src import utils  # noqa: E402  (import after importorskip on purpose)

pytestmark = [pytest.mark.unit, pytest.mark.needs_cv2]


# --- bbox geometry helpers -------------------------------------------------


def test_compute_bbox_area():
    assert utils._compute_bbox_area([0, 0, 10, 5]) == 50


def test_compute_image_area():
    assert utils._compute_image_area(4, 3) == 12


def test_format_bbox_dict():
    dets = {"id": [1, 2], "bbox": [[1.0, 2.0, 3.0, 4.0], [5, 6, 7, 8]]}
    assert utils.format_bbox_dict(dets) == {1: [1, 2, 3, 4], 2: [5, 6, 7, 8]}


def test_filter_big_bboxes_removes_oversized(bbox_detections):
    # image 60x60 = 3600 area; threshold 40% -> max 1440.
    # bbox [5,5,58,60] -> 53*55 = 2915 (removed); [10,10,30,40] -> 20*30 = 600 (kept).
    out = utils.filter_big_bboxes((60, 60), bbox_detections, threshold=40)
    assert out["id"] == [1]
    assert out["bbox"] == [[10, 10, 30, 40]]
    assert out["class"] == ["person"]


def test_filter_big_bboxes_keeps_all_when_threshold_high(bbox_detections):
    out = utils.filter_big_bboxes((60, 60), bbox_detections, threshold=100)
    assert out["id"] == [1, 2]


def test_filter_big_bboxes_empty_input():
    empty = {"id": [], "bbox": [], "class": [], "conf": []}
    out = utils.filter_big_bboxes((100, 100), empty, threshold=40)
    assert out == empty


# --- grounding JSON extraction / building ----------------------------------


def test_extract_grounding_json_ok():
    raw = 'noise before {"grounding_answer": [], "grounding_question": "q"} trailing'
    parsed = utils.extract_grounding_json(raw)
    assert parsed["grounding_question"] == "q"


def test_extract_grounding_json_no_braces():
    with pytest.raises(ValueError, match="JSON block not found"):
        utils.extract_grounding_json("no json here")


def test_build_detection_result_ok():
    raw = '{"grounding_question": "Locate x", "grounding_answer": [{"label": "x", "bbox_2d": [1,2,3,4]}]}'
    result = utils.build_detection_result(raw, "/data/img.jpg")
    assert result["images"] == ["/data/img.jpg"]
    assert result["question"] == "Locate x"
    assert result["answer"] == [{"label": "x", "bbox_2d": [1, 2, 3, 4]}]


def test_build_detection_result_missing_fields():
    raw = '{"grounding_question": "q"}'  # no grounding_answer
    with pytest.raises(ValueError, match="Invalid grounding JSON"):
        utils.build_detection_result(raw, "img.jpg")


def test_save_detection_json_writes_llama_factory_jsonl(tmp_path):
    result = {"images": ["x.jpg"], "question": "q", "answer": "a"}
    out = utils.save_detection_json(result, "some/dir/frame_007.png", tmp_path / "bboxing")
    assert out.name == "frame_007.jsonl"
    assert out.exists()
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["messages"] == [
        {"role": "user", "content": "<image>\nq"},
        {"role": "assistant", "content": "a"},
    ]
    assert record["images"] == [str(Path("some/dir/frame_007.png").resolve())]


# --- image collection / resize (Pillow-based, no cv2 needed at runtime) -----


def test_collect_prop_frames(raw_images_dir):
    frames = utils.collect_prop_frames(raw_images_dir)
    names = sorted(p.name for p in frames)
    assert names == ["frame_000.jpg", "frame_001.png", "frame_002.jpg"]


def test_collect_prop_frames_empty_dir(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        utils.collect_prop_frames(empty)


def test_collect_prop_frames_not_a_dir(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hi")
    with pytest.raises(NotADirectoryError):
        utils.collect_prop_frames(f)


def test_resize_single_image(single_image_file):
    img = utils.resize_single_image(single_image_file, (32, 16))
    assert img.size == (32, 16)  # PIL (W, H)
    assert img.mode == "RGB"


def test_resize_single_image_missing():
    with pytest.raises(FileExistsError):
        utils.resize_single_image("does/not/exist.jpg")


def test_save_resized_images_dir(raw_images_dir, tmp_path):
    dst = tmp_path / "processed"
    saved = utils.save_resized_images_dir(raw_images_dir, dst, target_size=(20, 10))
    assert len(saved) == 3
    assert all(p.suffix == ".jpg" for p in saved)
    assert all(p.exists() for p in saved)


def test_save_resized_images_dir_skips_existing(raw_images_dir, tmp_path):
    dst = tmp_path / "processed"
    utils.save_resized_images_dir(raw_images_dir, dst, target_size=(20, 10))
    mtimes = {p.name: p.stat().st_mtime_ns for p in dst.iterdir()}
    # Second run must skip already-processed files (no rewrite).
    saved2 = utils.save_resized_images_dir(raw_images_dir, dst, target_size=(20, 10))
    assert len(saved2) == 3
    for p in dst.iterdir():
        assert p.stat().st_mtime_ns == mtimes[p.name]


# --- centroid / mask drawing (needs cv2 at runtime) ------------------------


def test_mask_inner_point_on_square():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    pt = utils._mask_inner_point(mask)
    assert pt is not None
    x, y = pt
    # Inner point should be well inside the square block.
    assert 14 <= x <= 26
    assert 14 <= y <= 26


def test_mask_inner_point_empty_mask():
    assert utils._mask_inner_point(np.zeros((10, 10), dtype=np.uint8)) is None


def test_remove_padding_from_point():
    assert utils.remove_padding_from_point((15, 20), (8, 8)) == (7, 12)


def test_normalize_masks_shapes():
    two_d = np.ones((10, 10))
    assert utils._normalize_masks(two_d).shape == (1, 10, 10)
    three_d = np.ones((3, 10, 10))
    assert utils._normalize_masks(three_d).shape == (3, 10, 10)


def test_draw_sam2_masks_with_centroids(synthetic_rgb):
    h, w = synthetic_rgb.shape[:2]
    masks = np.zeros((2, h, w), dtype=bool)
    masks[0, 5:20, 5:20] = True
    masks[1, 30:50, 40:80] = True
    out, meta = utils.draw_sam2_masks_with_centroids(synthetic_rgb, masks)
    assert out.shape == synthetic_rgb.shape
    assert meta["id"] == [1, 2]
    assert len(meta["points"]) == 2


def test_draw_sam2_masks_size_mismatch(synthetic_rgb):
    bad = np.zeros((1, 5, 5), dtype=bool)
    with pytest.raises(ValueError, match="Mask size"):
        utils.draw_sam2_masks_with_centroids(synthetic_rgb, bad)


def test_draw_detections_cv2_returns_copy(synthetic_rgb, bbox_detections):
    out = utils.draw_detections_cv2(synthetic_rgb, bbox_detections)
    assert out.shape == synthetic_rgb.shape
    assert out is not synthetic_rgb  # must not mutate input


# --- _try_save_vlm_json (integration of build + validate + save) -----------


def test_try_save_vlm_json_bboxing_ok(tmp_path):
    raw = '{"grounding_question": "q", "grounding_answer": [{"label": "x", "bbox_2d": [1,2,3,4]}]}'
    out_dir = tmp_path / "bboxing"
    out = utils._try_save_vlm_json(raw, frame_path="f.jpg", out_dir=out_dir)
    assert out is not None and out.exists()


def test_try_save_vlm_json_invalid_returns_none(tmp_path):
    # bbox with x_min >= x_max fails validation -> None, no file written.
    raw = '{"grounding_question": "q", "grounding_answer": [{"label": "x", "bbox_2d": [9,2,3,4]}]}'
    out_dir = tmp_path / "bboxing"
    out = utils._try_save_vlm_json(raw, frame_path="f.jpg", out_dir=out_dir)
    assert out is None
    assert not (out_dir / "f.jsonl").exists()


def test_try_save_vlm_json_unknown_task_returns_none(tmp_path):
    raw = '{"grounding_question": "q", "grounding_answer": "text"}'
    out = utils._try_save_vlm_json(raw, frame_path="f.jpg", out_dir=tmp_path / "weird")
    assert out is None


# --- pandas-dependent: build_extended_detections ---------------------------


def test_build_extended_detections():
    pd = pytest.importorskip("pandas", reason="pandas not installed")
    df = pd.DataFrame(
        [
            {"class": "person", "conf": 0.912345, "x1": 1.6, "y1": 2.2, "x2": 3.9, "y2": 4.1},
            {"class": "cup", "conf": 0.5, "x1": 5, "y1": 6, "x2": 7, "y2": 8},
        ]
    )
    out = utils.build_extended_detections(df)
    assert out["id"] == [1, 2]  # idx + 1
    assert out["bbox"] == [[1, 2, 3, 4], [5, 6, 7, 8]]  # int-cast
    assert out["class"] == ["person", "cup"]
    assert out["conf"] == [0.912, 0.5]  # rounded to 3
