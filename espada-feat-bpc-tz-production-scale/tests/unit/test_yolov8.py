"""Unit tests for the YoloV8 wrapper.

We never load real weights: ``ultralytics.YOLO`` is patched with a fake so the
constructor and the result->DataFrame conversion can be tested in isolation.
Requires torch + pandas (skipped otherwise).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# torch is stubbed in the root conftest when not installed; pandas is real
# (DataFrame results are asserted), so require it.
pytest.importorskip("pandas", reason="pandas not installed")

from src.yolo.yolov8 import YoloV8  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.needs_torch]


def _fake_box(cls_id, conf, xyxy):
    b = MagicMock()
    b.cls = [cls_id]
    b.conf = [conf]
    b.xyxy = [xyxy]
    return b


def _fake_result(boxes):
    r = MagicMock()
    r.boxes = boxes
    return r


@patch("src.yolo.yolov8.YOLO")
def test_init_invalid_batch_size(mock_yolo):
    with pytest.raises(ValueError, match="batch_size"):
        YoloV8(batch_size=0)


@patch("src.yolo.yolov8.YOLO")
def test_select_device_cpu_when_no_cuda(mock_yolo):
    with patch("src.yolo.yolov8.torch.cuda.is_available", return_value=False):
        det = YoloV8(device="cuda")
    assert det.device == "cpu"


@patch("src.yolo.yolov8.YOLO")
def test_select_device_explicit_index_passthrough(mock_yolo):
    det = YoloV8(device="cuda:1")
    assert det.device == "cuda:1"


@patch("src.yolo.yolov8.YOLO")
def test_result_to_dataframe_empty(mock_yolo):
    det = YoloV8(device="cpu")
    df = det._result_to_dataframe(_fake_result([]))
    assert list(df.columns) == ["class", "class_id", "conf", "x1", "y1", "x2", "y2"]
    assert len(df) == 0


@patch("src.yolo.yolov8.YOLO")
def test_result_to_dataframe_rows(mock_yolo):
    instance = mock_yolo.return_value
    instance.names = {0: "person", 56: "chair"}
    det = YoloV8(device="cpu")
    det.model.names = {0: "person", 56: "chair"}

    result = _fake_result(
        [
            _fake_box(0, 0.9, [10.0, 20.0, 30.0, 40.0]),
            _fake_box(56, 0.5, [1.0, 2.0, 3.0, 4.0]),
        ]
    )
    df = det._result_to_dataframe(result)
    assert len(df) == 2
    assert df.iloc[0]["class"] == "person"
    assert df.iloc[0]["class_id"] == 0
    assert df.iloc[0]["x1"] == 10.0
    assert df.iloc[1]["class"] == "chair"


@patch("src.yolo.yolov8.YOLO")
def test_predict_calls_model_with_single_image(mock_yolo):
    instance = mock_yolo.return_value
    instance.names = {0: "person"}
    instance.predict.return_value = [_fake_result([_fake_box(0, 0.8, [0, 0, 5, 5])])]

    det = YoloV8(device="cpu")
    det.model.names = {0: "person"}
    import numpy as np

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    df = det.predict(img)
    assert len(df) == 1
    # model.predict invoked once with a single-image source list
    assert instance.predict.call_count == 1
    _, kwargs = instance.predict.call_args
    assert kwargs["device"] == "cpu"


def test_iter_chunks():
    chunks = list(YoloV8._iter_chunks([1, 2, 3, 4, 5], 2))
    assert chunks == [[1, 2], [3, 4], [5]]


@patch("src.yolo.yolov8.YOLO")
def test_predict_on_batch_empty(mock_yolo):
    det = YoloV8(device="cpu")
    assert det.predict_on_batch([]) == []


@patch("src.yolo.yolov8.YOLO")
def test_empty_cache_skipped_on_cpu(mock_yolo):
    det = YoloV8(device="cpu", empty_cache=True)
    with patch("src.yolo.yolov8.torch.cuda.empty_cache") as ec, patch(
        "src.yolo.yolov8.torch.cuda.is_available", return_value=True
    ):
        det._maybe_empty_cache()
    ec.assert_not_called()  # device is cpu -> no cache clearing


@patch("src.yolo.yolov8.YOLO")
def test_empty_cache_called_on_cuda(mock_yolo):
    det = YoloV8(device="cuda:0", empty_cache=True)
    with patch("src.yolo.yolov8.torch.cuda.empty_cache") as ec, patch(
        "src.yolo.yolov8.torch.cuda.is_available", return_value=True
    ):
        det._maybe_empty_cache()
    ec.assert_called_once()


@patch("src.yolo.yolov8.YOLO")
def test_empty_cache_disabled_by_default(mock_yolo):
    det = YoloV8(device="cuda:0")
    assert det.empty_cache is False
    with patch("src.yolo.yolov8.torch.cuda.empty_cache") as ec, patch(
        "src.yolo.yolov8.torch.cuda.is_available", return_value=True
    ):
        det._maybe_empty_cache()
    ec.assert_not_called()
