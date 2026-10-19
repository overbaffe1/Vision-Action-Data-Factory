"""Tests for src/sam2/utils.py — geometry wrappers and iou_scores dtype handling.

squeeze_iou_scores had no test coverage at all before this file; that gap is
exactly why the bfloat16 crash (real traceback: torch.Tensor.numpy() raising
"Got unsupported ScalarType BFloat16" inside np.asarray(iou_scores,
dtype=np.float32) BEFORE the requested dtype cast could apply) shipped and
only surfaced on a live GPU run with a bfloat16-loaded model — the existing
stub-based test suite never exercised a tensor whose .numpy() call can fail.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.sam2.utils import (
    normalize_xyxy_boxes,
    wrap_boxes_for_processor,
    wrap_points_for_processor,
    wrap_labels_for_processor,
    to_bool_masks,
    squeeze_iou_scores,
)


class _FakeBFloat16Tensor:
    """Emulates a torch bfloat16 tensor: .numpy() raises like the real one
    (numpy has no native bfloat16 dtype) until .float() has been called."""

    def __init__(self, data, is_bfloat16: bool = True):
        self._data = np.asarray(data, dtype=np.float32)
        self._is_bfloat16 = is_bfloat16

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return _FakeBFloat16Tensor(self._data, is_bfloat16=False)

    def numpy(self):
        if self._is_bfloat16:
            raise TypeError("Got unsupported ScalarType BFloat16")
        return self._data

    def __array__(self, dtype=None):
        # Mirrors torch.Tensor.__array__: calls .numpy() first, THEN casts —
        # so passing dtype=np.float32 to np.asarray() does not save a raw
        # bfloat16 tensor; this is the exact mechanism that broke in prod.
        arr = self.numpy()
        return arr.astype(dtype) if dtype is not None else arr


class TestSqueezeIouScoresDtypeHandling:
    def test_none_passthrough(self):
        assert squeeze_iou_scores(None) is None

    def test_regular_array_like_tensor_works(self):
        t = _FakeBFloat16Tensor([[0.9, 0.8]], is_bfloat16=False)
        result = squeeze_iou_scores(t)
        assert result is not None
        assert result.dtype == np.float32

    def test_bfloat16_tensor_does_not_raise(self):
        """The exact regression: without .float() before np.asarray(), this
        raised TypeError('Got unsupported ScalarType BFloat16') on every
        single SAM2 call when the model was loaded in bfloat16."""
        t = _FakeBFloat16Tensor([[0.9, 0.8, 0.7]], is_bfloat16=True)
        result = squeeze_iou_scores(t)  # must not raise
        assert result is not None
        assert result.dtype == np.float32

    def test_bfloat16_raw_asarray_would_have_failed(self):
        """Sanity check that the fake tensor actually reproduces the bug
        when .float() is skipped — proves the regression test is real."""
        t = _FakeBFloat16Tensor([[0.9]], is_bfloat16=True)
        with pytest.raises(TypeError, match="BFloat16"):
            np.asarray(t, dtype=np.float32)

    def test_squeezes_leading_batch_dim_of_1(self):
        t = _FakeBFloat16Tensor([[[0.9, 0.8], [0.5, 0.4]]], is_bfloat16=False)
        result = squeeze_iou_scores(t)
        assert result.shape == (2, 2)

    def test_squeezes_multiple_leading_dims_of_1(self):
        t = _FakeBFloat16Tensor([[[[0.9, 0.8]]]], is_bfloat16=False)
        result = squeeze_iou_scores(t)
        assert result.shape == (2,)

    def test_no_squeeze_when_batch_dim_not_1(self):
        t = _FakeBFloat16Tensor([[0.9, 0.8], [0.5, 0.4]], is_bfloat16=False)
        result = squeeze_iou_scores(t)
        assert result.shape == (2, 2)


class TestBoxWrappers:
    def test_normalize_xyxy_boxes_valid(self):
        boxes = normalize_xyxy_boxes([[1, 2, 3, 4], [5, 6, 7, 8]])
        assert boxes.shape == (2, 4)
        assert boxes.dtype == np.float32

    def test_normalize_xyxy_boxes_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="Expected boxes shape"):
            normalize_xyxy_boxes([[1, 2, 3]])

    def test_wrap_boxes_for_processor(self):
        wrapped = wrap_boxes_for_processor([[1, 2, 3, 4]])
        assert wrapped == [[[1.0, 2.0, 3.0, 4.0]]]


class TestPointWrappers:
    def test_wrap_points_for_processor_valid(self):
        wrapped = wrap_points_for_processor([[[1, 2], [3, 4]]])
        assert wrapped == [[[[1.0, 2.0], [3.0, 4.0]]]]

    def test_wrap_points_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="Expected points shape"):
            wrap_points_for_processor([[1, 2, 3]])

    def test_wrap_labels_for_processor_valid(self):
        wrapped = wrap_labels_for_processor([[1, 0]], [[[1, 2], [3, 4]]])
        assert wrapped == [[[1, 0]]]

    def test_wrap_labels_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Expected labels shape"):
            wrap_labels_for_processor([[1]], [[[1, 2], [3, 4]]])


class TestToBoolMasks:
    def test_already_bool(self):
        m = np.array([[True, False]])
        result = to_bool_masks(m)
        assert result.dtype == np.bool_

    def test_converts_float_to_bool(self):
        m = np.array([[0.9, -0.1, 0.0]])
        result = to_bool_masks(m)
        assert result.dtype == np.bool_
        assert list(result[0]) == [True, False, False]
