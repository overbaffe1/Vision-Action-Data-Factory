from __future__ import annotations

from PIL import Image
import numpy as np


def to_pil_rgb(image: Image.Image | np.ndarray) -> Image.Image:
    """Converts an image into a PIL RGB image."""

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(image)}")

    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3/4 array, got shape {image.shape}")

    if image.shape[2] == 4:
        image = image[:, :, :3]

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    return Image.fromarray(image, mode="RGB")


def normalize_xyxy_boxes(boxes_xyxy: np.ndarray) -> np.ndarray:
    """Normalizes boxes into float32 (N, 4) xyxy."""

    boxes = np.asarray(boxes_xyxy, dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Expected boxes shape (N, 4), got {boxes.shape}")
    return boxes


def wrap_boxes_for_processor(boxes_xyxy: np.ndarray) -> list[list[list[float]]]:
    """Wraps boxes into processor format: [ [ [x1,y1,x2,y2], ... ] ]."""

    boxes = normalize_xyxy_boxes(boxes_xyxy)
    return [[box.tolist() for box in boxes]]


def wrap_points_for_processor(points_xy: np.ndarray) -> list[list[list[list[float]]]]:
    """Wraps points into processor format: [ [ [ [x,y], ... ], ... ] ]."""

    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim != 3 or pts.shape[2] != 2:
        raise ValueError(f"Expected points shape (N_obj, N_points, 2), got {pts.shape}")

    per_object = []
    for obj_pts in pts:
        per_object.append([p.tolist() for p in obj_pts])

    return [per_object]


def wrap_labels_for_processor(labels: np.ndarray, points_xy: np.ndarray) -> list[list[list[int]]]:
    """Wraps labels into processor format: [ [ [l1,l2,...], ... ] ]."""

    lbs = np.asarray(labels, dtype=np.int64)
    pts = np.asarray(points_xy)

    if lbs.shape != pts.shape[:2]:
        raise ValueError(f"Expected labels shape {pts.shape[:2]}, got {lbs.shape}")

    per_object = []
    for obj_lbs in lbs:
        per_object.append([int(x) for x in obj_lbs.tolist()])

    return [per_object]


def to_bool_masks(masks: np.ndarray) -> np.ndarray:
    """Converts predicted masks into boolean numpy array."""

    m = np.asarray(masks)
    if m.dtype != np.bool_:
        m = m > 0
    return m


def squeeze_iou_scores(iou_scores: np.ndarray | None) -> np.ndarray | None:
    """Normalizes iou_scores to (N_obj,) or (N_obj, K) float32 if present."""

    if iou_scores is None:
        return None

    # .float() before .cpu()/np.asarray(): passing dtype=np.float32 to
    # np.asarray() does NOT save a raw bfloat16 tensor here — numpy calls
    # Tensor.__array__() first, which does .numpy() BEFORE casting, and
    # torch.Tensor.numpy() itself raises "Got unsupported ScalarType
    # BFloat16" (numpy has no native bfloat16 dtype) before the requested
    # dtype is ever applied. Must cast to float32 on the torch side first,
    # same fix as pred_masks.float() in segmenter.py's predict_from_boxes/
    # predict_from_points -- iou_scores was the one call site that fix
    # missed, and it fires on every single SAM2 call when the model is
    # loaded in bfloat16.
    iou_scores = iou_scores.detach().float().cpu()

    s = np.asarray(iou_scores, dtype=np.float32)

    # Common shapes observed in HF outputs:
    # (B, N_obj, K) or (B, N_obj, 1, K) depending on implementation details.
    while s.ndim > 1 and s.shape[0] == 1:
        s = s[0]

    # Now s is typically (N_obj, K) or (N_obj,)
    return s
