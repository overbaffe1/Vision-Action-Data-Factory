from pathlib import Path
import logging
import cv2
import json
import numpy as np
from PIL import Image as PILImage

from .structure_checkers import (
    _check_bbox_json_structure,
    _check_captioning_json_structure,
    _check_pointing_json_structure,
)

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    """Creates a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def _validate_frames_directory(frames_dir_path: str | Path) -> Path:
    """Validates that the directory exists and contains at least one file."""

    path = Path(frames_dir_path)

    if not path.is_dir():
        raise NotADirectoryError(f"'{path}' is not a directory.")

    file_count = sum(1 for item in path.iterdir() if item.is_file())

    if file_count == 0:
        raise FileNotFoundError(f"No files found in directory '{path}'.")

    return path


def _check_image_correctness(image_path: str | Path) -> bool:
    """Checks whether an image file exists and can be opened properly."""

    image_path = Path(image_path)

    if not image_path.is_file():
        return False

    image = cv2.imread(str(image_path))

    return image is not None


def collect_prop_frames(frames_dir_path: str | Path) -> list[Path]:
    """Collecting all frames in apropriate extinction."""

    frames_paths = []
    frames_dir_path = _validate_frames_directory(frames_dir_path)

    for path in frames_dir_path.iterdir():
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        is_prop = _check_image_correctness(path)

        if is_prop:
            frames_paths.append(path)

    return frames_paths


def resize_single_image(
    image_path: str | Path, target_size: tuple[int, int] = (768, 512)
) -> PILImage.Image:
    """Resizes image with fixed max_side."""

    image_path = Path(image_path)

    if not image_path.is_file() or not image_path.exists():
        raise FileExistsError(f"File not found in {str(image_path)}")

    with PILImage.open(image_path) as img:
        img = img.convert("RGB")
        img_resized = img.resize(target_size)

    return img_resized


def save_resized_images_dir(
    src_dir: str | Path,
    dst_dir: str | Path,
    supported_extensions: set[str] | None = None,
    target_size: tuple[int, int] = (768, 512),
    jpeg_quality: int = 85,
    progressive: bool = True,
) -> list[Path]:
    """Resizes images from src_dir and saves them as JPEG into dst_dir.

    Only images whose longest side exceeds max_side are downscaled; the rest are
    saved as-is (converted to JPEG). Already processed files are skipped.
    Returns a list of paths to the saved files.
    """
    if supported_extensions is None:
        supported_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for src_path in sorted(src_dir.iterdir()):
        if src_path.suffix.lower() not in supported_extensions:
            continue
        dst_path = dst_dir / (src_path.stem + ".jpg")
        if dst_path.exists():
            logger.debug("resize_images: skipping %s — already processed", src_path.name)
            saved.append(dst_path)
            continue
        try:
            img = resize_single_image(src_path, target_size)
            img.save(
                dst_path,
                format="JPEG",
                quality=jpeg_quality,
                optimize=True,
                progressive=progressive,
            )
            saved.append(dst_path)
        except Exception as exc:
            logger.warning("resize_images: skipped %s — %s", src_path.name, exc)

    return saved


def build_extended_detections(detections_df) -> dict[str, list]:
    """Converts a YOLO DataFrame into {id, bbox, class, conf} dict."""

    result: dict[str, list] = {"id": [], "bbox": [], "class": [], "conf": []}
    for idx, row in detections_df.iterrows():
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        result["id"].append(idx + 1)
        result["bbox"].append([x1, y1, x2, y2])
        result["class"].append(row["class"])
        result["conf"].append(round(float(row["conf"]), 3))
    return result


def _compute_bbox_area(bbox: list[int]) -> int:
    """Computes area of choosen bounding box."""

    x1, y1, x2, y2 = bbox

    length = x2 - x1
    height = y2 - y1

    area = length * height

    return area


def _compute_image_area(image_H: int, image_W: int) -> int:
    """Computes area of image with known image_H and image_W."""

    return image_H * image_W


def filter_big_bboxes(
    image_size: tuple[int, int], bboxes_info: dict[str, list], threshold: int
) -> dict[str, list]:
    """Removes bboxes, which area greater than threshold% of image."""

    image_H, image_W = image_size

    image_area = _compute_image_area(image_H, image_W)
    max_area = image_area * threshold / 100

    bboxes_info_filtered: dict[str, list] = {"id": [], "bbox": [], "class": [], "conf": []}

    for det_id, bbox, cls, conf in zip(
        bboxes_info["id"],
        bboxes_info["bbox"],
        bboxes_info["class"],
        bboxes_info["conf"],
    ):
        bbox_area = _compute_bbox_area(bbox)
        if bbox_area <= max_area:
            bboxes_info_filtered["id"].append(det_id)
            bboxes_info_filtered["bbox"].append(bbox)
            bboxes_info_filtered["class"].append(cls)
            bboxes_info_filtered["conf"].append(conf)

    return bboxes_info_filtered


def format_bbox_dict(detections: dict[str, list]) -> dict[int, list[int]]:
    """Formats detections as {id: [x1, y1, x2, y2]} mapping."""

    return {
        int(det_id): list(map(int, bbox))
        for det_id, bbox in zip(detections["id"], detections["bbox"])
    }


def draw_detections_cv2(
    image: np.ndarray,
    detections: dict[str, list],
    box_color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    font_scale: float = 0.6,
    font_thickness: int = 2,
) -> np.ndarray:
    """Draws bounding boxes with numeric IDs on a BGR image using OpenCV."""

    output = image.copy()

    for det_id, bbox in zip(detections["id"], detections["bbox"]):
        x1, y1, x2, y2 = map(int, bbox)

        cv2.rectangle(output, (x1, y1), (x2, y2), box_color, thickness)

        label = str(det_id)
        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_thickness,
        )

        text_x = x1
        text_y = y1 - 6

        if text_y - text_h - baseline < 0:
            text_y = y1 + text_h + baseline + 6

        pad = 2
        cv2.rectangle(
            output,
            (text_x - pad, text_y - text_h - baseline - pad),
            (text_x + text_w + pad, text_y + pad),
            box_color,
            -1,
        )

        cv2.putText(
            output,
            label,
            (text_x, text_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    return output


def extract_grounding_json(raw_output: str) -> dict:
    """Extracts and parses the JSON block from a raw model output string."""

    start = raw_output.find("{")
    end = raw_output.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("JSON block not found in the provided output.")

    json_block = raw_output[start : end + 1]
    return json.loads(json_block)


def build_detection_result(
    raw_output: str,
    image_path: str,
) -> dict:
    """Builds a normalized detection JSON in the required dataset format."""

    parsed = extract_grounding_json(raw_output)

    grounding_question = parsed.get("grounding_question")
    grounding_answer = parsed.get("grounding_answer")

    if grounding_question is None or grounding_answer is None:
        raise ValueError("Invalid grounding JSON structure.")

    return {
        "images": [str(image_path)],
        "question": grounding_question,
        "answer": grounding_answer,
    }


def save_detection_json(
    result_json: dict,
    image_path: str | Path,
    output_dir: Path,
) -> Path:
    """Saves one LLaMA Factory JSONL shard using the image filename."""

    from .bpc.storage import atomic_save_llama_factory_record

    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = Path(image_path)

    output_path = output_dir / f"{image_path.stem}.jsonl"

    atomic_save_llama_factory_record(result_json, image_path, output_path)
    return output_path


def _to_bgr(image: np.ndarray | PILImage.Image) -> np.ndarray:
    """Converts an input image (PIL or numpy) to a BGR uint8 numpy array."""

    if isinstance(image, PILImage.Image):
        rgb = np.array(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(image)}")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def _normalize_masks(masks: np.ndarray) -> np.ndarray:
    """Normalizes SAM-like masks to a boolean array of shape (N, H, W)."""

    if masks.ndim == 4:
        masks = masks.squeeze()

    if masks.ndim == 2:
        masks = masks[None, :, :]

    elif masks.ndim != 3:
        raise ValueError("Masks must be 2D or 3D array.")

    return masks.astype(bool)


def add_mask_padding(
    mask: np.ndarray,
    pad: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Adds zero padding around a mask and returns the offset."""

    padded = cv2.copyMakeBorder(
        mask,
        top=pad,
        bottom=pad,
        left=pad,
        right=pad,
        borderType=cv2.BORDER_CONSTANT,
        value=(0,),
    )
    return padded, (pad, pad)


def remove_padding_from_point(
    point: tuple[int, int],
    offset: tuple[int, int],
) -> tuple[int, int]:
    """Converts a point from padded coordinates back to original coordinates."""

    x, y = point
    offset_x, offset_y = offset
    return x - offset_x, y - offset_y


def _mask_inner_point(
    mask: np.ndarray,
    pad: int = 8,
) -> tuple[int, int] | None:
    """Finds an inner point in a padded mask and maps it back to original coordinates."""

    mask_u8 = (mask > 0).astype(np.uint8)
    if not np.any(mask_u8):
        return None

    padded_mask, offset = add_mask_padding(mask_u8, pad)

    dist = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist)
    if max_val <= 0:
        return None

    point = remove_padding_from_point(max_loc, offset)

    x, y = point
    h, w = mask_u8.shape
    if not (0 <= x < w and 0 <= y < h):
        return None

    return x, y


def draw_sam2_masks_with_centroids(
    image: np.ndarray,
    masks: np.ndarray,
    *,
    alpha: float = 0.45,
    circle_radius: int = 16,
    text_scale: float = 1.0,
    text_thickness: int = 3,
    start_index: int = 1,
) -> tuple[np.ndarray, dict[str, list]]:
    """Overlays masks, draws centroid markers with indices, and returns centroid metadata."""

    bgr = _to_bgr(image)
    masks_bool = _normalize_masks(masks)

    h, w = bgr.shape[:2]
    if masks_bool.shape[1:] != (h, w):
        raise ValueError("Mask size does not match image size.")

    out = bgr.copy()

    result: dict[str, list] = {
        "id": [],
        "points": [],
    }

    for i, mask in enumerate(masks_bool):
        if not mask.any():
            continue

        idx = start_index + i

        # Overlay mask
        color = np.array([0, 255, 0], dtype=np.uint8)
        overlay = out.copy()
        overlay[mask] = (overlay[mask] * (1.0 - alpha) + color * alpha).astype(np.uint8)
        out = overlay

        centroid = _mask_inner_point(mask)
        if centroid is None:
            continue

        cx, cy = centroid

        result["id"].append(idx)
        result["points"].append([cx, cy])

        # Draw circle
        cv2.circle(
            out,
            (cx, cy),
            circle_radius,
            (0, 0, 255),
            thickness=-1,
        )

        label = str(idx)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_thickness)

        tx = int(cx - tw / 2)
        ty = int(cy + th / 2)

        cv2.putText(
            out,
            label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return out, result


def _try_save_vlm_json(raw_output: str, *, frame_path: str, out_dir: Path) -> Path | None:
    """Builds, validates and saves one LLaMA Factory JSONL annotation shard."""
    _ensure_dir(out_dir)

    task_name = out_dir.name

    try:
        result = build_detection_result(raw_output, frame_path)

        if task_name == "bboxing":
            _check_bbox_json_structure(result)

        elif task_name == "pointing":
            _check_pointing_json_structure(result)

        elif task_name == "captions":
            _check_captioning_json_structure(result)

        else:
            raise ValueError(f"Undifined task name {task_name}")

        return save_detection_json(result, frame_path, out_dir)
    except Exception as exc:
        logger.warning("Failed to save VLM JSON for %s: %s", frame_path, str(exc))
        return None


def _collect_input_images(
    *,
    data_path: str | None,
    img_path: str | None,
    processed_images_dir: Path,
    resize_cfg: dict,
) -> list[Path]:
    """Collects, resizes and returns input images prepared for the pipeline."""

    supported_extensions = {ext.lower() for ext in resize_cfg["supported_extensions"]}
    target_size: tuple[int, int] = tuple(resize_cfg["target_size"])

    _ensure_dir(processed_images_dir)

    if img_path is not None:
        image_path = Path(img_path)
        logger.info("Resizing image: %s -> %s", image_path, processed_images_dir)

        if target_size:
            img = resize_single_image(image_path, target_size)
        else:
            img = resize_single_image(image_path)

        dst_path = processed_images_dir / f"{image_path.stem}.jpg"
        img.save(
            dst_path,
            format="JPEG",
            quality=resize_cfg["jpeg_quality"],
            optimize=True,
            progressive=resize_cfg["progressive"],
        )
        return [dst_path]

    raw_images_dir = Path(data_path) if data_path else None
    if raw_images_dir is None:
        raise ValueError("data_path must be set when img_path is not provided.")

    images_number = sum(
        1
        for file_path in raw_images_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions
    )
    if images_number == 0:
        raise FileNotFoundError(f"No supported files found in {raw_images_dir}.")

    logger.info("Resizing images: %s -> %s", raw_images_dir, processed_images_dir)

    return save_resized_images_dir(
        raw_images_dir,
        processed_images_dir,
        supported_extensions=supported_extensions,
        target_size=target_size,
        jpeg_quality=resize_cfg["jpeg_quality"],
        progressive=resize_cfg["progressive"],
    )
