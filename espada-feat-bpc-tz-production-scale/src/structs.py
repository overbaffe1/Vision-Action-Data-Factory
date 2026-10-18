from dataclasses import dataclass
import numpy as np
from PIL import Image
from pathlib import Path

from .vlm import get_system_user_prompts


@dataclass(slots=True)
class PipelineProgressEvent:
    """Represents progress of one image in the pipeline."""

    image_id: str
    kind: (
        str  # "bboxing_done", "captions_done", "pointing_done", "pointing_skipped", "image_failed"
    )


@dataclass(slots=True)
class FrameTask:
    """Represents a single image scheduled for pipeline processing."""

    image_id: str
    frame_path: Path


@dataclass(slots=True)
class YoloStageResult:
    """Stores image data and YOLO outputs for downstream stages."""

    image_id: str
    frame_path: Path
    original_image: np.ndarray
    pil_image: Image.Image
    detections: dict
    vis_image_boxes: np.ndarray


@dataclass(slots=True)
class VlmTask:
    """Represents one VLM request ready to be executed and saved."""

    task_name: str
    frame_path: Path
    system_text: str
    user_text: str
    images: list[np.ndarray]


def _build_bbox_vlm_task(
    *,
    frame_path: Path,
    original_image: np.ndarray,
    detections: dict,
    vis_image_boxes: np.ndarray,
    prompts: dict,
) -> VlmTask:
    """Builds the bboxing VLM task."""

    system_text, user_text = get_system_user_prompts(prompts, "bboxing", detections)

    return VlmTask(
        task_name="bboxing",
        frame_path=frame_path,
        system_text=system_text,
        user_text=user_text,
        images=[original_image, vis_image_boxes],
    )


def _build_caption_vlm_task(
    *,
    frame_path: Path,
    original_image: np.ndarray,
    prompts: dict,
) -> VlmTask:
    """Builds the captioning VLM task."""

    system_text, user_text = get_system_user_prompts(prompts, "general_captioning")

    return VlmTask(
        task_name="captions",
        frame_path=frame_path,
        system_text=system_text,
        user_text=user_text,
        images=[original_image],
    )


def _build_pointing_vlm_task(
    *,
    frame_path: Path,
    original_image: np.ndarray,
    meta: dict,
    vis_image_points: np.ndarray,
    prompts: dict,
) -> VlmTask:
    """Builds the pointing VLM task."""

    system_text, user_text = get_system_user_prompts(prompts, "pointing", meta)

    return VlmTask(
        task_name="pointing",
        frame_path=frame_path,
        system_text=system_text,
        user_text=user_text,
        images=[original_image, vis_image_points],
    )
