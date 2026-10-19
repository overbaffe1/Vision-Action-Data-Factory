from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .builder import Sam2Bundle, build_sam2
from .types import Sam2Masks
from .utils import (
    squeeze_iou_scores,
    to_bool_masks,
    to_pil_rgb,
    wrap_boxes_for_processor,
    wrap_labels_for_processor,
    wrap_points_for_processor,
)


class Sam2Segmenter:
    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-large",
        device: str | torch.device = "cuda",
        dtype: str | None = None,
        empty_cache: bool = False,
    ) -> None:
        self.bundle: Sam2Bundle = build_sam2(model_id=model_id, device=device, dtype=dtype)
        self._image: Image.Image | None = None
        self._original_size: tuple[int, int] | None = None
        self.empty_cache = empty_cache

    @property
    def device(self) -> torch.device:
        return self.bundle.device

    def _maybe_empty_cache(self) -> None:
        """Frees CUDA cache after inference when enabled and on a CUDA device."""

        if self.empty_cache and self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _cast_float_inputs(self, inputs: dict) -> dict:
        """Casts float tensors in the processor output to the model's dtype.

        Sam2Processor always returns float32 pixel tensors regardless of the
        model's loaded dtype. Sam2Model.from_pretrained(torch_dtype=...) only
        moves the model's own weights (e.g. bfloat16) — the pixel_values input
        stays float32, so a raw forward pass raises "Input type (float) and
        bias type (c10::BFloat16) should be the same". Integer/bool tensors
        (e.g. input_labels, original_sizes) must NOT be cast.
        """
        model_dtype = self.bundle.dtype
        if model_dtype is None:
            return inputs
        return {
            key: value.to(model_dtype) if torch.is_tensor(value) and torch.is_floating_point(value) else value
            for key, value in inputs.items()
        }

    def set_image(self, image: Image.Image | np.ndarray) -> None:
        """Sets the current image used for subsequent predictions."""

        pil = to_pil_rgb(image)
        self._image = pil
        self._original_size = (pil.height, pil.width)

    def set_image_path(self, image_path: str | Path) -> None:
        """Loads and sets the current image from disk."""

        pil = Image.open(image_path).convert("RGB")
        self._image = pil
        self._original_size = (pil.height, pil.width)

    def predict_from_boxes(
        self,
        boxes_xyxy: np.ndarray,
        multimask_output: bool = False,
        return_logits: bool = False,
    ) -> Sam2Masks:
        """Predicts masks for the current image from bounding boxes prompts."""

        if self._image is None or self._original_size is None:
            raise RuntimeError("Image is not set. Call set_image()/set_image_path() first.")

        input_boxes = wrap_boxes_for_processor(boxes_xyxy)

        inputs = self.bundle.processor(
            images=self._image,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self.device)
        inputs = self._cast_float_inputs(inputs)

        with torch.no_grad():
            outputs = self.bundle.model(**inputs, multimask_output=multimask_output)

        # HF examples for SAM2 use post_process_masks(pred_masks, original_sizes)[0] :contentReference[oaicite:1]{index=1}
        # .float() before .cpu(): numpy has no native bfloat16 dtype, so
        # np.asarray() on a raw bfloat16 tensor would raise/behave unreliably
        # across torch/numpy versions when the model was loaded in bfloat16.
        post = self.bundle.processor.post_process_masks(
            outputs.pred_masks.float().cpu(), inputs["original_sizes"]
        )
        masks = to_bool_masks(np.asarray(post[0]))

        scores = squeeze_iou_scores(getattr(outputs, "iou_scores", None))
        if isinstance(scores, np.ndarray):
            scores = scores.astype(np.float32)
        else:
            scores = None

        logits = None
        if return_logits:
            logits = np.asarray(outputs.pred_masks.detach().float().cpu())

        self._maybe_empty_cache()
        return Sam2Masks(
            masks=masks,
            scores=scores,
            logits=logits,
            original_size=self._original_size,
        )

    def predict_from_points(
        self,
        points_xy: np.ndarray,
        labels: np.ndarray,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Sam2Masks:
        """Predicts masks for the current image from point prompts."""

        if self._image is None or self._original_size is None:
            raise RuntimeError("Image is not set. Call set_image()/set_image_path() first.")

        input_points = wrap_points_for_processor(points_xy)
        input_labels = wrap_labels_for_processor(labels, points_xy)

        inputs = self.bundle.processor(
            images=self._image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(self.device)
        inputs = self._cast_float_inputs(inputs)

        with torch.no_grad():
            outputs = self.bundle.model(**inputs, multimask_output=multimask_output)

        # .float() before .cpu(): numpy has no native bfloat16 dtype (see
        # predict_from_boxes for the same guard).
        post = self.bundle.processor.post_process_masks(
            outputs.pred_masks.float().cpu(), inputs["original_sizes"]
        )
        masks = to_bool_masks(np.asarray(post[0]))

        scores = squeeze_iou_scores(getattr(outputs, "iou_scores", None))
        if isinstance(scores, np.ndarray):
            scores = scores.astype(np.float32)
        else:
            scores = None

        logits = None
        if return_logits:
            logits = np.asarray(outputs.pred_masks.detach().float().cpu())

        self._maybe_empty_cache()
        return Sam2Masks(
            masks=masks,
            scores=scores,
            logits=logits,
            original_size=self._original_size,
        )
