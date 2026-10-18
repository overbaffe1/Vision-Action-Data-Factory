from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Sam2Masks:
    masks: np.ndarray
    scores: np.ndarray | None
    logits: np.ndarray | None
    original_size: tuple[int, int]

    # masks:
    #   - (N_obj, H, W) bool when multimask_output=False
    #   - (N_obj, K, H, W) bool when multimask_output=True
    #
    # scores:
    #   - (N_obj,) or (N_obj, K) float32 if model provides iou_scores
