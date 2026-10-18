from __future__ import annotations

import logging
from dataclasses import dataclass
import torch
from transformers import Sam2Model, Sam2Processor

# Suppress specific SAM2 loading warnings
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.ERROR)


@dataclass(frozen=True)
class Sam2Bundle:
    model: Sam2Model
    processor: Sam2Processor
    device: torch.device
    dtype: torch.dtype | None = None


def build_sam2(
    model_id: str = "facebook/sam2.1-hiera-large",
    device: str | torch.device = "cuda",
    dtype: str | None = None,
) -> Sam2Bundle:
    """Loads SAM2 model+processor from Hugging Face transformers."""

    dev = torch.device(device) if not isinstance(device, torch.device) else device

    torch_dtype = None
    if dtype is not None:
        if dtype == "float16":
            torch_dtype = torch.float16
        elif dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif dtype == "float32":
            torch_dtype = torch.float32
        else:
            raise ValueError("dtype must be one of: float16, bfloat16, float32")

    model = Sam2Model.from_pretrained(model_id, torch_dtype=torch_dtype).to(dev)
    processor = Sam2Processor.from_pretrained(model_id)

    model.eval()
    return Sam2Bundle(model=model, processor=processor, device=dev, dtype=torch_dtype)
