"""SmolVLM2 evaluation model for lmms-eval.

Supports Hugging Face SmolVLM2 image checkpoints via ``AutoProcessor`` and
``AutoModelForImageTextToText``. Video inputs are not supported yet; requests
with video paths fail with a clear error instead of hanging.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, List, Optional, Tuple, Union

import PIL.Image
import torch
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from loguru import logger as eval_logger
from transformers import AutoModelForImageTextToText, AutoProcessor

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.progress import make_progress

warnings.filterwarnings("ignore")

DEFAULT_IMAGE_TOKEN = "<image>"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif")

try:
    import flash_attn  # noqa: F401

    _DEFAULT_ATTN = "flash_attention_2"
except ImportError:
    _DEFAULT_ATTN = None


def _smolvlm2_debug_enabled() -> bool:
    return os.environ.get("LMMS_EVAL_SMOLVLM2_DEBUG", "").strip() in {"1", "true", "True", "yes"}


def _smolvlm2_debug(message: str, **kwargs: Any) -> None:
    if not _smolvlm2_debug_enabled():
        return
    parts = [message]
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            parts.append(f"{key}=Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})")
        else:
            parts.append(f"{key}={value!r}")
    eval_logger.info("[smolvlm2] " + " | ".join(parts))


def _tensor_summary(inputs: dict) -> str:
    parts = []
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            parts.append(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            parts.append(f"{key}: {type(value).__name__}")
    return ", ".join(parts)


def _resolve_torch_dtype(dtype: Optional[Union[str, torch.dtype]]) -> Optional[torch.dtype]:
    if dtype is None or dtype == "auto":
        return torch.bfloat16
    if isinstance(dtype, str):
        return getattr(torch, dtype)
    return dtype


def _to_pil_rgb(visual: Any) -> PIL.Image.Image:
    if isinstance(visual, PIL.Image.Image):
        return visual.convert("RGB")
    if isinstance(visual, str):
        return PIL.Image.open(visual).convert("RGB")
    raise TypeError(f"Unsupported visual type for SmolVLM2: {type(visual).__name__}")


def _is_video_path(path: str) -> bool:
    return path.lower().endswith(VIDEO_EXTENSIONS)


@register_model("smolvlm2")
class SmolVLM2(lmms):
    """SmolVLM2 wrapper for image-based VLM benchmarks.

  Example::

      accelerate launch -m lmms_eval \\
          --model smolvlm2 \\
          --model_args pretrained=HuggingFaceTB/SmolVLM2-500M-Video-Instruct,device_map=auto,torch_dtype=bfloat16 \\
          --tasks mme \\
          --batch_size 1 \\
          --log_samples \\
          --output_path ./logs/smolvlm2_500m
    """

    def __init__(
        self,
        pretrained: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        revision: str = "main",
        device: str = "cuda",
        torch_dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        batch_size: int = 1,
        trust_remote_code: Optional[bool] = True,
        attn_implementation: Optional[str] = _DEFAULT_ATTN,
        device_map: str = "auto",
        use_cache: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            eval_logger.warning(f"Ignoring unsupported kwargs for smolvlm2: {sorted(kwargs.keys())}")

        self.pretrained = pretrained
        self.use_cache = use_cache
        resolved_dtype = _resolve_torch_dtype(torch_dtype)

        accelerator = Accelerator()
        if accelerator.num_processes > 1 and device_map in ("", None):
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
            self.device_map = device_map

        model_kwargs: dict[str, Any] = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
        }
        if resolved_dtype is not None:
            model_kwargs["dtype"] = resolved_dtype
        if self.device_map:
            model_kwargs["device_map"] = self.device_map
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation

        self._processor = AutoProcessor.from_pretrained(pretrained, revision=revision, trust_remote_code=trust_remote_code)
        self._model = AutoModelForImageTextToText.from_pretrained(pretrained, **model_kwargs).eval()
        self._tokenizer = self._processor.tokenizer
        self._config = self._model.config
        self._max_length = getattr(self._config, "max_position_embeddings", 8192)
        self._model_dtype = resolved_dtype or torch.float32
        self.batch_size_per_gpu = int(batch_size)

        if self.batch_size_per_gpu != 1:
            eval_logger.warning("SmolVLM2 wrapper currently supports batch_size=1 only; forcing batch_size=1.")
            self.batch_size_per_gpu = 1

        _smolvlm2_debug(
            "model loaded",
            pretrained=pretrained,
            processor_class=type(self._processor).__name__,
            model_class=type(self._model).__name__,
            device_map=self.device_map,
            torch_dtype=self._model_dtype,
        )

        if accelerator.num_processes > 1 and device_map in ("", None):
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                ds_kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **ds_kwargs)
                eval_logger.info("Detected DistributedType.DEEPSPEED. Run `accelerate config` with zero stage 0.")
            if accelerator.distributed_type in {DistributedType.FSDP, DistributedType.DEEPSPEED}:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and self.device_map == "auto":
            eval_logger.info("Using device_map=auto pipeline parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            if self.device_map in ("", None):
                self.model.to(self._device)
            eval_logger.info(f"Using device: {self._device}")
            self._rank = 0
            self._world_size = 1

        self.accelerator = accelerator

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def flatten(self, items: List) -> List:
        flat: List = []
        for item in items:
            if isinstance(item, list):
                flat.extend(item)
            elif item is None:
                continue
            else:
                flat.append(item)
        return flat

    def _model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _move_inputs_to_model(self, inputs: dict) -> dict:
        model_device = self._model_device()
        moved = {}
        for key, value in inputs.items():
            if not isinstance(value, torch.Tensor):
                moved[key] = value
                continue
            if value.is_floating_point():
                moved[key] = value.to(model_device, dtype=self._model_dtype)
            else:
                moved[key] = value.to(model_device)
        return moved

    def _build_messages(self, context: str, num_images: int) -> List[dict]:
        content: List[dict] = [{"type": "image"} for _ in range(num_images)]
        content.append({"type": "text", "text": context})
        return [{"role": "user", "content": content}]

    def _prepare_visuals(self, visuals: List) -> List[PIL.Image.Image]:
        pil_images: List[PIL.Image.Image] = []
        for visual in visuals:
            if isinstance(visual, str) and _is_video_path(visual):
                raise NotImplementedError(
                    "SmolVLM2 video evaluation is not implemented in lmms-eval yet. "
                    "Use an image benchmark or a dedicated video wrapper."
                )
            pil_images.append(_to_pil_rgb(visual))
        return pil_images

    def _prepare_processor_inputs(self, context: str, pil_images: List[PIL.Image.Image]) -> dict:
        messages = self._build_messages(context, len(pil_images))
        prompt_text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        _smolvlm2_debug(
            "prepared prompt",
            num_images=len(pil_images),
            prompt_text=prompt_text,
        )
        if pil_images:
            inputs = self._processor(
                text=prompt_text,
                images=pil_images,
                text_kwargs={"return_tensors": "pt"},
            )
        else:
            inputs = self._processor(text=prompt_text, text_kwargs={"return_tensors": "pt"})
        if hasattr(inputs, "items"):
            inputs = dict(inputs.items())
        else:
            inputs = dict(inputs)
        _smolvlm2_debug("processor output", keys=list(inputs.keys()), summary=_tensor_summary(inputs))
        return self._move_inputs_to_model(inputs)

    def _build_generation_kwargs(self, gen_kwargs: dict) -> dict:
        max_new_tokens = gen_kwargs.get("max_new_tokens", 128)
        temperature = gen_kwargs.get("temperature", 0)
        top_p = gen_kwargs.get("top_p", 1.0)
        num_beams = gen_kwargs.get("num_beams", 1)
        do_sample = gen_kwargs.get("do_sample", temperature > 0)

        call_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "do_sample": do_sample,
            "use_cache": self.use_cache,
        }
        if do_sample:
            call_kwargs["temperature"] = temperature
            call_kwargs["top_p"] = top_p
        return call_kwargs

    def _decode_generation(self, inputs: dict, output_ids: torch.Tensor) -> str:
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_len:]
        answer = self._processor.decode(generated_ids, skip_special_tokens=True)
        _smolvlm2_debug("decoded output", raw_output=answer)
        return answer

    def _apply_stop_strings(self, answer: str, until: Optional[Union[str, List[str]]]) -> str:
        if until is None:
            until = [self.tokenizer.decode(self.eot_token_id)]
        if isinstance(until, str):
            until = [until]
        until = [item for item in until if item != "\n\n"]
        for term in until:
            if term:
                answer = answer.split(term)[0]
        return answer.strip()

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("SmolVLM2 loglikelihood scoring is not implemented.")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res: List[str] = []
        pbar = make_progress(total=len(requests), disable=(self.rank != 0), desc="SmolVLM2 Responding")

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            visuals = self.flatten(visuals if isinstance(visuals, list) else [visuals])

            context = contexts
            if DEFAULT_IMAGE_TOKEN in context:
                context = context.replace(DEFAULT_IMAGE_TOKEN, "").strip()

            pil_images = self._prepare_visuals(visuals) if visuals else []
            inputs = self._prepare_processor_inputs(context, pil_images)
            generation_kwargs = self._build_generation_kwargs(gen_kwargs)
            until = gen_kwargs.get("until")

            _smolvlm2_debug("generation kwargs", generation_kwargs=generation_kwargs, device=str(self._model_device()))

            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, **generation_kwargs)

            answer = self._decode_generation(inputs, output_ids)
            answer = self._apply_stop_strings(answer, until)

            res.append(answer)
            self.cache_hook.add_partial("generate_until", (contexts, gen_kwargs), answer)
            pbar.update(1)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("SmolVLM2 does not support multi-round generation.")
