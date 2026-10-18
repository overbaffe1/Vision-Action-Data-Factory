"""Utilities for ERQA evaluation on locally converted JSONL + image files."""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

eval_logger = logging.getLogger("lmms-eval")

ERQA_DATA_DIR_ENV = "ERQA_DATA_DIR"
ERQA_JSONL_NAME = "erqa.jsonl"
OPTION_LETTERS = frozenset({"A", "B", "C", "D"})
ANSWER_INSTRUCTION = (
    "\nAnswer with the single letter corresponding to the correct option. "
    "Do not explain. Your answer must be one of A, B, C, or D."
)

_TFRECORD_SUFFIXES = (".tfrecord", ".tfrecords")

_ANSWER_PATTERNS = [
    re.compile(r"answer\s*(?:is|:|-)\s*\(?([A-D])\)?\b", re.IGNORECASE),
    re.compile(r"the\s+(?:correct\s+)?answer\s+is\s*\(?([A-D])\)?\b", re.IGNORECASE),
    re.compile(r"option\s*([A-D])\b", re.IGNORECASE),
    re.compile(r"\(([A-D])\)", re.IGNORECASE),
    re.compile(r"^[\(\s]*([A-D])[\)\.\s]*$", re.IGNORECASE),
    re.compile(r"\b([A-D])\b", re.IGNORECASE),
]

_DATA_DIR: Path | None = None


class ERQADataError(ValueError):
    """Raised when ERQA data is missing or not converted."""


def _expand_path(path: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(path))).resolve()


def _is_tfrecord_path(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in _TFRECORD_SUFFIXES)


def get_erqa_data_dir(*, required: bool = True) -> Path:
    """Return the directory containing converted ERQA JSONL and images."""
    global _DATA_DIR
    if _DATA_DIR is not None:
        return _DATA_DIR

    raw = os.environ.get(ERQA_DATA_DIR_ENV, "").strip()
    if not raw:
        if not required:
            raise ERQADataError(
                f"{ERQA_DATA_DIR_ENV} is not set. Convert ERQA offline with:\n"
                "  python lmms_eval/tasks/erqa/convert_erqa.py "
                "--input /path/to/erqa.tfrecord --output_dir /path/to/erqa_converted\n"
                f"Then export {ERQA_DATA_DIR_ENV}=/path/to/erqa_converted"
            )
        raise ERQADataError(
            f"{ERQA_DATA_DIR_ENV} is not set. Convert ERQA TFRecord offline first; see lmms_eval/tasks/erqa/README.md."
        )

    path = _expand_path(raw)
    if _is_tfrecord_path(path):
        raise ERQADataError(
            f"{ERQA_DATA_DIR_ENV} points to a TFRecord file ({path}). "
            "Runtime evaluation does not read TFRecord. Run convert_erqa.py first, then set "
            f"{ERQA_DATA_DIR_ENV} to the conversion output directory."
        )
    if path.is_file():
        if _is_tfrecord_path(path):
            raise ERQADataError(
                f"{ERQA_DATA_DIR_ENV} must be the converted data directory, not {path.name}. "
                "Run convert_erqa.py and point the env var at --output_dir."
            )
        path = path.parent

    jsonl_path = path / ERQA_JSONL_NAME
    if not jsonl_path.is_file():
        if (path / "erqa.tfrecord").is_file() or any(path.glob("*.tfrecord")):
            raise ERQADataError(
                f"Found TFRecord under {path} but missing {ERQA_JSONL_NAME}. "
                "Convert the dataset offline:\n"
                "  python lmms_eval/tasks/erqa/convert_erqa.py "
                f"--input {path / 'erqa.tfrecord'} --output_dir {path}"
            )
        raise ERQADataError(
            f"Missing {jsonl_path}. Run convert_erqa.py, then set {ERQA_DATA_DIR_ENV}={path}"
        )

    _DATA_DIR = path
    return path


def erqa_process_docs(dataset):
    """Validate converted data is present before evaluation starts."""
    get_erqa_data_dir(required=True)
    return dataset


def _resolve_image_path(image_path: str, data_dir: Path) -> Path:
    if not image_path or not isinstance(image_path, str):
        raise FileNotFoundError("ERQA sample has an empty image path.")

    if _is_tfrecord_path(Path(image_path)):
        raise ERQADataError(
            f"Image path {image_path!r} refers to TFRecord. Use converted JSONL + images only."
        )

    candidate = Path(image_path)
    if candidate.is_file():
        return candidate.resolve()

    rooted = (data_dir / candidate).resolve()
    if rooted.is_file():
        return rooted

    raise FileNotFoundError(
        f"ERQA image not found: {image_path!r} (resolved under {data_dir}). "
        "Re-run convert_erqa.py or fix image_paths in erqa.jsonl."
    )


def extract_answer_letter(text: str) -> str:
    """Extract the first valid multiple-choice letter (A–D) from model output."""
    normalized = str(text).strip()
    if not normalized:
        return ""

    for pattern in _ANSWER_PATTERNS:
        match = pattern.search(normalized)
        if match:
            letter = match.group(1).upper()
            if letter in OPTION_LETTERS:
                return letter

    if len(normalized) <= 3 and normalized[:1].upper() in OPTION_LETTERS:
        return normalized[:1].upper()

    return ""


def normalize_answer_letter(answer: Any) -> str:
    letter = str(answer).strip().upper()
    if not letter:
        return ""
    first = letter[0]
    return first if first in OPTION_LETTERS else ""


def erqa_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, Any] | None = None) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", ANSWER_INSTRUCTION)
    question = str(doc.get("question", "")).strip()
    return f"{pre_prompt}{question}{post_prompt}"


def erqa_doc_to_target(doc: dict[str, Any]) -> str:
    return normalize_answer_letter(doc.get("answer", ""))


def erqa_doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    data_dir = get_erqa_data_dir(required=True)
    image_paths = doc.get("image_paths") or []
    if not image_paths:
        raise FileNotFoundError(
            f"ERQA sample {doc.get('id', '<unknown>')} has no image_paths. "
            "Re-run convert_erqa.py or fix erqa.jsonl."
        )

    images: list[Image.Image] = []
    for image_path in image_paths:
        resolved = _resolve_image_path(image_path, data_dir)
        images.append(Image.open(resolved).convert("RGB"))
    return images


def erqa_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, dict[str, Any]]:
    response = results[0] if results else ""
    pred_letter = extract_answer_letter(response)
    gold_letter = erqa_doc_to_target(doc)
    is_correct = bool(pred_letter) and pred_letter == gold_letter

    submission = {
        "id": doc.get("id"),
        "gt_content": gold_letter,
        "pred_parsed": pred_letter,
        "pred": response,
        "question_type": doc.get("question_type") or "unknown",
        "is_correct": is_correct,
    }
    return {"erqa_acc": submission}


def erqa_aggregate_results(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0

    sub_task_to_scores: dict[str, list[int]] = defaultdict(list)
    total_correct = 0

    for sample in results:
        sub_task = sample.get("question_type") or "unknown"
        is_correct = int(bool(sample.get("is_correct")))
        total_correct += is_correct
        sub_task_to_scores[sub_task].append(is_correct)

    accuracy = total_correct / len(results)
    sub_task_accuracies = {
        sub_task: sum(scores) / len(scores) if scores else 0.0 for sub_task, scores in sub_task_to_scores.items()
    }

    eval_logger.info("%-40s", "ERQA per-question_type accuracy")
    eval_logger.info("-" * 40)
    for sub_task, acc in sorted(sub_task_accuracies.items()):
        eval_logger.info("%-30s: %.4f (%d)", sub_task, acc, len(sub_task_to_scores[sub_task]))
    eval_logger.info("=" * 40)
    eval_logger.info("ERQA overall accuracy: %.4f (%d samples)", accuracy, len(results))

    return accuracy
