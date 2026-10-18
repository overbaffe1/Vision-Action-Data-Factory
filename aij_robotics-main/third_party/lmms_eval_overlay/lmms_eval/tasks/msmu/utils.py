"""MSMU-Bench (cpystan/MSMU) task utilities for lmms-eval.

Official evaluation reference: https://github.com/cpystan/SD-VLM
  - evaluation/inference.py: test split, conversations, type -> preds.json
  - evaluation/llm_generate.py: GPT-4-Turbo judging
  - evaluation/evaluate_final.py: per-type ratios; Average = sum(8 category scores) / 8

Deterministic local scoring for standard VLMs. Scores differ from official GPT-judged MSMU-Bench.
"""

from __future__ import annotations

import json
import logging
import re
import string
from collections import defaultdict
from pathlib import Path
from typing import Any

import datasets
from PIL import Image

eval_logger = logging.getLogger("lmms-eval")

MSMU_EVAL_SPLIT = "test"
MSMU_DATASET_ID = "cpystan/MSMU"

# HF test `type` values (verified on cpystan/MSMU test split, n=987).
DATASET_TYPE_TO_CANONICAL: dict[str, str] = {
    "existence": "existence",
    "count": "object_counting",
    "scale_estimation": "scale_estimation",
    "grounding": "grounding",
    "relative_position": "relative_position",
    "absolute_distance": "absolute_distance",
    "scale_compare": "scale_comparison",
    "refer_obj_estimation": "reference_object_estimation",
}

CANONICAL_TO_DATASET_TYPE: dict[str, str] = {v: k for k, v in DATASET_TYPE_TO_CANONICAL.items()}

# Hugging Face leaderboard column order (8 categories + Average).
MSMU_CANONICAL_CATEGORIES: tuple[str, ...] = (
    "existence",
    "object_counting",
    "scale_estimation",
    "grounding",
    "relative_position",
    "absolute_distance",
    "scale_comparison",
    "reference_object_estimation",
)

# SD-VLM evaluation/llm_generate.py type lists (raw dataset names).
QUANTITATIVE_TYPES = frozenset(
    {"scale_estimation", "absolute_distance", "count", "grounding", "refer_obj_estimation"}
)
QUALITATIVE_TYPES = frozenset({"relative_position", "scale_compare", "existence"})

OFFICIAL_RATIO_THRESHOLD = 1.25
ABS_TOL = 1e-3
REL_TOL = 0.05

_NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_OPTION_LETTERS = frozenset("ABCD")

_TEXT_PROMPT_SUFFIX = (
    "\nAnswer the question based only on the image. Give a concise answer."
)
_NUMERIC_PROMPT_SUFFIX = (
    "\nAnswer the question based only on the image. "
    "If the answer is numeric, output only the number and unit if necessary."
)

_AFFIRMATIVE = frozenset({"yes", "yeah", "yep", "correct", "true", "indeed"})
_NEGATIVE = frozenset({"no", "none", "not", "without", "missing", "absent", "n/a"})


def map_dataset_type_to_canonical(dataset_type: str | None) -> str | None:
    if dataset_type is None:
        return None
    return DATASET_TYPE_TO_CANONICAL.get(str(dataset_type))


def _strip_image_token(text: str) -> str:
    text = str(text).strip()
    if text.startswith("<image>"):
        text = text[len("<image>") :].lstrip("\n")
    return text.strip()


def parse_msmu_conversation(doc: dict[str, Any]) -> tuple[str, str]:
    conversations = doc.get("conversations")
    if not conversations:
        raise ValueError(f"MSMU sample {doc.get('doc_id')} missing conversations")

    if isinstance(conversations, dict):
        values = conversations.get("value") or []
    elif isinstance(conversations, list):
        values = [turn.get("value", "") for turn in conversations]
    else:
        raise ValueError(f"Unsupported conversations type: {type(conversations)}")

    if len(values) < 2:
        raise ValueError(f"MSMU sample {doc.get('doc_id')} needs human+gpt turns, got {len(values)}")

    return _strip_image_token(values[0]), str(values[1]).strip()


def msmu_process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _map_doc(doc: dict[str, Any], idx: int) -> dict[str, Any]:
        question, answer = parse_msmu_conversation(doc)
        raw_type = doc.get("type")
        raw_type_str = str(raw_type) if raw_type is not None else "unknown"
        canonical = map_dataset_type_to_canonical(raw_type_str if raw_type is not None else None)

        if raw_type is not None and canonical is None:
            eval_logger.warning("MSMU unknown dataset type: %s (doc_id=%s)", raw_type_str, idx)

        return {
            "doc_id": str(idx),
            "question": question,
            "answer": answer,
            "task_type": raw_type_str,
            "canonical_category": canonical or "unknown",
            "is_quantitative": raw_type_str in QUANTITATIVE_TYPES,
            "is_qualitative": raw_type_str in QUALITATIVE_TYPES,
        }

    return dataset.map(_map_doc, with_indices=True)


def msmu_doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    image = doc.get("image")
    if image is None:
        raise FileNotFoundError(f"MSMU sample {doc.get('doc_id')} has no image.")
    return [image.convert("RGB")]


def msmu_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, Any] | None = None) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")

    question = str(doc.get("question", "")).strip()
    task_type = str(doc.get("task_type", ""))

    if post_prompt:
        suffix = post_prompt
    elif task_type in QUANTITATIVE_TYPES:
        suffix = _NUMERIC_PROMPT_SUFFIX
    else:
        suffix = _TEXT_PROMPT_SUFFIX

    return f"{pre_prompt}Question: {question}{suffix}"


def msmu_doc_to_target(doc: dict[str, Any]) -> str:
    return str(doc.get("answer", "")).strip()


def normalize_text(text: str) -> str:
    normalized = str(text).strip().lower()
    normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    return " ".join(normalized.split())


def extract_all_numbers(text: str) -> list[float]:
    numbers: list[float] = []
    for match in _NUMBER_PATTERN.findall(str(text)):
        try:
            numbers.append(float(match))
        except ValueError:
            continue
    return numbers


def extract_option_letter(text: str) -> str:
    normalized = str(text).strip()
    if not normalized:
        return ""
    patterns = [
        re.compile(r"answer\s*(?:is|:|-)\s*\(?([A-D])\)?\b", re.IGNORECASE),
        re.compile(r"\(([A-D])\)", re.IGNORECASE),
        re.compile(r"^[\(\s]*([A-D])[\)\.\s]*$", re.IGNORECASE),
        re.compile(r"\b([A-D])\b", re.IGNORECASE),
    ]
    for pattern in patterns:
        match = pattern.search(normalized)
        if match:
            letter = match.group(1).upper()
            if letter in _OPTION_LETTERS:
                return letter
    return ""


def official_ratio_match(pred_numbers: list[float], gt_numbers: list[float]) -> bool:
    """SD-VLM quantitative δ1: all pairwise ratios < 1.25 (evaluate_final.py)."""
    if not pred_numbers or not gt_numbers:
        return False
    n = min(len(pred_numbers), len(gt_numbers))
    for i in range(n):
        pred, gt = pred_numbers[i], gt_numbers[i]
        if abs(gt) <= ABS_TOL and abs(pred) <= ABS_TOL:
            continue
        if abs(gt) <= ABS_TOL or abs(pred) <= ABS_TOL:
            return False
        ratio = max(pred / gt, gt / pred)
        if ratio >= OFFICIAL_RATIO_THRESHOLD:
            return False
    return True


def numbers_match_relaxed(pred: float, gt: float) -> bool:
    if abs(pred - gt) <= ABS_TOL:
        return True
    denom = max(abs(gt), ABS_TOL)
    return abs(pred - gt) / denom <= REL_TOL


def _numeric_match(pred: str, tgt: str) -> bool:
    pred_numbers = extract_all_numbers(pred)
    gt_numbers = extract_all_numbers(tgt)
    if not pred_numbers or not gt_numbers:
        if len(pred_numbers) == 1 and len(gt_numbers) == 1:
            return numbers_match_relaxed(pred_numbers[0], gt_numbers[0])
        return False
    if official_ratio_match(pred_numbers, gt_numbers):
        return True
    if len(pred_numbers) == 1 and len(gt_numbers) == 1:
        return numbers_match_relaxed(pred_numbers[0], gt_numbers[0])
    return False


def _qualitative_match(pred: str, gt: str, task_type: str) -> bool:
    pred_norm = normalize_text(pred)
    gt_norm = normalize_text(gt)
    if not pred_norm or not gt_norm:
        return False

    if pred_norm == gt_norm:
        return True

    pred_tokens = set(pred_norm.split())
    gt_tokens = set(gt_norm.split())
    if pred_tokens & _AFFIRMATIVE and (gt_tokens & _AFFIRMATIVE or gt_norm.startswith("indeed")):
        return True
    if (pred_tokens & _NEGATIVE or "not" in pred_norm) and (gt_tokens & _NEGATIVE or gt_norm.startswith("no ")):
        return True

    if pred_norm in gt_norm or gt_norm in pred_norm:
        return True

    if pred_tokens:
        overlap = len(pred_tokens & gt_tokens) / max(len(pred_tokens), 1)
        if overlap >= 0.5:
            return True

    if task_type == "scale_compare":
        for marker in ("is taller", "is lower", "is closer", "is further", "is shortest", "is highest", "is lowest"):
            if marker in gt_norm:
                before = gt_norm.split(marker)[0].strip()
                object_hint = " ".join(before.split()[-4:])
                if object_hint and object_hint in pred_norm:
                    return True

    if task_type == "relative_position":
        for obj in ("office chair", "cluttered shelf", "bookshelf", "table", "curtain", "rack"):
            if obj in gt_norm and obj in pred_norm:
                if ("closer" in gt_norm and "closer" in pred_norm) or ("further" in gt_norm and "further" in pred_norm):
                    return True

    return False


def score_prediction(prediction: str, target: str, task_type: str | None = None) -> dict[str, Any]:
    pred = str(prediction).strip()
    tgt = str(target).strip()
    task_type = task_type or "unknown"

    pred_norm = normalize_text(pred)
    tgt_norm = normalize_text(tgt)
    exact_match = bool(pred_norm) and bool(tgt_norm) and pred_norm == tgt_norm

    pred_numbers = extract_all_numbers(pred)
    gt_numbers = extract_all_numbers(tgt)
    numeric_scorable = bool(pred_numbers) and bool(gt_numbers)

    pred_letter = extract_option_letter(pred)
    tgt_letter = extract_option_letter(tgt)

    is_correct = False
    match_mode = "exact"

    if task_type in QUANTITATIVE_TYPES:
        if numeric_scorable or (pred_numbers and gt_numbers):
            is_correct = _numeric_match(pred, tgt)
            match_mode = "quantitative_ratio"
        elif pred_letter and tgt_letter:
            is_correct = pred_letter == tgt_letter
            match_mode = "multiple_choice"
        else:
            is_correct = exact_match
            match_mode = "exact"
    elif task_type in QUALITATIVE_TYPES:
        is_correct = _qualitative_match(pred, tgt, task_type)
        match_mode = "qualitative"
    elif pred_letter and tgt_letter:
        is_correct = pred_letter == tgt_letter
        match_mode = "multiple_choice"
    elif _numeric_match(pred, tgt):
        is_correct = True
        match_mode = "numeric_fallback"
    else:
        is_correct = exact_match
        match_mode = "exact"

    return {
        "is_correct": is_correct,
        "exact_match": exact_match,
        "numeric_scorable": numeric_scorable,
        "match_mode": match_mode,
        "pred_normalized": pred_norm,
        "target_normalized": tgt_norm,
        "pred_numbers": pred_numbers,
        "gt_numbers": gt_numbers,
    }


def _build_result_entry(doc: dict[str, Any], prediction: str, scored: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": doc.get("doc_id"),
        "question": doc.get("question"),
        "task_type": doc.get("task_type") or "unknown",
        "canonical_category": doc.get("canonical_category") or "unknown",
        "gt_content": msmu_doc_to_target(doc),
        "pred": prediction,
        "pred_normalized": scored["pred_normalized"],
        "target_normalized": scored["target_normalized"],
        "is_correct": scored["is_correct"],
        "exact_match": scored["exact_match"],
        "match_mode": scored["match_mode"],
    }


def msmu_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, Any]:
    prediction = results[0] if results else ""
    scored = score_prediction(prediction, msmu_doc_to_target(doc), task_type=str(doc.get("task_type", "")))
    entry = _build_result_entry(doc, prediction, scored)

    out: dict[str, Any] = {
        "msmu_overall_accuracy": entry,
        "msmu_average": entry,
    }

    canonical = doc.get("canonical_category")
    if canonical in MSMU_CANONICAL_CATEGORIES:
        out[f"msmu_{canonical}"] = entry

    return out


def _mean_bool(results: list[dict[str, Any]], key: str) -> float:
    if not results:
        return 0.0
    return sum(1.0 if r.get(key) else 0.0 for r in results) / len(results)


def _category_scores_from_results(results: list[dict[str, Any]]) -> dict[str, float]:
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for sample in results:
        cat = sample.get("canonical_category")
        if cat in MSMU_CANONICAL_CATEGORIES:
            by_cat[cat].append(bool(sample.get("is_correct")))
    return {cat: (sum(v) / len(v) if v else 0.0) for cat, v in by_cat.items()}


def _log_category_breakdown(results: list[dict[str, Any]]) -> None:
    scores = _category_scores_from_results(results)
    eval_logger.info("%-40s", "MSMU per-category accuracy (deterministic, HF names)")
    eval_logger.info("-" * 40)
    for cat in MSMU_CANONICAL_CATEGORIES:
        if cat in scores:
            n = sum(1 for r in results if r.get("canonical_category") == cat)
            eval_logger.info("%-30s: %.4f (%d)", cat, scores[cat], n)
        else:
            eval_logger.info("%-30s: (no samples in run)", cat)
    eval_logger.info("=" * 40)


def msmu_aggregate_overall_accuracy(results: list[dict[str, Any]]) -> float:
    """Micro-average: fraction correct over all evaluated samples."""
    acc = _mean_bool(results, "is_correct")
    _log_category_breakdown(results)
    eval_logger.info(
        "MSMU overall (micro): %.4f (%d samples). Official MSMU-Bench uses GPT-4 judging (SD-VLM).",
        acc,
        len(results),
    )
    return acc


def msmu_aggregate_average(results: list[dict[str, Any]]) -> float:
    """
    Macro-average over MSMU categories (HF leaderboard / SD-VLM evaluate_final.py style).

    On the full test split all 8 categories are present; average divides by 8.
    With --limit, only categories present in the run are averaged (logged as partial).
    """
    scores = _category_scores_from_results(results)
    if not scores:
        eval_logger.warning("MSMU average: no scorable category samples.")
        return 0.0

    if len(scores) == len(MSMU_CANONICAL_CATEGORIES):
        macro = sum(scores[cat] for cat in MSMU_CANONICAL_CATEGORIES) / 8.0
        eval_logger.info("MSMU average (macro, 8 categories): %.4f", macro)
        return macro

    macro = sum(scores.values()) / len(scores)
    eval_logger.warning(
        "MSMU average (partial macro): %.4f over %d/%d categories (--limit or filtered run). "
        "Full benchmark uses 8 categories like SD-VLM evaluate_final.py.",
        macro,
        len(scores),
        len(MSMU_CANONICAL_CATEGORIES),
    )
    return macro


def _aggregate_by_canonical(results: list[dict[str, Any]], canonical: str) -> float:
    subset = [r for r in results if r.get("canonical_category") == canonical]
    if not subset:
        return 0.0
    return _mean_bool(subset, "is_correct")


def msmu_aggregate_existence(results: list[dict[str, Any]]) -> float:
    return _aggregate_by_canonical(results, "existence")


def msmu_aggregate_object_counting(results: list[dict[str, Any]]) -> float:
    return _aggregate_by_canonical(results, "object_counting")


def msmu_aggregate_scale_estimation(results: list[dict[str, Any]]) -> float:
    return _aggregate_by_canonical(results, "scale_estimation")


def msmu_aggregate_grounding(results: list[dict[str, Any]]) -> float:
    return _aggregate_by_canonical(results, "grounding")


def msmu_aggregate_relative_position(results: list[dict[str, Any]]) -> float:
    return _aggregate_by_canonical(results, "relative_position")


def msmu_aggregate_absolute_distance(results: list[dict[str, Any]]) -> float:
    return _aggregate_by_canonical(results, "absolute_distance")


def msmu_aggregate_scale_comparison(results: list[dict[str, Any]]) -> float:
    return _aggregate_by_canonical(results, "scale_comparison")


def msmu_aggregate_reference_object_estimation(results: list[dict[str, Any]]) -> float:
    return _aggregate_by_canonical(results, "reference_object_estimation")


def build_sdvlm_preds_record(doc: dict[str, Any], prediction: str) -> dict[str, Any]:
    question = str(doc.get("question", ""))
    raw_type = doc.get("task_type")
    if raw_type in (None, "unknown") and doc.get("canonical_category") in CANONICAL_TO_DATASET_TYPE:
        raw_type = CANONICAL_TO_DATASET_TYPE[doc["canonical_category"]]
    return {
        "question": f"<image>\n{question}",
        "pred": prediction,
        "gt": msmu_doc_to_target(doc),
        "type": raw_type,
        "doc_id": doc.get("doc_id"),
    }


def export_samples_to_sdvlm_preds(
    samples: list[dict[str, Any]],
    output_path: str | Path,
    *,
    model_name: str = "lmms-eval",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"model": model_name}) + "\n")
        for row in samples:
            meta = row.get("msmu_overall_accuracy") or row.get("msmu_average") or {}
            doc = {
                "doc_id": row.get("doc_id"),
                "question": row.get("input", ""),
                "answer": row.get("target"),
                "task_type": meta.get("task_type"),
                "canonical_category": meta.get("canonical_category"),
            }
            pred = row.get("filtered_resps") or meta.get("pred", "")
            if isinstance(pred, list):
                pred = pred[0] if pred else ""
            if "Question:" in str(doc["question"]):
                doc["question"] = str(doc["question"]).split("Question:", 1)[-1].strip()
            record = build_sdvlm_preds_record(doc, str(pred))
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path
