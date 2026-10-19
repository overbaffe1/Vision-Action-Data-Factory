"""VLM Judge for BPC pipeline: refines labels for bboxing and pointing.

The Judge may ONLY edit text label fields. It must NOT change coordinates.
After receiving Judge output we verify coordinate immutability and reject the
result if any coordinate was modified.
"""
from __future__ import annotations

import logging
import math
from copy import deepcopy
from typing import Any

from .config import JudgeConfig
from .validation import ValidationError


logger = logging.getLogger("bpc.judge")


class CoordinateChangedError(ValidationError):
    """Raised when Judge output contains modified coordinates."""


def _coords_equal(a: Any, b: Any) -> bool:
    """Numerically compares two coordinate values (int or float)."""
    if type(a) != type(b):
        # Allow int/float equivalence: 10 == 10.0
        try:
            return math.isclose(float(a), float(b), rel_tol=0, abs_tol=0)
        except (TypeError, ValueError):
            return False
    if isinstance(a, float):
        return math.isclose(a, b, rel_tol=0, abs_tol=0)
    return a == b


def _verify_bbox_coords_unchanged(original: list[dict], judged: list[dict]) -> None:
    """Checks that all bbox_2d coordinates are identical between original and judged."""
    if len(original) != len(judged):
        raise CoordinateChangedError(
            f"Judge changed number of objects: {len(original)} -> {len(judged)}."
        )
    for i, (orig, jdg) in enumerate(zip(original, judged)):
        orig_bbox = orig.get("bbox_2d", [])
        jdg_bbox = jdg.get("bbox_2d", [])
        if len(orig_bbox) != len(jdg_bbox):
            raise CoordinateChangedError(
                f"Annotation [{i}] bbox_2d length changed: {orig_bbox} -> {jdg_bbox}."
            )
        for j, (oc, jc) in enumerate(zip(orig_bbox, jdg_bbox)):
            if not _coords_equal(oc, jc):
                raise CoordinateChangedError(
                    f"Annotation [{i}] bbox_2d[{j}] changed: {oc} -> {jc}."
                )


def _verify_pointing_coords_unchanged(original: list[dict], judged: list[dict]) -> None:
    """Checks that all point_2d coordinates are identical between original and judged."""
    if len(original) != len(judged):
        raise CoordinateChangedError(
            f"Judge changed number of points: {len(original)} -> {len(judged)}."
        )
    for i, (orig, jdg) in enumerate(zip(original, judged)):
        orig_pt = orig.get("point_2d", [])
        jdg_pt = jdg.get("point_2d", [])
        if len(orig_pt) != len(jdg_pt):
            raise CoordinateChangedError(
                f"Point [{i}] point_2d length changed: {orig_pt} -> {jdg_pt}."
            )
        for j, (oc, jc) in enumerate(zip(orig_pt, jdg_pt)):
            if not _coords_equal(oc, jc):
                raise CoordinateChangedError(
                    f"Point [{i}] point_2d[{j}] changed: {oc} -> {jc}."
                )


def verify_judge_output(
    original_result: dict[str, Any],
    judged_result: dict[str, Any],
    task_name: str,
) -> None:
    """Verifies that judge did not alter coordinates.

    Raises CoordinateChangedError if any coordinate was modified.
    """
    orig_answer = original_result.get("answer", [])
    jdg_answer = judged_result.get("answer", [])

    if not isinstance(orig_answer, list) or not isinstance(jdg_answer, list):
        raise CoordinateChangedError("answer must be a list for coordinate verification.")

    if task_name == "bboxing":
        _verify_bbox_coords_unchanged(orig_answer, jdg_answer)
    elif task_name == "pointing":
        _verify_pointing_coords_unchanged(orig_answer, jdg_answer)
    else:
        raise ValueError(f"Judge not supported for task '{task_name}'.")


def build_judge_prompt(
    task_name: str,
    original_result: dict[str, Any],
    prompts: dict[str, Any],
) -> tuple[str, str]:
    """Builds system/user prompts for the VLM Judge.

    The output contract MUST match ``build_detection_result`` in src/utils.py,
    which expects a JSON object with "grounding_question" and "grounding_answer"
    keys (the same shape the base bboxing/pointing prompts already produce).
    Judge output goes through the identical parser, so the prompt must ask for
    exactly this shape — an "answer"-shaped or free-text reply fails parsing.
    """
    from ..vlm.prompts_processing import get_prompt_block, render_template

    block_path = f"judge.{task_name}"
    try:
        block = get_prompt_block(prompts, block_path)
    except (KeyError, TypeError):
        # Fallback: generic judge prompt matching the grounding_question/
        # grounding_answer contract expected by build_detection_result().
        import json as _json

        coord_key = "bbox_2d" if task_name == "bboxing" else "point_2d"
        answer_json = _json.dumps(original_result.get("answer", []), ensure_ascii=False)

        system = (
            "You are a quality-control judge for spatial annotations. "
            "You may ONLY improve the 'label' text field of each item. "
            f"You must NOT change, add, remove, or reorder any '{coord_key}' "
            "coordinates or items. "
            "Return STRICT JSON with EXACTLY these two top-level keys: "
            '"grounding_question" (string, restate the task) and '
            '"grounding_answer" (a list with the exact same items and '
            f"'{coord_key}' values as the input, only 'label' may change). "
            "Return ONLY the JSON object, no explanation, no markdown."
        )
        user = (
            f"Review and improve label quality for this {task_name} annotation. "
            f"Input items (grounding_answer):\n{answer_json}\n\n"
            "Return JSON: "
            '{"grounding_question": "<restate the task briefly>", '
            f'"grounding_answer": <the same list, same {coord_key} values, '
            'only "label" text may be improved>}'
        )
        return system, user

    ctx = {"original": original_result}
    system = render_template(block["system"], ctx)
    user = render_template(block["user"], ctx)
    return system, user
