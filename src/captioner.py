"""Генерация подписей сцены — темплейты + опциональный локальный LLM.

Для VLM-бенчей (BLINK, MMMU, ERQA) нужны разнородные описания:
- hands, objects, spatial layout, colors, counts, actions.

QAOA-параллель: как в lottery.py — множество шаблонов, из них ratchet выбирает лучший.
"""
from __future__ import annotations

import logging
import random
import re

from .spatial import BBox

logger = logging.getLogger("factory.captioner")


TEMPLATE_CAPTIONS = [
    # Robot-centric
    "The robot gripper is positioned {gripper_pos}. {objects_desc} The task is: {instruction}.",
    "A robotic arm operates in a tabletop scene. {objects_desc} Gripper state: {gripper_state}. Instruction: {instruction}.",
    # Egocentric / hands
    "A person's hands are visible in {hands_pos}. {objects_desc} The person is {action}.",
    # Scene
    "The scene shows {objects_count}. {spatial_desc} Overall setting: {environment}.",
]

ENVIRONMENTS = ["tabletop manipulation setup", "kitchen-like workspace", "indoor tabletop with varied objects", "robot workspace with camera view"]

def _describe_objects(bboxes: list[BBox], instruction: str) -> str:
    if not bboxes:
        return "No distinct objects detected, the view is mostly background or the gripper."
    # Группируем по лейблам
    from collections import Counter
    cnt = Counter(b.label for b in bboxes)
    parts = []
    for label, c in cnt.most_common(4):
        if c == 1:
            parts.append(f"one {label}")
        else:
            parts.append(f"{c} {label}s")
    return f"Visible objects include {', '.join(parts)}. " + f"Instruction context: {instruction}"

def _spatial_to_sentence(spatial_rels: list[str]) -> str:
    if not spatial_rels:
        return "Spatial layout is not clearly determined."
    # Берём 2-3 случайных
    sel = spatial_rels[:3]
    return " ".join(s.capitalize() + "." for s in sel)

def _gripper_position(bboxes: list[BBox], width: int, height: int) -> str:
    gripper = [b for b in bboxes if "gripper" in b.label.lower() or "hand" in b.label.lower()]
    if not gripper:
        return "near the center of the view"
    g = gripper[0]
    cx = g.cx / width
    cy = g.cy / height
    pos = []
    if cx < 0.33:
        pos.append("left")
    elif cx > 0.66:
        pos.append("right")
    else:
        pos.append("center")
    if cy < 0.33:
        pos.append("top")
    elif cy > 0.66:
        pos.append("bottom")
    else:
        pos.append("middle")
    return f"on the {'-'.join(pos)} ({int(g.cx)}, {int(g.cy)})"

def generate_caption(
    bboxes: list[BBox],
    spatial_rels: list[str],
    instruction: str,
    width: int,
    height: int,
    progress_text: str | None = None,
    subtask: str | None = None,
    seed: int = 42,
    mode: str = "template",
) -> str:
    """Генерирует 5-7 предложений (требование ESPADA general_captioning)."""
    random.seed(seed)
    gripper_pos = _gripper_position(bboxes, width, height)
    objects_desc = _describe_objects(bboxes, instruction)
    spatial_desc = _spatial_to_sentence(spatial_rels)
    gripper_state = "gripper appears to be near an object" if any("gripper" in b.label.lower() for b in bboxes) else "gripper not clearly visible"
    environment = random.choice(ENVIRONMENTS)
    objects_count = f"{len(bboxes)} objects are visible" if bboxes else "the scene is sparse"
    hands_pos = gripper_pos
    action = subtask.lower() if subtask else "manipulating objects on the table"

    # Выбираем темплейт
    tpl = random.choice(TEMPLATE_CAPTIONS)
    caption = tpl.format(
        gripper_pos=gripper_pos,
        objects_desc=objects_desc,
        instruction=instruction,
        gripper_state=gripper_state,
        hands_pos=hands_pos,
        action=action,
        objects_count=objects_count,
        spatial_desc=spatial_desc,
        environment=environment,
    )
    # Добавляем прогресс если есть
    if progress_text:
        caption += f" Task progress: {progress_text}."
    if subtask:
        caption += f" Current subtask: {subtask}."
    # Нормализуем длину 5-7 предложений: считаем предложения
    sentences = [s.strip() for s in re.split(r"\.\s+", caption) if s.strip()]
    # Если меньше 5 — добавляем spatial
    while len(sentences) < 5 and spatial_rels:
        sentences.append(spatial_rels[len(sentences) % len(spatial_rels)].capitalize())
    # Если больше 7 — режем
    sentences = sentences[:7]
    caption = ". ".join(sentences)
    if not caption.endswith("."):
        caption += "."
    return caption


def generate_qa_variants(
    bboxes: list[BBox],
    spatial_rels: list[str],
    instruction: str,
    width: int,
    height: int,
    caption: str,
    progress: str,
    subtask_current: str,
    subtask_next: str | None,
    seed: int = 0,
) -> list[tuple[str, str]]:
    """Генерирует QA пары для VLM-бенчей — многообразие как в QAOA lottery."""
    random.seed(seed)
    qas: list[tuple[str, str]] = []

    # 1. Scene description QA (для MSC/MSMU)
    qas.append((
        "Describe the scene in detail. Focus on objects, their states, and spatial relations for robot action planning.",
        caption
    ))
    # 2. Counting
    from collections import Counter
    cnt = Counter(b.label for b in bboxes)
    for label, c in list(cnt.items())[:2]:
        qas.append((f"How many {label}s are visible? Count and list their approximate positions.", f"There are {c} {label}(s). Positions: " + "; ".join(f"[{b.x_min},{b.y_min},{b.x_max},{b.y_max}]" for b in bboxes if b.label==label)))
    # 3. Spatial
    for rel in spatial_rels[:2]:
        qas.append((f"Is it true that {rel}? Explain based on the image.", f"Yes, {rel}. This can be verified by observing the relative positions of the objects in the image."))
    # 4. Subtask
    qas.append((f"What subtask is the robot currently performing? Instruction: {instruction}", f"The robot is currently: {subtask_current}. Progress: {progress}."))
    if subtask_next:
        qas.append((f"What should the robot do next to complete '{instruction}'?", f"Next subtask: {subtask_next}. The robot should execute this after completing the current step."))
    # 5. Progress
    qas.append((f"How far has the robot progressed in the task '{instruction}'? Estimate percentage and explain.", f"Progress: {progress}. Estimated from frame position and subtask completion."))
    # 6. Completion
    qas.append((f"Has the task '{instruction}' been completed successfully? Provide reasoning.", "Task is in progress, not yet completed. Gripper still engaged and objects not in final positions." if "in progress" in progress.lower() else "Task appears completed."))
    # 7. Action prediction (для VLA)
    qas.append((f"What action should the robot take next to progress toward '{instruction}'?", f"Next action: {subtask_next or 'verify completion'}. Move gripper accordingly with appropriate open/close."))

    # 8. Temporal (Egocentric)
    qas.append(("What happened just before this moment? Infer from the scene configuration.", f"Prior step was likely {subtask_current}. Gripper moved to current position as part of '{instruction}'."))
    qas.append(("What will happen immediately after this frame?", f"Next: {subtask_next or 'task completion check'}."))

    # 9. Error/recovery (если есть)
    # Будет добавлено отдельно в subtask.py

    # 10. Grounding-style
    if bboxes:
        labels = sorted(set(b.label for b in bboxes))
        qas.append((f"Locate every {', '.join(labels)} in the image. Return the answer in JSON.", str([{"bbox_2d":[b.x_min,b.y_min,b.x_max,b.y_max],"label":b.label} for b in bboxes])))
        qas.append((f"Point to every {labels[0]} in the image. Return the answer in JSON.", str([{"point_2d":[int(b.cx),int(b.cy)],"label":b.label} for b in bboxes if b.label==labels[0]])))

    return qas
