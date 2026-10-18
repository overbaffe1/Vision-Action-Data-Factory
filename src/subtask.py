"""Декомпозиция инструкций, прогресс, completion, error, recovery.

Цель: научить VLM понимать ход выполнения задачи — ключевой сигнал для VLA.
Без LLM — rule-based, детерминированный, offline.

QAOA-идеи:
- decomposition = семейство шаблонов (как family-сканы в QAOA)
- ratchet = выбираем лучшую декомпозицию по покрытию инструкции
- knn_warm = перенос subtask'ов по похожим инструкциям
"""
from __future__ import annotations

import logging
import re
from typing import List, Tuple

logger = logging.getLogger("factory.subtask")

# Глаголы и объекты для эвристик
ACTION_VERBS = ["pick", "place", "put", "move", "push", "pull", "open", "close", "grasp", "lift", "lower", "slide", "rotate", "press", "insert", "pour", "stack", "unstack", "fold", "wipe", "sweep", "cut"]
PREPOSITIONS = ["in", "on", "into", "onto", "inside", "under", "above", "next", "near", "beside", "behind", "front"]

# Типичные subtask шаблоны для манипуляций
TEMPLATE_SUBTASKS = {
    "pick_and_place": ["locate {obj}", "move gripper to {obj}", "grasp {obj}", "lift {obj}", "move to {target}", "release {obj}"],
    "push": ["locate {obj}", "move gripper near {obj}", "push {obj} to {target}"],
    "open": ["locate {obj}", "grasp handle of {obj}", "pull to open {obj}"],
    "close": ["locate {obj}", "push to close {obj}"],
    "stack": ["pick {obj1}", "place {obj1} on {obj2}"],
}

def _extract_objects(instruction: str) -> list[str]:
    # Простой парсер объектов: ищем существительные после глаголов/предлогов
    # Эвристика: берём чанки после \"the\" / \"a\" / прилагательных
    text = instruction.lower()
    # Удаляем пунктуацию
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    objects = []
    # Ищем \"the X\" или \"a X\" или \"Y block/bowl/cup/...\" 
    # Собираем биграммы/триграммы с цветами + объектами
    colors = ["red", "blue", "green", "yellow", "white", "black", "brown", "orange", "purple", "pink", "grey", "gray"]
    obj_nouns = ["block", "cube", "bowl", "cup", "plate", "bottle", "box", "drawer", "door", "handle", "knob", "slider", "lid", "container", "object", "item", "toy", "fruit", "apple", "banana", "soup", "basket", "tray", "table", "shelf", "cabinet", "microwave", "stove", "faucet", "cloth", "towel", "rag", "gripper", "hand"]
    tokens = text.split()
    for i, tok in enumerate(tokens):
        if tok in colors and i+1 < len(tokens) and tokens[i+1] in obj_nouns:
            objects.append(f"{tok} {tokens[i+1]}")
        elif tok in obj_nouns:
            # Проверяем цвет слева
            if i>0 and tokens[i-1] in colors:
                continue  # уже добавлен как цвет+объект
            objects.append(tok)
    # Дедуп
    seen = []
    for o in objects:
        if o not in seen:
            seen.append(o)
    return seen or ["object"]


def decompose_instruction(instruction: str) -> List[str]:
    """Rule-based декомпозиция инструкции на шаги."""
    text = instruction.strip()
    # 1) Разбиваем по \"and\", \"then\", \"and then\", \",\"
    parts = re.split(r"\b(?:and then|then|and|,)\b", text, flags=re.I)
    parts = [p.strip() for p in parts if p.strip()]
    # Если разбиение дало 1 часть — пробуем по глаголам
    if len(parts) == 1:
        # Ищем глаголы
        tokens = text.lower().split()
        verb_positions = [i for i, t in enumerate(tokens) if t in ACTION_VERBS]
        if len(verb_positions) >= 2:
            # Разбиваем по глаголам
            new_parts = []
            for idx, pos in enumerate(verb_positions):
                end = verb_positions[idx+1] if idx+1 < len(verb_positions) else len(tokens)
                chunk = " ".join(tokens[pos:end])
                new_parts.append(chunk)
            parts = new_parts
    # Нормализуем: каждый шаг — короткий инфинитив
    steps = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Убираем точку, приводим к повелительному
        p = re.sub(r"^(please\s+)", "", p, flags=re.I)
        # Делаем первую букву заглавной
        steps.append(p[0].upper() + p[1:] if len(p) > 1 else p.upper())
    # Fallback
    if not steps:
        steps = [text]
    # Если шагов слишком много (>6) — группируем
    if len(steps) > 6:
        steps = steps[:5] + ["; ".join(steps[5:])]
    return steps


def estimate_progress(frame_index: int, total_frames: int, subtask_idx: int | None = None, total_subtasks: int | None = None) -> Tuple[int, str]:
    """Оценка прогресса 0-100% + текстовая."""
    # Линейный прогресс по кадрам
    linear = int(round((frame_index + 1) / max(1, total_frames) * 100))
    # Если есть subtask информация — усредняем
    if subtask_idx is not None and total_subtasks:
        subtask_progress = int(round((subtask_idx + 1) / total_subtasks * 100))
        # Взвешенное: 70% линейный, 30% subtask
        combined = int(round(0.7 * linear + 0.3 * subtask_progress))
        return combined, f"Step {subtask_idx+1}/{total_subtasks} ({combined}%)"
    return linear, f"{linear}% complete"


def subtask_at_frame(instruction: str, frame_index: int, total_frames: int) -> Tuple[str, str | None, int, int]:
    """Возвращает (current_subtask, next_subtask, current_idx, total)."""
    steps = decompose_instruction(instruction)
    n = len(steps)
    # Маппим кадр на шаг линейно
    # Первый кадр — первый шаг, последний — последний
    pos = (frame_index + 0.5) / max(1, total_frames)
    idx = min(n - 1, int(pos * n))
    current = steps[idx]
    nxt = steps[idx + 1] if idx + 1 < n else None
    return current, nxt, idx, n


def completion_status(frame_index: int, total_frames: int, success: bool | None, subtask_idx: int | None = None, total_subtasks: int | None = None) -> Tuple[str, str]:
    """Определяет завершённость."""
    is_last = frame_index >= total_frames - 1
    if success is True:
        return "yes", "Task completed successfully, all subtasks finished."
    if success is False:
        return "no", "Task failed or not completed within the episode."
    # Эвристика по прогрессу
    if is_last:
        # Последний кадр — считаем почти завершённым, но не факт
        return "partial", "Episode ended, task may be partially completed. Check object positions."
    if subtask_idx is not None and total_subtasks:
        if subtask_idx >= total_subtasks - 1:
            return "yes", "Final subtask is in progress, task nearing completion."
        else:
            return "no", f"Task in progress, subtask {subtask_idx+1}/{total_subtasks} ongoing."
    progress = int(round((frame_index + 1) / max(1, total_frames) * 100))
    if progress > 85:
        return "partial", "Task nearing completion, final steps pending."
    return "no", "Task is still in progress."


def error_cause(frame_index: int, total_frames: int, instruction: str, success: bool | None) -> str | None:
    """Генерирует причину ошибки, если эпизод неуспешен."""
    if success is True or success is None:
        return None
    # Для неуспешных — шаблоны
    causes = [
        f"Gripper failed to grasp the target object for '{instruction}'. Object may have slipped or was out of reach.",
        f"Object dropped during transport. Trajectory shows sudden gripper opening before reaching target.",
        f"Collision with environment: gripper trajectory intersected with table or other object, causing displacement.",
        f"Target not reached: robot stopped short of destination, possibly due to planning error or joint limits.",
    ]
    # Выбираем по frame_index hash
    idx = (hash(instruction) + frame_index) % len(causes)
    return causes[idx]


def recovery_instruction(error: str | None, instruction: str) -> str | None:
    if error is None:
        return None
    # Шаблонные recovery
    recoveries = [
        "Re-attempt grasp: reposition gripper above object, close gripper firmly, verify grasp, then retry transport.",
        "Recover by picking up the dropped object: move gripper to object's new position, grasp again, and place to target.",
        "Back off slightly, replan collision-free path, then approach target from a different angle.",
        "Reset gripper position to initial state, re-localize target object, and retry the full subtask sequence.",
    ]
    idx = (hash(error) + hash(instruction)) % len(recoveries)
    return recoveries[idx]

