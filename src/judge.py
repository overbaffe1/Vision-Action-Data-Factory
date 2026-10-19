"""Эвристический судья для выбора лучших аннотаций — аналог QAOA ratchet.

Для каждого кадра генерируем N кандидатов (lottery), выбираем лучший по скоре.
Также фильтруем низкокачественные примеры (polish_worst).
"""
from __future__ import annotations

import logging
import re
from collections import Counter

logger = logging.getLogger("factory.judge")

SPATIAL_TERMS = ["left", "right", "above", "below", "inside", "next to", "behind", "front", "on top", "under", "holding", "touching", "near", "close"]
ACTION_VERBS = ["grasp", "hold", "move", "place", "pick", "push", "lift", "lower", "open", "close", "insert"]

def score_annotation(question: str, answer: str, instruction: str, bboxes: list | None = None) -> float:
    """0..1 — чем выше, тем лучше."""
    if not answer or len(answer.strip()) < 10:
        return 0.0
    score = 0.0
    # 1) length (0..1)
    length = len(answer)
    length_score = min(1.0, length / 300)  # 300 chars ~ ideal
    if length < 20:
        length_score *= 0.3
    if length > 800:
        length_score *= 0.8
    score += 0.15 * length_score

    # 2) spatial terms
    lower = answer.lower()
    spatial_count = sum(1 for term in SPATIAL_TERMS if term in lower)
    spatial_score = min(1.0, spatial_count / 3)
    score += 0.25 * spatial_score

    # 3) object diversity — сколько уникальных объектов упомянуто
    if bboxes is not None:
        labels = set(b.label.lower() for b in bboxes)
        mentioned = sum(1 for l in labels if l.lower() in lower)
        diversity = mentioned / max(1, len(labels))
        score += 0.20 * min(1.0, diversity * 1.2)
    else:
        # Fallback: считаем существительные
        words = re.findall(r"\b\w+\b", lower)
        diversity = len(set(words)) / max(1, len(words))
        score += 0.20 * min(1.0, diversity * 2)

    # 4) instruction coverage — ключевые слова инструкции в ответе
    instr_tokens = set(re.findall(r"\b\w+\b", instruction.lower()))
    # Убираем стоп-слова
    stop = {"the","a","an","and","or","to","in","on","of","is","are","be","for","with","this","that"}
    instr_tokens = instr_tokens - stop
    if instr_tokens:
        covered = sum(1 for t in instr_tokens if t in lower)
        coverage = covered / len(instr_tokens)
        score += 0.20 * coverage
    else:
        score += 0.10

    # 5) uniqueness — не generic фразы
    generic = ["image shows", "this is an image", "as an ai", "i cannot see"]
    if any(g in lower for g in generic):
        score *= 0.5
    # Запрещённые фразы из конфига
    # Уникальность: если ответ содержит конкретные координаты/числа — бонус
    if re.search(r"\[\s*\d+\s*,\s*\d+", answer):
        score += 0.05
    if re.search(r"\d+\s*%", answer):
        score += 0.05

    # Штраф за повторы
    sentences = [s.strip() for s in re.split(r"[.!?]+", answer) if s.strip()]
    if len(sentences) >= 2:
        uniq = len(set(sentences)) / len(sentences)
        score += 0.10 * uniq
    else:
        score += 0.05

    return min(1.0, max(0.0, score))


def judge_candidates(candidates: list[tuple[str, str, float | None]], instruction: str, bboxes: list | None = None) -> tuple[str, str, float]:
    """Выбирает лучшего кандидата. candidates: (q, a, precomputed_score|None)"""
    scored = []
    for q, a, s in candidates:
        if s is None:
            s = score_annotation(q, a, instruction, bboxes)
        scored.append((q, a, s))
    # Сортируем по score
    scored.sort(key=lambda x: x[2], reverse=True)
    best = scored[0]
    logger.debug(f"Judge выбрал лучший score {best[2]:.2f} из {len(scored)} кандидатов")
    return best


def filter_by_threshold(examples: list[dict], threshold: float = 0.4, instruction: str = "", bboxes: list | None = None) -> list[dict]:
    """Фильтрует примеры ниже порога."""
    kept = []
    for ex in examples:
        # ex имеет messages
        try:
            q = ex["messages"][0]["content"]
            a = ex["messages"][1]["content"]
            # Убираем префикс < image>
            q = q.replace("< image>", "").strip()
            s = score_annotation(q, a, instruction, bboxes)
            if s >= threshold:
                kept.append(ex)
            else:
                logger.debug(f"Отфильтрован пример score {s:.2f} < {threshold}: {q[:60]}")
        except Exception:
            kept.append(ex)
    return kept
