"""Выборка кадров из эпизода: uniform / keyframe / adaptive + polish_worst.

Идея из QAOA: polish_worst — дополнительно семплируем кадры в худших эпизодах (low quality),
а adaptive — аналог coordinate descent по дельте действия.
"""
from __future__ import annotations

import logging
import random
from typing import List

import numpy as np

from .dataset_loader import Episode, FrameInfo

logger = logging.getLogger("factory.sampler")


def _gripper_state(frame: FrameInfo) -> float | None:
    # state[6] или [7] часто — gripper open/close в LIBERO/Panda
    if frame.state is None:
        return None
    s = frame.state
    if len(s) >= 7:
        # эвристика: последний элемент state — gripper
        return float(s[-1])
    return None


def _action_delta(a: list[float] | None, b: list[float] | None) -> float:
    if a is None or b is None:
        return 0.0
    try:
        return float(np.linalg.norm(np.array(a) - np.array(b)))
    except Exception:
        return 0.0


def uniform_sampling(frames: List[FrameInfo], k: int, include_first_last: bool = True) -> List[FrameInfo]:
    n = len(frames)
    if n <= k:
        return frames
    if include_first_last:
        # Всегда берём первый и последний, остальные равномерно
        if k == 1:
            return [frames[0]]
        if k == 2:
            return [frames[0], frames[-1]]
        # k-2 внутренних
        step = (n - 1) / (k - 1)
        idxs = [0]
        for i in range(1, k - 1):
            idxs.append(int(round(i * step)))
        idxs.append(n - 1)
        # Убираем дубли
        idxs = sorted(set(idxs))
        # Если не хватило из-за дедупа — добиваем равномерными
        while len(idxs) < k:
            for c in range(n):
                if c not in idxs:
                    idxs.append(c)
                    break
            idxs = sorted(idxs)[:k]
        return [frames[i] for i in idxs]
    else:
        step = n / k
        idxs = [int(i * step) for i in range(k)]
        return [frames[i] for i in idxs]


def keyframe_sampling(frames: List[FrameInfo], k: int, threshold: float = 0.08) -> List[FrameInfo]:
    """Keyframe по дельте action + gripper."""
    n = len(frames)
    if n <= k:
        return frames
    # Считаем дельты
    deltas = [0.0] * n
    for i in range(1, n):
        d = _action_delta(frames[i].action, frames[i-1].action)
        # Усиливаем если gripper изменился
        g_prev = _gripper_state(frames[i-1])
        g_curr = _gripper_state(frames[i])
        if g_prev is not None and g_curr is not None:
            if abs(g_curr - g_prev) > 0.1:
                d += 1.0  # бонус
        deltas[i] = d
    # Пики дельт — кандидаты
    # Берём top-k по дельте + первый
    indexed = sorted(enumerate(deltas), key=lambda x: x[1], reverse=True)
    # Фильтр threshold
    candidates = [idx for idx, d in indexed if d >= threshold]
    if len(candidates) < k:
        candidates = [idx for idx, _ in indexed]
    # Всегда добавляем 0
    if 0 not in candidates[:k]:
        candidates = [0] + candidates
    # Берём k лучших, сортируем по времени
    selected = sorted(candidates[:k])
    # Дедуп и обрезка
    selected = sorted(set(selected))[:k]
    return [frames[i] for i in selected]


def adaptive_sampling(
    frames: List[FrameInfo],
    k: int,
    threshold: float = 0.08,
    include_first_last: bool = True,
    min_frames: int = 3,
    max_frames: int = 8,
) -> List[FrameInfo]:
    """Комбинация uniform + keyframe + gripper events (аналог QAOA merged2: ratchet лучшего)."""
    n = len(frames)
    if n <= k:
        return frames
    # 1) uniform половина
    k_uniform = max(1, k // 2)
    uniform = uniform_sampling(frames, k_uniform, include_first_last=include_first_last)
    uniform_idxs = {f.frame_index for f in uniform}
    # 2) keyframe вторая половина
    k_key = k - len(uniform)
    key = keyframe_sampling(frames, k_key + 5, threshold=threshold)  # берём с запасом
    key = [f for f in key if f.frame_index not in uniform_idxs]
    # Объединяем
    combined = uniform + key[:k_key]
    combined = sorted(combined, key=lambda f: f.frame_index)
    # Если не хватило — добиваем uniform
    if len(combined) < k:
        remaining = [f for f in frames if f not in combined]
        # Ближайшие к gaps
        combined += remaining[: k - len(combined)]
        combined = sorted(combined, key=lambda f: f.frame_index)
    # Обрезаем до max_frames / min_frames
    combined = combined[:max_frames]
    if len(combined) < min_frames and n >= min_frames:
        # добиваем
        extra = [f for f in frames if f not in combined]
        combined += extra[: min_frames - len(combined)]
        combined = sorted(combined, key=lambda f: f.frame_index)
    return combined[:k]


def sample_frames(
    episode: Episode,
    strategy: str = "adaptive",
    frames_per_episode: int = 5,
    uniform_stride: int = 6,
    keyframe_threshold: float = 0.08,
    min_frames: int = 3,
    max_frames: int = 8,
    include_first_last: bool = True,
    seed: int = 42,
) -> List[FrameInfo]:
    random.seed(seed + hash(episode.episode_id) % 100000)
    n = len(episode.frames)
    k = min(frames_per_episode, n)
    if strategy == "uniform":
        return uniform_sampling(episode.frames, k, include_first_last=include_first_last)
    elif strategy == "keyframe":
        return keyframe_sampling(episode.frames, k, threshold=keyframe_threshold)
    elif strategy == "adaptive":
        return adaptive_sampling(
            episode.frames, k, threshold=keyframe_threshold,
            include_first_last=include_first_last, min_frames=min_frames, max_frames=max_frames
        )
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")


def polish_worst_extra(
    episode: Episode,
    current_sample: List[FrameInfo],
    quality_scores: dict[int, float] | None,
    threshold: float = 0.35,
    extra_frames: int = 2,
) -> List[FrameInfo]:
    """Если качество эпизода низкое — добавляем кадры (аналог QAOA polish_worst.py)."""
    if quality_scores is None:
        return current_sample
    # Считаем средний score эпизода
    scores = [quality_scores.get(f.frame_index, 1.0) for f in current_sample]
    avg = sum(scores) / len(scores) if scores else 1.0
    if avg >= threshold:
        return current_sample
    # Нужны дополнительные кадры — берём gaps
    all_indices = {f.frame_index for f in episode.frames}
    cur_indices = {f.frame_index for f in current_sample}
    remaining = sorted(all_indices - cur_indices)
    if not remaining:
        return current_sample
    # Выбираем кадры в середине gaps
    # Сортируем по дистанции до ближайших выбранных
    def gap_score(idx: int) -> float:
        nearest = min(abs(idx - c) for c in cur_indices)
        return float(nearest)
    remaining = sorted(remaining, key=gap_score, reverse=True)
    extra = []
    for idx in remaining[:extra_frames]:
        extra.append(episode.frames[idx])
    result = sorted(current_sample + extra, key=lambda f: f.frame_index)
    logger.info(f"Polish worst {episode.episode_id}: avg_quality {avg:.2f} < {threshold}, +{len(extra)} frames")
    return result
