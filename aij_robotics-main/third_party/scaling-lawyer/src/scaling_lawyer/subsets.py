from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable

import numpy as np


def nested_indices(total: int, count: int, seed: int) -> list[int]:
    if total < 1:
        raise ValueError("total must be positive.")
    if not 1 <= count <= total:
        raise ValueError(f"count must be in [1, {total}], got {count}.")
    order = np.random.default_rng(seed).permutation(total)
    return [int(index) for index in order[:count]]


def task_balanced_episode_order(
    episode_tasks: Iterable[tuple[int, str]],
    seed: int,
) -> list[int]:
    groups: dict[str, list[int]] = defaultdict(list)
    seen = set()
    for episode_index, task in episode_tasks:
        if episode_index in seen:
            raise ValueError(f"Duplicate episode index: {episode_index}")
        seen.add(episode_index)
        groups[str(task)].append(int(episode_index))
    if not groups:
        raise ValueError("No episodes were provided.")

    rng = np.random.default_rng(seed)
    shuffled: dict[str, deque[int]] = {}
    for task, indices in groups.items():
        values = np.asarray(sorted(indices), dtype=np.int64)
        rng.shuffle(values)
        shuffled[task] = deque(int(value) for value in values)

    task_order = sorted(shuffled)
    rng.shuffle(task_order)
    result = []
    while any(shuffled.values()):
        for task in task_order:
            if shuffled[task]:
                result.append(shuffled[task].popleft())
    return result


def select_task_balanced_episodes(
    episode_tasks: Iterable[tuple[int, str]],
    count: int,
    seed: int,
) -> list[int]:
    order = task_balanced_episode_order(episode_tasks, seed)
    if not 1 <= count <= len(order):
        raise ValueError(f"count must be in [1, {len(order)}], got {count}.")
    return order[:count]
