"""Утилиты."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


def set_seed(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def image_hash(path: Path) -> str:
    return hashlib.md5(str(path).encode()).hexdigest()[:8]


def get_image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size  # (w,h)
    except Exception:
        return 640, 480


def make_sharegpt_record(question: str, answer: str, image_path: Path, relative: bool = False, media_dir: Path | None = None) -> dict:
    """Формирует ShareGPT запись."""
    if relative and media_dir is not None:
        try:
            rel = image_path.relative_to(media_dir)
            img_str = str(rel)
        except ValueError:
            # если не внутри media_dir — basename
            img_str = image_path.name
    else:
        # Абсолютный путь — как в примере, но лучше относительный
        img_str = str(image_path.resolve()) if not relative else str(image_path)
    return {
        "messages": [
            {"role": "user", "content": f"< image>\n{question.strip()}"},
            {"role": "assistant", "content": answer.strip()},
        ],
        "images": [img_str],
    }


def dedup_by_hash(records: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in records:
        try:
            q = r["messages"][0]["content"]
            img = r["images"][0]
            key = hashlib.md5(f"{q}|{img}".encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        except Exception:
            out.append(r)
    return out
