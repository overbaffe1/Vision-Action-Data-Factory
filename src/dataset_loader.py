"""Загрузка робо-датасетов: LeRobot v3 + generic image folders + Egocentric.

Поддерживает:
- BridgeData2 / Fractal / Language Table / LIBERO — LeRobot v3 (meta/info.json + parquet)
- Папку с картинками (smoke)
- Видео (Egocentric-100K) — пока заглушка, извлекает uniform frames через cv2
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

logger = logging.getLogger("factory.loader")

@dataclass
class FrameInfo:
    episode_id: str
    frame_index: int
    total_frames: int
    image_path: Path  # абсолютный путь к картинке (или к видео-кадру)
    instruction: str
    state: list[float] | None = None
    action: list[float] | None = None
    reward: float | None = None
    success: bool | None = None
    timestamp: float | None = None
    raw_meta: dict | None = None

@dataclass
class Episode:
    episode_id: str
    instruction: str
    frames: list[FrameInfo]
    dataset_root: Path
    task_index: int | None = None


def _find_lerobot_roots(input_dir: Path) -> list[Path]:
    """Находит все подпапки с meta/info.json (LeRobot датасет)."""
    roots: list[Path] = []
    for info in input_dir.rglob("meta/info.json"):
        root = info.parent.parent
        roots.append(root)
    # Также сам input может быть LeRobot root
    if (input_dir / "meta" / "info.json").exists():
        if input_dir not in roots:
            roots.append(input_dir)
    return sorted(set(roots))


def _load_tasks_map(lerobot_root: Path) -> dict[int, str]:
    """Загружает tasks.parquet -> {task_index: instruction}.

    В LIBERO/HuggingFaceVLA tasks.parquet хранится как индекс=task string,
    колонка task_index — поэтому df.columns = ['task_index'], а индекс — сами инструкции.
    """
    tasks_parquet = lerobot_root / "meta" / "tasks.parquet"
    if not tasks_parquet.exists():
        return {}
    try:
        import pandas as pd
        df = pd.read_parquet(tasks_parquet)
        m: dict[int, str] = {}
        # Случай 1: индекс = task string
        if df.index.dtype == object and df.index.name == "task":
            for task_str, row in df.iterrows():
                idx = int(row["task_index"])
                m[idx] = str(task_str)
            return m
        # Случай 2: колонки task и task_index
        if "task" in df.columns and "task_index" in df.columns:
            for _, row in df.iterrows():
                m[int(row["task_index"])] = str(row["task"])
            return m
        # Fallback: пробуем pyarrow напрямую
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(tasks_parquet)
            # column task may be string
            if "task" in table.column_names and "task_index" in table.column_names:
                tasks = table.column("task").to_pylist()
                indices = table.column("task_index").to_pylist()
                for t, idx in zip(tasks, indices):
                    m[int(idx)] = str(t)
                return m
        except Exception:
            pass
        # Последний fallback — как раньше
        for _, row in df.iterrows():
            idx = int(row.get("task_index", row.get("index", 0)))
            task = str(row.get("task", row.get("instruction", idx)))
            m[idx] = task
        return m
    except Exception as e:
        logger.warning(f"Не удалось прочитать tasks.parquet {tasks_parquet}: {e}")
        return {}


def _load_episodes_meta(lerobot_root: Path) -> dict[int, dict]:
    """Загружает meta/episodes/**/file-*.parquet -> {episode_index: {length, task_index}}"""
    meta = {}
    episodes_dir = lerobot_root / "meta" / "episodes"
    if not episodes_dir.exists():
        return meta
    try:
        import pandas as pd
        for p in episodes_dir.rglob("*.parquet"):
            df = pd.read_parquet(p)
            for _, row in df.iterrows():
                ep_idx = int(row.get("episode_index", row.get("index", 0)))
                meta[ep_idx] = {
                    "episode_index": ep_idx,
                    "length": int(row.get("length", row.get("dataset_from_index", 0))),
                    "task_index": int(row.get("task_index", 0)) if "task_index" in row else None,
                }
    except Exception as e:
        logger.warning(f"Не удалось прочитать episodes meta {lerobot_root}: {e}")
    return meta


def _iter_lerobot_episodes(lerobot_root: Path) -> Iterator[Episode]:
    """Итерируется по эпизодам LeRobot датасета.

    Для offline/CI fallback: если parquet не читается, пытается найти картинки в data/.
    """
    tasks = _load_tasks_map(lerobot_root)
    episodes_meta = _load_episodes_meta(lerobot_root)
    info_path = lerobot_root / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:
        info = {}

    # Путь к данным: data/chunk-XXX/file-XXX.parquet
    data_path_tmpl = info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    # Парсим шаблон
    # Простой способ: найти все parquet под data/
    data_files = sorted((lerobot_root / "data").rglob("*.parquet")) if (lerobot_root / "data").exists() else []
    if not data_files:
        # Возможно, датасет хранит видео, а не картинки — ищем videos/
        logger.warning(f"Нет data/*.parquet в {lerobot_root}, пропускаем")
        return

    # Если pandas нет — fallback: считаем что 1 файл = 1 эпизод (?) — лучше пропустить
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas не установлен — LeRobot parquet не читается, fallback на image-folder режим")
        return

    # Читаем все файлы, группируем по episode_index
    from collections import defaultdict
    ep_groups: dict[int, list] = defaultdict(list)
    for data_file in data_files:
        try:
            df = pd.read_parquet(data_file)
        except Exception as e:
            logger.warning(f"Не удалось прочитать {data_file}: {e}")
            continue
        if "episode_index" not in df.columns:
            # Весь файл — один эпизод
            ep_idx = 0
            ep_groups[ep_idx].append((data_file, df))
        else:
            for ep_idx, g in df.groupby("episode_index"):
                ep_groups[int(ep_idx)].append((data_file, g))

    for ep_idx, chunks in sorted(ep_groups.items()):
        # Склеиваем чанки одного эпизода
        dfs = [c[1] for c in chunks]
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
        df = df.sort_values("frame_index") if "frame_index" in df.columns else df
        n = len(df)
        meta = episodes_meta.get(ep_idx, {})
        task_idx = meta.get("task_index")
        if task_idx is None and "task_index" in df.columns:
            task_idx = int(df.iloc[0]["task_index"])
        instruction = tasks.get(task_idx, "") if task_idx is not None else ""
        if not instruction:
            # Пробуем вытащить из df
            if "task" in df.columns:
                instruction = str(df.iloc[0]["task"])
            elif "instruction" in df.columns:
                instruction = str(df.iloc[0]["instruction"])
            else:
                instruction = f"Perform manipulation task {ep_idx}"

        # Определяем где лежат картинки: в parquet dtype=image хранится как bytes/dict
        # Для HuggingFaceVLA/libero — картинки внутри parquet, но мы их извлечём во временную папку
        # Упрощение: если есть observation.images.image как numpy/bytes — сохраняем на диск
        # Иначе ищем рядом файлы
        frames: list[FrameInfo] = []
        # Проверяем, есть ли колонки с изображениями
        image_cols = [c for c in df.columns if c.startswith("observation.images")]
        if image_cols and any(df[col].dtype == object for col in image_cols):
            # Нужно материализовать картинки на диск (кэш)
            cache_dir = lerobot_root / ".factory_cache" / f"episode_{ep_idx:06d}"
            cache_dir.mkdir(parents=True, exist_ok=True)
            for i, (_, row) in enumerate(df.iterrows()):
                # Берём первую камеру как основную
                img_col = image_cols[0]
                img_data = row[img_col]
                # img_data может быть bytes, np.ndarray, PIL, dict с bytes
                try:
                    from PIL import Image
                    import numpy as np
                    import io
                    if isinstance(img_data, dict) and "bytes" in img_data:
                        img_bytes = img_data["bytes"]
                        img = Image.open(io.BytesIO(img_bytes))
                    elif isinstance(img_data, bytes):
                        img = Image.open(io.BytesIO(img_data))
                    elif isinstance(img_data, np.ndarray):
                        # HWC uint8
                        arr = img_data
                        if arr.ndim == 3 and arr.shape[0] in (1,3,4) and arr.shape[-1] not in (1,3,4):
                            arr = np.transpose(arr, (1,2,0))
                        img = Image.fromarray(arr.astype("uint8"))
                    else:
                        # Попытка как PIL
                        img = Image.fromarray(np.array(img_data))
                    # Сохраняем
                    img_path = cache_dir / f"frame_{i:05d}.jpg"
                    if not img_path.exists():
                        img.convert("RGB").save(img_path, quality=90)
                    frames.append(FrameInfo(
                        episode_id=f"{lerobot_root.name}_{ep_idx:06d}",
                        frame_index=i,
                        total_frames=n,
                        image_path=img_path,
                        instruction=instruction,
                        state=row.get("observation.state").tolist() if hasattr(row.get("observation.state"), "tolist") else None,
                        action=row.get("action").tolist() if hasattr(row.get("action"), "tolist") else None,
                        raw_meta={"episode_index": ep_idx, "frame_index": i, "task_index": task_idx},
                    ))
                except Exception as e:
                    logger.debug(f"Не удалось сохранить кадр {i} эпизода {ep_idx}: {e}")
                    continue
        else:
            # Ищем картинки рядом: videos/ или images/
            # Для простоты — пропускаем, если нет картинок
            logger.warning(f"Эпизод {ep_idx} не содержит image-колонок — пропускаем (нужен кэш)")
            continue

        if not frames:
            continue
        yield Episode(episode_id=f"{lerobot_root.name}_{ep_idx:06d}", instruction=instruction, frames=frames, dataset_root=lerobot_root, task_index=task_idx)


def _iter_image_folder(input_dir: Path, extensions: set[str]) -> Iterator[Episode]:
    """Fallback: папка с картинками -> 1 эпизод на подпапку или весь input как 1 эпизод."""
    images = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in extensions)
    if not images:
        return
    # Группируем по родительской папке (если много эпизодов)
    # Если в input_dir есть подпапки с картинками — каждая подпапка = эпизод
    # Иначе весь input = 1 эпизод
    from collections import defaultdict
    groups: dict[str, list[Path]] = defaultdict(list)
    for img in images:
        # Относительный путь — первый уровень
        rel = img.relative_to(input_dir)
        # Если глубина >1, группируем по первой компоненте
        if len(rel.parts) > 1:
            key = rel.parts[0]
        else:
            key = "_root"
        groups[key].append(img)

    # Эвристика: если групп >1 и каждая <100 кадров, считаем группами
    # Иначе все картинки — один эпизод
    if len(groups) > 1 and all(len(v) < 200 for v in groups.values()):
        use_groups = True
    else:
        # Один эпизод из всех картинок
        groups = {"episode_000000": sorted(images)}

    # Пытаемся найти инструкции рядом: *.json, *.txt, tasks.parquet
    for gid, imgs in sorted(groups.items()):
        imgs = sorted(imgs)
        # Ищем instruction
        instruction = ""
        # Рядом с первой картинкой может быть .json с таким же именем
        cand_json = imgs[0].with_suffix(".json")
        if cand_json.exists():
            try:
                data = json.loads(cand_json.read_text(encoding="utf-8"))
                instruction = data.get("instruction", data.get("task", data.get("prompt", "")))
            except Exception:
                pass
        if not instruction:
            # Ищем instruction.txt в папке эпизода
            episode_dir = imgs[0].parent if len(groups) > 1 else input_dir
            for cand in [episode_dir / "instruction.txt", episode_dir / "task.txt", input_dir / "instructions.jsonl"]:
                if cand.exists():
                    try:
                        if cand.suffix == ".jsonl":
                            # Берём первую непустую
                            for line in cand.read_text(encoding="utf-8").splitlines():
                                if line.strip():
                                    j = json.loads(line)
                                    if gid in j.get("episode_id", "") or True:
                                        instruction = j.get("instruction", "")
                                        break
                        else:
                            instruction = cand.read_text(encoding="utf-8").strip().splitlines()[0]
                    except Exception:
                        pass
                    break
        if not instruction:
            instruction = f"Perform manipulation task {gid}"

        frames = []
        n = len(imgs)
        for i, img_path in enumerate(imgs):
            frames.append(FrameInfo(
                episode_id=str(gid),
                frame_index=i,
                total_frames=n,
                image_path=img_path,
                instruction=instruction,
                raw_meta={"source": "image_folder"},
            ))
        yield Episode(episode_id=str(gid), instruction=instruction, frames=frames, dataset_root=input_dir)


def discover_episodes(input_dir: Path, image_extensions: list[str] | None = None) -> list[Episode]:
    """Главная функция discovery: LeRobot -> fallback image folder."""
    if image_extensions is None:
        image_extensions = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    exts = {e.lower() for e in image_extensions}
    lerobot_roots = _find_lerobot_roots(input_dir)
    episodes: list[Episode] = []
    if lerobot_roots:
        logger.info(f"Найдено LeRobot датасетов: {lerobot_roots}")
        for root in lerobot_roots:
            for ep in _iter_lerobot_episodes(root):
                episodes.append(ep)
        if episodes:
            logger.info(f"Загружено {len(episodes)} эпизодов из LeRobot")
            return episodes
        else:
            logger.warning("LeRobot корни найдены, но эпизоды не загрузились — fallback на image folder")
    # Fallback
    for ep in _iter_image_folder(input_dir, exts):
        episodes.append(ep)
    logger.info(f"Загружено {len(episodes)} эпизодов из image folder {input_dir}")
    return episodes

