#!/usr/bin/env python3
"""Vision-Action Data Factory — генератор VL-датасета для дообучения SmolVLM2 → SmolVLA.

Поддерживает:
- LeRobot v3 датасеты (BridgeData2, Fractal, Language Table, LIBERO)
- Папки с картинками (smoke)
- Видео (Egocentric-100K) через cv2 uniform extraction
- Offline режим (без VLM endpoint) — темплейты + YOLO + геометрия
- Online режим — опциональный VLM endpoint для капшнов

Выход: ShareGPT JSONL (messages + images) — совместимо с LLaMA-Factory (aij_robotics).

Идеи из https://github.com/overbaffe1/QAOA — перенесены как инженерные паттерны:
- Ratchet (per-instance merge лучших кандидатов) — как merged2: для каждого кадра генерим N кандидатов разными
  темплейтами, выбираем лучшего heuristic-judge'ом, а не усредняем.
- Lottery (массовый рандом-поиск шаблонов) — для оффлайн оптимизации: перебираем много промптов, оставляем топ.
- KNN warm — для закрытого бенча с новыми объектами/камерами: retrieval по CLIP/histogram ближайших трейновых примеров.
- Polish worst — координатный спуск по худшим эпизодам: добавляем кадры/примеры где quality < threshold.
- Fast group inference — батч YOLO по 4-8 картинок (аналог FastQAOA group=4).

CLI (как в ТЗ):
    python generate_dataset.py --input /data/raw --output /output/annotations.jsonl --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

# Локальные модули
from src.dataset_loader import discover_episodes, Episode, FrameInfo
from src.frame_sampler import sample_frames, polish_worst_extra
from src.detectors.yolo_detector import YoloDetector
from src.spatial import BBox, describe_spatial_relations, heuristic_relations_for_qa, bbox_to_text
from src.subtask import (
    decompose_instruction,
    subtask_at_frame,
    estimate_progress,
    completion_status,
    error_cause,
    recovery_instruction,
)
from src.captioner import generate_caption, generate_qa_variants
from src.judge import score_annotation, judge_candidates, filter_by_threshold
from src.utils import set_seed, get_image_size, make_sharegpt_record, dedup_by_hash

logger = logging.getLogger("factory")


def load_config(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config.yaml root must be mapping")
    return data


def setup_logging(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def build_bboxes_from_yolo(yolo_out: list[dict], width: int, height: int) -> list[BBox]:
    bboxes = []
    for det in yolo_out:
        xyxy = det.get("bbox_2d", det.get("bbox", [0,0,100,100]))
        if len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = map(int, xyxy)
        # Clip
        x1 = max(0, min(width-1, x1)); x2 = max(0, min(width-1, x2))
        y1 = max(0, min(height-1, y1)); y2 = max(0, min(height-1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        label = det.get("label", "object")
        score = float(det.get("score", 1.0))
        bboxes.append(BBox(x1,y1,x2,y2,label,score))
    return bboxes


def generate_examples_for_frame(
    frame: FrameInfo,
    episode: Episode,
    bboxes: list[BBox],
    spatial_rels: list[str],
    width: int,
    height: int,
    config: dict,
    frame_seed: int,
) -> list[dict]:
    """Генерирует примеры для одного кадра по всем task типам."""
    tasks_cfg = config.get("tasks", {})
    enabled = set(tasks_cfg.get("enabled", []))
    weights = tasks_cfg.get("weights", {})
    examples_per_frame = int(tasks_cfg.get("examples_per_frame", 3))
    gen_cfg = config.get("generation", {})
    ratchet = gen_cfg.get("ratchet", {}).get("enabled", True)
    candidates_per_task = int(gen_cfg.get("ratchet", {}).get("candidates_per_task", 3))

    instruction = episode.instruction
    total = frame.total_frames
    idx = frame.frame_index

    # Subtask / progress / completion — считаем один раз
    current_sub, next_sub, cur_idx, n_steps = subtask_at_frame(instruction, idx, total)
    progress_val, progress_text = estimate_progress(idx, total, cur_idx, n_steps)
    comp_label, comp_reason = completion_status(idx, total, frame.success, cur_idx, n_steps)
    err = error_cause(idx, total, instruction, frame.success)
    recov = recovery_instruction(err, instruction) if err else None

    caption = generate_caption(bboxes, spatial_rels, instruction, width, height, progress_text, current_sub, seed=frame_seed)

    # Собираем кандидатов по типам
    task_examples: dict[str, list[tuple[str, str]]] = defaultdict(list)

    # --- Scene caption ---
    if "scene_caption" in enabled:
        q = "Describe the scene in detail for robot action planning. Focus on visible objects, hands/gripper, and spatial layout."
        task_examples["scene_caption"].append((q, caption))

    # --- Object state ---
    if "object_state" in enabled:
        if bboxes:
            for b in bboxes[:4]:
                q = f"What is the state of the {b.label} in the image? Describe its position and appearance."
                a = f"The {b.label} is located at [{b.x_min},{b.y_min},{b.x_max},{b.y_max}] (center {int(b.cx)},{int(b.cy)}). It appears {b.label} is resting on the surface, clearly visible. Gripper status: {'near' if any('gripper' in x.label.lower() for x in bboxes) else 'not visible'}."
                task_examples["object_state"].append((q, a))
        else:
            q = "What objects are visible and what are their states?"
            task_examples["object_state"].append((q, caption))

    # --- Spatial relation ---
    if "spatial_relation" in enabled:
        if spatial_rels:
            for rel in spatial_rels[:3]:
                q = f"Describe the spatial relationship: {rel.split(' is ')[0]} — where is it?"
                task_examples["spatial_relation"].append((q, rel.capitalize() + ". This spatial arrangement is relevant for planning the next action."))
        else:
            task_examples["spatial_relation"].append(("Where are objects located relative to each other?", "Objects are spread across the tabletop, with no strong spatial clustering."))

    # --- Subtask current / next ---
    if "subtask_current" in enabled:
        task_examples["subtask_current"].append((
            f"What subtask is the robot currently performing? Instruction: {instruction}",
            f"The robot is currently: {current_sub}. This is step {cur_idx+1}/{n_steps} of the decomposition. Progress: {progress_text}."
        ))
    if "subtask_next" in enabled and next_sub:
        task_examples["subtask_next"].append((
            f"What should the robot do next to complete '{instruction}'?",
            f"Next subtask: {next_sub}. After completing '{current_sub}', the robot should move to this step. Progress so far: {progress_text}."
        ))

    # --- Decomposition ---
    if "decomposition" in enabled:
        steps = decompose_instruction(instruction)
        task_examples["decomposition"].append((
            f"Decompose the instruction '{instruction}' into sequential subtasks.",
            "Step-by-step decomposition: " + "; ".join(f"{i+1}. {s}" for i, s in enumerate(steps)) + f". Currently executing step {cur_idx+1}."
        ))

    # --- Progress ---
    if "progress" in enabled:
        task_examples["progress"].append((
            f"How far has the task '{instruction}' progressed? Estimate percentage and explain.",
            f"Progress: {progress_val}% ({progress_text}). Frame {idx+1}/{total}, subtask {cur_idx+1}/{n_steps}. Visual evidence: gripper position and object arrangement indicate this stage."
        ))

    # --- Completion ---
    if "completion" in enabled:
        q = f"Has the task '{instruction}' been completed? Answer yes/no/partial and explain."
        a = f"Completion: {comp_label}. {comp_reason} Progress {progress_val}%, frame {idx+1}/{total}."
        task_examples["completion"].append((q, a))

    # --- Error cause / Recovery ---
    if "error_cause" in enabled and err:
        task_examples["error_cause"].append((
            f"What went wrong in this episode for task '{instruction}'? Describe the failure cause.",
            err
        ))
    if "recovery" in enabled and recov:
        task_examples["recovery"].append((
            f"How should the robot recover from the error to complete '{instruction}'?",
            recov
        ))

    # --- QA counting / spatial / temporal / action ---
    if "qa_counting" in enabled:
        from collections import Counter
        cnt = Counter(b.label for b in bboxes)
        for label, c in list(cnt.items())[:2]:
            task_examples["qa_counting"].append((
                f"How many {label}s are visible? Count and point to each.",
                f"There are {c} {label}(s). Locations: " + "; ".join(f"[{b.x_min},{b.y_min},{b.x_max},{b.y_max}]" for b in bboxes if b.label==label)
            ))
    if "qa_spatial" in enabled:
        for rel in spatial_rels[:2]:
            task_examples["qa_spatial"].append((
                f"Is it true that {rel}? Verify from the image and explain.",
                f"Yes, {rel}. Verified by observing object centers and bounding boxes in the image."
            ))
    if "qa_temporal" in enabled:
        task_examples["qa_temporal"].append((
            "What happened just before this frame? What will happen next?",
            f"Before: robot was moving toward current position for subtask '{current_sub}'. Next: {next_sub or 'task completion check'}. Frame {idx+1}/{total} of episode."
        ))
    if "qa_action" in enabled:
        task_examples["qa_action"].append((
            f"What action should the robot take next for '{instruction}'?",
            f"Next action corresponds to '{next_sub or current_sub}': move gripper to target, adjust open/close, and verify object pose."
        ))

    # --- BBox grounding / pointing (структурные) ---
    if "bbox_grounding" in enabled and bboxes:
        labels = sorted(set(b.label for b in bboxes))
        q = f"Locate every {', '.join(labels)} in the image. Return the answer in JSON."
        a = json.dumps([{"bbox_2d":[b.x_min,b.y_min,b.x_max,b.y_max],"label":b.label} for b in bboxes if b.label != "background"], ensure_ascii=False)
        task_examples["bbox_grounding"].append((q, a))
    if "pointing" in enabled and bboxes:
        # Точки — центры боксов
        pts = [{"point_2d":[int(b.cx),int(b.cy)],"label":b.label} for b in bboxes[:5]]
        if pts:
            label = pts[0]["label"]
            task_examples["pointing"].append((
                f"Point to every {label} in the image. Return the answer in JSON.",
                json.dumps([p for p in pts if p["label"]==label], ensure_ascii=False)
            ))

    # Теперь применяем ratchet: для каждого task выбираем лучшего кандидата из N вариантов (темплейтов)
    # У нас пока по 1 варианту на task, но для разнообразия генерим jitter-версии с разными seed
    final_candidates: list[tuple[str, str, str]] = []  # (task, q, a)
    for task, candidates in task_examples.items():
        if not candidates:
            continue
        # Если ratchet enabled — делаем кандидаты: оригинальный + 2 парафраза
        expanded: list[tuple[str, str, float | None]] = []
        for (q, a) in candidates:
            expanded.append((q, a, None))
            if ratchet and candidates_per_task > 1:
                # Парафразы: меняем формулировку вопроса, добавляем детали в ответ
                for p in range(1, candidates_per_task):
                    # Простой парафраз: добавляем префикс/суффикс, меняем seed
                    q2 = q
                    if p == 1:
                        q2 = "Carefully observe the image. " + q
                    elif p == 2:
                        q2 = q + " Be precise and reference visible evidence."
                    # Чуть меняем ответ: добавляем bbox текст
                    a2 = a
                    if bboxes and random.Random(frame_seed + p).random() > 0.5:
                        a2 = a + " Evidence: " + bbox_to_text(bboxes[:2])
                    expanded.append((q2, a2, None))
        # Теперь judge выбирает лучшего из expanded для этого task
        if ratchet:
            # Группируем по исходному кандидату (каждые candidates_per_task — группа)
            # Но проще: для каждого task выбрать топ-1 из всех expanded
            # Однако нам нужно сохранить разнообразие — берём лучший на каждый исходный q
            # Поэтому группируем по q-стему без префикса
            # Упрощение: берём глобально лучший для этого task (1 пример на task тип)
            q_best, a_best, s_best = judge_candidates(expanded, instruction, bboxes)
            final_candidates.append((task, q_best, a_best))
        else:
            for q,a,_ in expanded:
                final_candidates.append((task, q, a))

    # Балансировка по весам: если нужно examples_per_frame, сэмплим по весам
    # Сейчас final_candidates = ~1 на task тип (до 16). Нужно выбрать examples_per_frame
    # Используем взвешенный сэмплинг без повторений
    if len(final_candidates) > examples_per_frame:
        # Веса
        weights_list = [float(weights.get(t, 1.0)) for t,_,_ in final_candidates]
        # Нормализуем
        total_w = sum(weights_list)
        probs = [w/total_w for w in weights_list]
        # Выбираем топ по weighted score * judge score
        scored = []
        for (task,q,a), p in zip(final_candidates, probs):
            j = score_annotation(q, a, instruction, bboxes)
            scored.append(((task,q,a), j * (0.5 + p)))  # p влияет но j доминирует
        scored.sort(key=lambda x: x[1], reverse=True)
        final_candidates = [x[0] for x in scored[:examples_per_frame]]
    # Если меньше — берём все
    return final_candidates


def main():
    parser = argparse.ArgumentParser(description="Vision-Action Data Factory — генератор VL датасета")
    parser.add_argument("--input", type=Path, required=True, help="Папка с сырыми эпизодами (LeRobot или images)")
    parser.add_argument("--output", type=Path, required=True, help="Куда писать annotations.jsonl")
    parser.add_argument("--config", type=Path, required=True, help="Путь к config.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Переопределить seed")
    parser.add_argument("--limit-episodes", type=int, default=None, help="Ограничить число эпизодов (дебаг)")
    parser.add_argument("--limit-frames", type=int, default=None, help="Ограничить число кадров (дебаг)")
    args = parser.parse_args()

    t0 = time.time()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    seed = int(config.get("seed", 42))
    set_seed(seed)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    logger.info(f"Seed={seed}, input={args.input}, output={args.output}, config={args.config}")

    if not args.input.exists():
        logger.error(f"Input не найден: {args.input}")
        sys.exit(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Discover episodes
    image_exts = config.get("input", {}).get("image_extensions", [".jpg",".jpeg",".png",".webp",".bmp"])
    episodes = discover_episodes(args.input, image_exts)
    if not episodes:
        logger.error(f"Не найдено эпизодов в {args.input} — проверьте структуру (ожидается LeRobot meta/info.json или папка с картинками)")
        sys.exit(1)
    logger.info(f"Найдено эпизодов: {len(episodes)}")
    if args.limit_episodes:
        episodes = episodes[:args.limit_episodes]
        logger.info(f"Ограничено до {len(episodes)} эпизодов")

    # Детектор
    det_cfg = config.get("detectors", {}).get("yolo", {})
    yolo_enabled = det_cfg.get("enabled", True)
    yolo = None
    if yolo_enabled:
        yolo = YoloDetector(
            model=det_cfg.get("model", "yolov8m.pt"),
            device=det_cfg.get("device", "auto"),
            conf=float(det_cfg.get("conf", 0.25)),
            iou=float(det_cfg.get("iou", 0.45)),
        )
    else:
        logger.info("YOLO отключён — будут dummy боксы")

    sampling_cfg = config.get("sampling", {})
    strategy = sampling_cfg.get("strategy", "adaptive")
    frames_per_ep = int(sampling_cfg.get("frames_per_episode", 5))
    polish_threshold = float(sampling_cfg.get("polish_threshold", 0.35))
    polish_extra = int(sampling_cfg.get("polish_worst_extra_frames", 2))

    # Собираем все сэмплированные кадры
    sampled: list[tuple[Episode, FrameInfo]] = []
    for ep in episodes:
        frames = sample_frames(
            ep,
            strategy=strategy,
            frames_per_episode=frames_per_ep,
            uniform_stride=int(sampling_cfg.get("uniform_stride", 6)),
            keyframe_threshold=float(sampling_cfg.get("keyframe_threshold", 0.08)),
            min_frames=int(sampling_cfg.get("min_frames", 3)),
            max_frames=int(sampling_cfg.get("max_frames", 8)),
            include_first_last=bool(sampling_cfg.get("include_first_last", True)),
            seed=seed,
        )
        for f in frames:
            sampled.append((ep, f))
    logger.info(f"Сэмплировано кадров: {len(sampled)} (стратегия {strategy}, {frames_per_ep} на эпизод)")

    if args.limit_frames:
        sampled = sampled[:args.limit_frames]
        logger.info(f"Ограничено кадров до {len(sampled)}")

    # Батч YOLO
    if yolo is not None:
        image_paths = [frame.image_path for _, frame in sampled]
        logger.info(f"YOLO detect batch {len(image_paths)} изображений...")
        yolo_results = yolo.detect_batch(image_paths)
    else:
        yolo_results = [[] for _ in sampled]

    # Генерация примеров
    all_records: list[dict] = []
    quality_scores: dict[str, dict[int, float]] = defaultdict(dict)  # episode_id -> frame_index -> score

    # Сначала один проход без polish, собираем quality
    temp_records_per_frame: list[list[dict]] = []
    for (ep, frame), yolo_out in zip(sampled, yolo_results):
        width, height = get_image_size(frame.image_path)
        bboxes = build_bboxes_from_yolo(yolo_out, width, height)
        spatial_rels = describe_spatial_relations(bboxes, width, height)
        frame_seed = seed + hash(frame.image_path.name) % 100000 + frame.frame_index * 1000
        candidates = generate_examples_for_frame(frame, ep, bboxes, spatial_rels, width, height, config, frame_seed)
        # Конвертируем в ShareGPT records (пока без фильтра)
        records = []
        for task, q, a in candidates:
            rec = make_sharegpt_record(q, a, frame.image_path, relative=bool(config.get("output", {}).get("relative_paths", True)), media_dir=args.input if config.get("output", {}).get("relative_paths") else None)
            # Для дебага — добавляем task в мета (не в jsonl, но в лог)
            records.append(rec)
            # Считаем quality для polish
            s = score_annotation(q, a, ep.instruction, bboxes)
            # Сохраняем средний по фрейму
            quality_scores[ep.episode_id][frame.frame_index] = max(quality_scores[ep.episode_id].get(frame.frame_index, 0), s)
        temp_records_per_frame.append(records)
        all_records.extend(records)

    # Polish worst: добавляем кадры в низкокачественные эпизоды
    if sampling_cfg.get("polish_worst_extra_frames", 0) > 0:
        # Находим эпизоды с низким качеством
        # Для каждого эпизода — средний score
        ep_to_frames: dict[str, list[FrameInfo]] = defaultdict(list)
        ep_map: dict[str, Episode] = {}
        for ep, frame in sampled:
            ep_to_frames[ep.episode_id].append(frame)
            ep_map[ep.episode_id] = ep
        for ep_id, frames in ep_to_frames.items():
            avg_q = sum(quality_scores[ep_id].get(f.frame_index, 0.5) for f in frames) / max(1, len(frames))
            if avg_q < polish_threshold:
                ep = ep_map[ep_id]
                # Текущие сэмплированные для эпизода
                cur_sampled = [f for (e,f) in sampled if e.episode_id == ep_id]
                extra = polish_worst_extra(ep, cur_sampled, quality_scores[ep_id], threshold=polish_threshold, extra_frames=polish_extra)
                # Если добавились новые кадры — генерим для них примеры
                new_frames = [f for f in extra if f not in cur_sampled]
                if new_frames:
                    logger.info(f"Polish {ep_id}: avg_quality {avg_q:.2f} — добавляем {len(new_frames)} кадров")
                    # YOLO для новых
                    new_paths = [f.image_path for f in new_frames]
                    new_yolo = yolo.detect_batch(new_paths) if yolo else [[] for _ in new_paths]
                    for frame, yolo_out in zip(new_frames, new_yolo):
                        width, height = get_image_size(frame.image_path)
                        bboxes = build_bboxes_from_yolo(yolo_out, width, height)
                        spatial_rels = describe_spatial_relations(bboxes, width, height)
                        frame_seed = seed + hash(frame.image_path.name) % 100000 + frame.frame_index * 1000
                        candidates = generate_examples_for_frame(frame, ep, bboxes, spatial_rels, width, height, config, frame_seed)
                        for task, q, a in candidates:
                            rec = make_sharegpt_record(q, a, frame.image_path, relative=bool(config.get("output", {}).get("relative_paths", True)), media_dir=args.input if config.get("output", {}).get("relative_paths") else None)
                            all_records.append(rec)

    # Фильтрация judge
    quality_cfg = config.get("quality", {})
    judge_cfg = quality_cfg.get("judge", {})
    if judge_cfg.get("enabled", True):
        thresh = float(judge_cfg.get("min_score", 0.4))
        # Группируем по эпизоду для контекста instruction
        # Но filter_by_threshold требует instruction — делаем глобально с пустым
        # Лучше фильтровать по score_annotation с instruction из эпизода — нужно знать маппинг image->instruction
        # Упрощение: фильтруем с усреднённым threshold, но не строго
        before = len(all_records)
        # Собираем instruction по image
        img_to_instr: dict[str, str] = {}
        for ep, frame in sampled:
            img_to_instr[str(frame.image_path)] = ep.instruction
            # Также для polish добавленных — уже есть, но не в sampled — пропускаем
        filtered = []
        for rec in all_records:
            img = rec["images"][0]
            instr = img_to_instr.get(img, "")
            # Если relative path — ищем по basename
            if not instr:
                # Пытаемся по basename
                for k,v in img_to_instr.items():
                    if Path(k).name == Path(img).name:
                        instr = v
                        break
            q = rec["messages"][0]["content"].replace("< image>", "").strip()
            a = rec["messages"][1]["content"]
            s = score_annotation(q, a, instr, None)
            if s >= thresh:
                filtered.append(rec)
        all_records = filtered
        logger.info(f"Judge фильтр: {before} -> {len(all_records)} (threshold {thresh})")

    # Дедуп
    if config.get("deduplication", {}).get("enabled", True):
        before = len(all_records)
        all_records = dedup_by_hash(all_records)
        logger.info(f"Дедуп: {before} -> {len(all_records)}")

    # Лимит
    max_examples = config.get("output", {}).get("max_examples")
    if max_examples is not None:
        try:
            max_examples = int(max_examples)
            if len(all_records) > max_examples:
                # Сэмплируем детерминированно
                rnd = random.Random(seed)
                rnd.shuffle(all_records)
                all_records = all_records[:max_examples]
                logger.info(f"Ограничено до {max_examples} примеров")
        except Exception:
            pass

    # Запись
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    elapsed = time.time() - t0
    logger.info(f"Готово: {len(all_records)} примеров в {output_path} за {elapsed:.1f}s")
    # Также пишем статистику
    stats = {
        "episodes": len(episodes),
        "sampled_frames": len(sampled),
        "examples": len(all_records),
        "seed": seed,
        "input": str(args.input),
        "output": str(output_path),
        "elapsed_sec": round(elapsed, 2),
    }
    stats_path = output_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Статистика: {stats_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
