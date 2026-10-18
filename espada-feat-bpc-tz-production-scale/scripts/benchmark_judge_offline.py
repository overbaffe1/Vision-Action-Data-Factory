#!/usr/bin/env python3
"""Benchmark/offline-run BPC Judge over already produced bboxing/pointing JSONs."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.bpc.config import load_bpc_config  # noqa: E402
from src.bpc.endpoints import EndpointPool, build_targets  # noqa: E402
from src.bpc.judge import CoordinateChangedError, build_judge_prompt, verify_judge_output  # noqa: E402
from src.bpc.state import StateLogger  # noqa: E402
from src.bpc.storage import atomic_save_json, setup_output_dirs  # noqa: E402
from src.vlm.prompts_processing import load_yaml  # noqa: E402
from src.vlm.client import chat_completion  # noqa: E402
from src.utils import build_detection_result  # noqa: E402


POINT_KEYS = ("point_2d", "point", "points")


@dataclass(frozen=True)
class JudgeJob:
    task_name: str
    image_id: str
    source_path: Path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_json_payload(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        object_start = text.find("{")
        object_end = text.rfind("}")
        list_start = text.find("[")
        list_end = text.rfind("]")
        object_span = object_end - object_start if object_start >= 0 and object_end >= 0 else -1
        list_span = list_end - list_start if list_start >= 0 and list_end >= 0 else -1
        if object_span >= list_span and object_span > 0:
            return json.loads(text[object_start : object_end + 1])
        if list_span > 0:
            return json.loads(text[list_start : list_end + 1])
        raise


def _coord_key(item: dict[str, Any], task_name: str) -> str:
    if task_name == "bboxing":
        return "bbox_2d"
    for key in POINT_KEYS:
        if key in item:
            return key
    return "point_2d"


def _normalize_label(label: Any) -> str:
    if isinstance(label, dict):
        for key in ("label", "corrected_label", "new_label", "class", "category", "object", "name"):
            value = label.get(key)
            if isinstance(value, str):
                label = value
                break
    if not isinstance(label, str):
        raise ValueError(f"Cannot extract label from {label!r}")
    label = " ".join(label.strip().lower().replace("_", " ").split())
    if not label:
        raise ValueError("Empty label returned by judge.")
    return label


def _extract_labels(payload: Any, expected_count: int) -> list[str]:
    if isinstance(payload, dict):
        for key in ("labels", "corrected_labels", "new_labels", "fixed_labels", "answer", "objects", "items"):
            if key in payload:
                payload = payload[key]
                break
    if isinstance(payload, dict):
        items = sorted(payload.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 10**9)
        labels = [_normalize_label(value) for _, value in items]
    elif isinstance(payload, list):
        if all(isinstance(item, dict) and "index" in item for item in payload):
            payload = sorted(payload, key=lambda item: int(item["index"]))
        labels = [_normalize_label(item) for item in payload]
    else:
        raise ValueError(f"Unsupported labels payload: {type(payload).__name__}")
    if len(labels) != expected_count:
        raise ValueError(f"Expected {expected_count} labels, got {len(labels)}")
    return labels


def _labels_only_prompt(annotation: dict[str, Any], task_name: str) -> tuple[str, str]:
    items = []
    for index, item in enumerate(annotation.get("answer", [])):
        key = _coord_key(item, task_name)
        items.append(
            {
                "index": index,
                "coordinate_key": key,
                "coordinates": item.get(key),
                "current_label": item.get("label", ""),
            }
        )
    system = (
        "You are a strict visual annotation label corrector. "
        "Return only one JSON object with one key, \"labels\". "
        "The value must be an array of short visual object labels in the same order as input items. "
        "Do not return coordinates, explanations, markdown, or extra keys."
    )
    user = json.dumps(
        {
            "task": task_name,
            "instruction": (
                "Inspect the image and correct only the label text for each item. "
                "Keep the array length exactly equal to the number of input items."
            ),
            "items": items,
            "output_format": {"labels": ["label_for_item_0", "label_for_item_1"]},
        },
        ensure_ascii=False,
    )
    return system, user


def _apply_labels(annotation: dict[str, Any], labels: list[str], task_name: str) -> dict[str, Any]:
    updated = json.loads(json.dumps(annotation, ensure_ascii=False))
    for item, label in zip(updated.get("answer", []), labels, strict=True):
        item["label"] = label
    unique_labels = list(dict.fromkeys(labels))
    label_text = ", ".join(unique_labels)
    if task_name == "bboxing":
        updated["question"] = f"Locate every {label_text} in the image. Return the answer in JSON."
    else:
        updated["question"] = f"Point to every {label_text} in the image. Return the answer in JSON."
    return updated


def _resolve_image_path(annotation: dict[str, Any], image_id: str, input_dir: Path) -> Path:
    images = annotation.get("images")
    candidates: list[Path] = []

    if isinstance(images, list) and len(images) == 1 and isinstance(images[0], str):
        raw = Path(images[0])
        if raw.is_absolute():
            candidates.append(raw)
        candidates.append(input_dir / raw.name)
        if raw.suffix:
            candidates.append(input_dir / f"{image_id}{raw.suffix}")

    candidates.extend(input_dir / f"{image_id}{suffix}" for suffix in (".jpg", ".jpeg", ".JPG", ".JPEG"))

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Could not resolve source image for {image_id} in {input_dir}")


def _collect_jobs(
    *,
    run_dir: Path,
    tasks: list[str],
    limit_per_task: int | None,
    overwrite: bool,
) -> list[JudgeJob]:
    jobs: list[JudgeJob] = []
    for task_name in tasks:
        source_dir = run_dir / "outputs" / task_name
        judged_dir = run_dir / "outputs" / f"{task_name}-judged"
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Missing source dir: {source_dir}")

        added = 0
        for source_path in sorted(source_dir.glob("*.json")):
            image_id = source_path.stem
            if not overwrite and (judged_dir / source_path.name).exists():
                continue
            jobs.append(
                JudgeJob(
                    task_name=task_name,
                    image_id=image_id,
                    source_path=source_path,
                )
            )
            added += 1
            if limit_per_task is not None and added >= limit_per_task:
                break
    return jobs


def _build_judge_pool(args: argparse.Namespace, cfg: Any) -> EndpointPool:
    if args.use_vlm_targets:
        targets = build_targets(
            targets=cfg.vlm.targets,
            endpoints=cfg.vlm.endpoints,
            router_url=cfg.vlm.router_url,
            models=cfg.vlm.models,
        )
        default_cap = cfg.vlm.max_concurrency_per_endpoint
    else:
        targets = build_targets(
            targets=cfg.judge.targets,
            endpoints=cfg.judge.endpoints,
            router_url=cfg.judge.router_url,
            models=cfg.judge.models,
        )
        default_cap = cfg.judge.max_concurrency_per_endpoint

    if args.target_regex:
        pattern = re.compile(args.target_regex)
        targets = [target for target in targets if pattern.search(str(target))]
        if not targets:
            raise ValueError(f"No targets matched --target-regex={args.target_regex!r}")

    if args.cap is not None:
        from src.bpc.endpoints import EndpointTarget

        targets = [
            EndpointTarget(base_url=t.base_url, model=t.model, max_concurrency=args.cap)
            for t in targets
        ]
        default_cap = args.cap

    return EndpointPool(targets, max_concurrency_per_endpoint=default_cap)


def _process_job(
    *,
    job: JudgeJob,
    run_dir: Path,
    input_dir: Path,
    cfg: Any,
    prompts: dict[str, Any],
    state_logger: StateLogger,
    judge_pool: EndpointPool,
    api_token: str,
    model: str,
    worker_id: str,
    mode: str,
) -> None:
    dirs = setup_output_dirs(run_dir, save_visualizations=False)
    annotation = _read_json(job.source_path)
    image_path = _resolve_image_path(annotation, job.image_id, input_dir)
    judged_dest = dirs[f"{job.task_name}_judged"] / f"{job.image_id}.json"

    if mode == "labels-only" and not annotation.get("answer"):
        atomic_save_json(annotation, judged_dest)
        state_logger.event(
            image_id=job.image_id,
            relative_path=str(image_path),
            stage="judge",
            task_name=job.task_name,
            event="done",
            worker_id=worker_id,
            duration_sec=0,
            output_path=str(judged_dest),
            extra={"endpoint": "local-empty-copy", "model": "none"},
        )
        return

    if mode == "labels-only":
        system_text, user_text = _labels_only_prompt(annotation, job.task_name)
    else:
        system_text, user_text = build_judge_prompt(job.task_name, annotation, prompts)

    with judge_pool.acquire() as target:
        call_model = target.model or model
        started = time.monotonic()
        try:
            result = chat_completion(
                base_url=target.base_url,
                api_token=api_token,
                model=call_model,
                system_text=system_text,
                user_text=user_text,
                images=[image_path],
                timeout_s=cfg.vlm.timeout_s,
            )
            if mode == "labels-only":
                labels = _extract_labels(_extract_json_payload(result.content), len(annotation.get("answer", [])))
                parsed = _apply_labels(annotation, labels, job.task_name)
            else:
                parsed = build_detection_result(result.content, "")
            duration = time.monotonic() - started
            judge_pool.report_success(target)
        except Exception:
            judge_pool.report_failure(target)
            raise

    try:
        verify_judge_output(annotation, parsed, job.task_name)
    except CoordinateChangedError:
        state_logger.event(
            image_id=job.image_id,
            relative_path=str(image_path),
            stage="judge",
            task_name=job.task_name,
            event="failed",
            worker_id=worker_id,
            error="Judge changed coordinates.",
        )
        raise

    atomic_save_json(parsed, judged_dest)
    state_logger.event(
        image_id=job.image_id,
        relative_path=str(image_path),
        stage="judge",
        task_name=job.task_name,
        event="done",
        worker_id=worker_id,
        duration_sec=duration,
        output_path=str(judged_dest),
        extra={"endpoint": str(target), "model": call_model},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Existing BPC output run.")
    parser.add_argument("--input", required=True, type=Path, help="Flat image directory for this run.")
    parser.add_argument("--config", default="configs/bpc.direct8.nojudge.yaml", help="BPC config path.")
    parser.add_argument("--tasks", default="bboxing,pointing", help="Comma-separated tasks to judge.")
    parser.add_argument("--limit-per-task", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=0, help="Stop submitting new work after this many seconds.")
    parser.add_argument("--workers", type=int, default=0, help="Judge worker threads. Default: endpoint capacity.")
    parser.add_argument(
        "--mode",
        choices=("labels-only", "full-json"),
        default="labels-only",
        help="labels-only matches the legacy fast judge: model returns labels, code preserves coordinates.",
    )
    parser.add_argument("--use-vlm-targets", action="store_true", help="Use vlm.targets instead of judge targets.")
    parser.add_argument("--target-regex", default="", help="Filter targets by regex over 'base_url::model'.")
    parser.add_argument("--cap", type=int, default=None, help="Override per-target cap for this benchmark.")
    parser.add_argument("--overwrite", action="store_true", help="Re-judge files even if judged output exists.")
    args = parser.parse_args()

    cfg = load_bpc_config(args.config)
    prompts = load_yaml("configs/prompts.yaml")
    judge_pool = _build_judge_pool(args, cfg)
    workers = args.workers or judge_pool.total_capacity
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    jobs = _collect_jobs(
        run_dir=args.run,
        tasks=tasks,
        limit_per_task=args.limit_per_task,
        overwrite=args.overwrite,
    )
    model = cfg.judge.models[0] if cfg.judge.models and not args.use_vlm_targets else (cfg.vlm.model or "")
    api_token = os.getenv("API_TOKEN", "")
    state_logger = StateLogger(args.run / "logs", f"judge-bench-{uuid.uuid4()}")

    print(f"run={args.run}")
    print(f"jobs={len(jobs)} tasks={tasks} workers={workers} endpoint_capacity={judge_pool.total_capacity}")
    for target in judge_pool.targets:
        print(f"target={target} cap={target.max_concurrency or 'default'}")

    started = time.monotonic()
    done = 0
    failed = 0
    pending = iter(jobs)
    futures = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for _ in range(min(workers, len(jobs))):
            job = next(pending)
            futures[
                executor.submit(
                    _process_job,
                    job=job,
                    run_dir=args.run,
                    input_dir=args.input,
                    cfg=cfg,
                    prompts=prompts,
                    state_logger=state_logger,
                    judge_pool=judge_pool,
                    api_token=api_token,
                    model=model,
                    worker_id=f"judge_bench_{done}",
                    mode=args.mode,
                )
            ] = job

        with tqdm(total=len(jobs), desc="Judge benchmark") as progress:
            while futures:
                completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed:
                    job = futures.pop(future)
                    try:
                        future.result()
                    except Exception as exc:
                        failed += 1
                        print(f"[failed] {job.task_name}/{job.image_id}: {exc}")
                    done += 1
                    progress.update(1)

                    if args.max_seconds and (time.monotonic() - started) >= args.max_seconds:
                        continue
                    try:
                        next_job = next(pending)
                    except StopIteration:
                        continue
                    futures[
                        executor.submit(
                            _process_job,
                            job=next_job,
                            run_dir=args.run,
                            input_dir=args.input,
                            cfg=cfg,
                            prompts=prompts,
                            state_logger=state_logger,
                            judge_pool=judge_pool,
                            api_token=api_token,
                            model=model,
                            worker_id=f"judge_bench_{done}",
                            mode=args.mode,
                        )
                    ] = next_job

    elapsed = max(time.monotonic() - started, 1e-9)
    print()
    print("=== Judge benchmark result ===")
    print(f"elapsed_sec={elapsed:.1f}")
    print(f"processed={done}")
    print(f"failed={failed}")
    print(f"judge_tasks_per_sec={done / elapsed:.3f}")
    print(f"judge_tasks_per_hour={done / elapsed * 3600:.0f}")
    if set(tasks) == {"bboxing", "pointing"}:
        print(f"images_per_hour_est={done / elapsed * 3600 / 2:.0f}")


if __name__ == "__main__":
    main()
