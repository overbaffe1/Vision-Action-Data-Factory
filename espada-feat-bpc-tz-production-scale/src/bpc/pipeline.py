"""Core BPC pipeline: queue-based stages with retry, resume, and validation."""
from __future__ import annotations

import logging
import queue
import random
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..structs import (
    FrameTask,
    YoloStageResult,
    VlmTask,
    PipelineProgressEvent,
    _build_bbox_vlm_task,
    _build_caption_vlm_task,
    _build_pointing_vlm_task,
)
from ..utils import (
    build_extended_detections,
    filter_big_bboxes,
    draw_detections_cv2,
    draw_sam2_masks_with_centroids,
)
from ..vlm import chat_completion
from ..vlm.parsing import parse_chat_completion
from ..workers import (
    SENTINEL,
    _handle_worker_error,
    _read_frame_images,
    _emit_yolo_outputs,
    progress_monitor_worker,
)

from .config import BpcConfig
from .state import StateLogger
from .storage import (
    atomic_save_llama_factory_record,
    load_existing_output,
)
from .validation import (
    validate_bbox_result,
    validate_pointing_result,
    validate_caption_result,
    ValidationError,
)
from .judge import verify_judge_output, CoordinateChangedError, build_judge_prompt

logger = logging.getLogger("bpc.pipeline")
vlm_logger = logging.getLogger("bpc.vlm")
judge_logger = logging.getLogger("bpc.judge")
validation_logger = logging.getLogger("bpc.validation")


class _StaticTarget:
    """Trivial context manager yielding a fixed EndpointTarget (no pool acquire needed)."""

    def __init__(self, target: Any) -> None:
        self._target = target

    def __enter__(self) -> Any:
        return self._target

    def __exit__(self, *exc_info: object) -> None:
        return None


def _enqueue_judge(
    judge_queue: queue.Queue,
    item: dict[str, Any],
    *,
    state_logger: StateLogger,
    timeout: float = 30.0,
) -> None:
    """Enqueues a Judge task without blocking VLM workers indefinitely.

    If judge_queue stays full for ``timeout`` seconds (Judge workers stuck
    retrying a systemic failure), the item is dropped and logged instead of
    blocking every VLM worker forever — a stuck Judge stage must not stall
    bboxing/pointing/captions throughput.

    Empty-answer results are skipped entirely: the Judge may only edit label
    text on existing items, so an empty list gives it nothing to do. On live
    runs the 35B judge reliably hallucinated new objects into empty lists
    ("Judge changed number of objects: 0 -> N"), which verify_judge_output
    then rejected — i.e. every empty-answer judge call was a guaranteed-
    wasted 35B request plus retries.
    """
    if not item["original_result"].get("answer"):
        judge_logger.info(
            "Skipping Judge for %s/%s: empty answer, nothing to relabel.",
            item["task_name"], item["image_id"],
        )
        state_logger.event(
            image_id=item["image_id"],
            relative_path=item["relative_path"],
            stage="judge",
            task_name=item["task_name"],
            event="skip",
            error="Empty answer; nothing for Judge to edit.",
        )
        return

    try:
        judge_queue.put(item, timeout=timeout)
    except queue.Full:
        judge_logger.error(
            "judge_queue full for %.0fs — dropping Judge task for %s/%s "
            "(base annotation is still saved; Judge review skipped for this image).",
            timeout, item["task_name"], item["image_id"],
        )
        state_logger.event(
            image_id=item["image_id"],
            relative_path=item["relative_path"],
            stage="judge",
            task_name=item["task_name"],
            event="failed",
            error=f"judge_queue full for {timeout:.0f}s, dropped",
        )


def _get_image_size(frame_path: Path) -> tuple[int, int]:
    """Returns (width, height) of an image without full decode."""
    img = cv2.imread(str(frame_path))
    if img is None:
        raise ValueError(f"Cannot read image: {frame_path}")
    h, w = img.shape[:2]
    return w, h


def _vlm_call_with_retry(
    *,
    task_name: str,
    image_id: str,
    relative_path: str,
    system_text: str,
    user_text: str,
    images: list,
    endpoints: list[str] | None = None,
    endpoint_pool: Any = None,
    api_token: str,
    model: str,
    state_logger: StateLogger,
    max_attempts: int,
    backoff_seconds: float,
    timeout_s: int = 60,
    worker_id: str = "",
) -> dict[str, Any] | None:
    """Calls VLM with retry, logs events, returns parsed result or None on exhaustion.

    Endpoint selection: if ``endpoint_pool`` (EndpointPool) is given, each attempt
    acquires a slot from the pool (round-robin + per-endpoint concurrency cap and
    cooldown on repeated failures). The pool's EndpointTarget carries both the
    base_url AND the model name, so a single router URL can fan out across many
    distinct model-routed backends (e.g. gateway.frontierai.ru + "odie2".."odie8").
    Otherwise falls back to round-robin over the plain ``endpoints`` list by
    attempt number, using the fixed ``model`` parameter for all of them.
    """
    from ..utils import build_detection_result
    from .endpoints import EndpointTarget

    if endpoint_pool is None and not endpoints:
        raise ValueError("Either endpoint_pool or endpoints must be provided.")

    attempt = 0
    last_error: str = ""
    t_start = time.monotonic()

    while attempt < max_attempts:
        attempt += 1

        # Acquire an endpoint: pool gives a concurrency-capped (url, model) slot;
        # else plain round-robin over the endpoints list with the fixed model.
        if endpoint_pool is not None:
            slot_ctx = endpoint_pool.acquire()
        else:
            url = endpoints[(attempt - 1) % len(endpoints)]
            slot_ctx = _StaticTarget(EndpointTarget(base_url=url, model=model))

        with slot_ctx as target:
            call_model = target.model or model
            state_logger.event(
                image_id=image_id,
                relative_path=relative_path,
                stage="vlm",
                task_name=task_name,
                event="started",
                worker_id=worker_id,
                attempt=attempt,
            )

            try:
                result = chat_completion(
                    base_url=target.base_url,
                    api_token=api_token,
                    model=call_model,
                    system_text=system_text,
                    user_text=user_text,
                    images=images,
                    timeout_s=timeout_s,
                )
                raw = result.content
                parsed = build_detection_result(raw, "")
                duration = time.monotonic() - t_start

                if endpoint_pool is not None:
                    endpoint_pool.report_success(target)

                # "response", NOT "done": the model replied and parsed, but the
                # result has not yet been validated or saved. Emitting "done"
                # here (as an earlier version did) both double-counted every
                # task in status AND corrupted resume — load_completed() keys
                # off "done", so a task whose HTTP succeeded but whose
                # validation later failed (-> failed.jsonl) would be treated as
                # complete on restart and silently never reprocessed. The
                # single authoritative "done" is written by the caller
                # (_bpc_vlm_worker) only after a successful atomic save.
                state_logger.event(
                    image_id=image_id,
                    relative_path=relative_path,
                    stage="vlm",
                    task_name=task_name,
                    event="response",
                    worker_id=worker_id,
                    attempt=attempt,
                    duration_sec=duration,
                    extra={"endpoint": str(target), "model": call_model},
                )
                return parsed

            except Exception as exc:
                last_error = str(exc)
                if endpoint_pool is not None:
                    endpoint_pool.report_failure(target)
                vlm_logger.warning(
                    "VLM attempt %d/%d failed for %s/%s @ %s: %s",
                    attempt, max_attempts, task_name, image_id, target, last_error,
                )
                state_logger.event(
                    image_id=image_id,
                    relative_path=relative_path,
                    stage="vlm",
                    task_name=task_name,
                    event="retry",
                    worker_id=worker_id,
                    attempt=attempt,
                    error=last_error,
                )

        if attempt < max_attempts:
            # Jitter: if many worker threads fail around the same moment
            # (e.g. an endpoint saturates and rejects a burst of concurrent
            # requests, exactly the odie2 timeout scenario seen on the live
            # 4-GPU run), a fixed sleep makes them all retry in lockstep
            # backoff_seconds later, hammering the same endpoint again at
            # once. Adding up to 30% random jitter spreads retries out.
            time.sleep(backoff_seconds + random.uniform(0, backoff_seconds * 0.3))

    state_logger.failed(
        image_id=image_id,
        relative_path=relative_path,
        stage="vlm",
        task_name=task_name,
        reason=last_error,
        attempt=attempt,
    )
    return None


def _validate_and_save(
    *,
    result: dict[str, Any],
    task_name: str,
    image_id: str,
    relative_path: str,
    image_path: Path,
    dest: Path,
    image_width: int,
    image_height: int,
    cfg: BpcConfig,
    state_logger: StateLogger,
) -> bool:
    """Validates result and atomically saves to dest. Returns True on success."""
    try:
        if task_name == "bboxing":
            validate_bbox_result(
                result,
                image_width=image_width,
                image_height=image_height,
                cfg=cfg.validation.bbox,
            )
        elif task_name == "pointing":
            validate_pointing_result(
                result,
                image_width=image_width,
                image_height=image_height,
                cfg=cfg.validation.pointing,
            )
        elif task_name == "captions":
            validate_caption_result(result, cfg=cfg.validation.captions)
        else:
            raise ValueError(f"Unknown task: {task_name}")
    except ValidationError as exc:
        validation_logger.warning(
            "Validation failed for %s/%s: %s", task_name, image_id, exc
        )
        state_logger.event(
            image_id=image_id,
            relative_path=relative_path,
            stage="validation",
            task_name=task_name,
            event="failed",
            error=str(exc),
        )
        return False

    atomic_save_llama_factory_record(result, image_path, dest)
    validation_logger.info("Saved %s/%s -> %s", task_name, image_id, dest)
    return True


def _judge_task(
    *,
    task_name: str,
    image_id: str,
    relative_path: str,
    original_result: dict[str, Any],
    judged_dest: Path,
    images: list,
    cfg: BpcConfig,
    state_logger: StateLogger,
    api_token: str,
    model: str,
    prompts: dict[str, Any],
    worker_id: str = "",
    endpoint_pool: Any = None,
) -> None:
    """Runs VLM Judge for one task. Saves judged output if coordinates unchanged."""
    system_text, user_text = build_judge_prompt(task_name, original_result, prompts)
    endpoints = None if endpoint_pool is not None else (cfg.judge.endpoints or cfg.vlm.endpoints)

    judged = _vlm_call_with_retry(
        task_name=f"{task_name}-judge",
        image_id=image_id,
        relative_path=relative_path,
        system_text=system_text,
        user_text=user_text,
        images=images,
        endpoints=endpoints,
        endpoint_pool=endpoint_pool,
        api_token=api_token,
        model=model,
        state_logger=state_logger,
        max_attempts=cfg.retry.max_attempts,
        backoff_seconds=cfg.retry.backoff_seconds,
        timeout_s=cfg.vlm.timeout_s,
        worker_id=worker_id,
    )
    if judged is None:
        return

    try:
        verify_judge_output(original_result, judged, task_name)
    except CoordinateChangedError as exc:
        judge_logger.warning(
            "Judge changed coordinates for %s/%s: %s", task_name, image_id, exc
        )
        state_logger.event(
            image_id=image_id,
            relative_path=relative_path,
            stage="judge",
            task_name=task_name,
            event="failed",
            error=str(exc),
        )
        return

    atomic_save_llama_factory_record(judged, relative_path, judged_dest)
    state_logger.event(
        image_id=image_id,
        relative_path=relative_path,
        stage="judge",
        task_name=task_name,
        event="done",
        worker_id=worker_id,
        output_path=str(judged_dest),
    )


def run_bpc_pipeline(
    *,
    images: list[Path],
    output_root: Path,
    cfg: BpcConfig,
    prompts: dict[str, Any],
    api_token: str,
    segmenter: Any,
    yolo_model: Any,
    dirs: dict[str, Path],
    state_logger: StateLogger,
    run_id: str,
) -> None:
    """Runs the full BPC pipeline with queues, resume, retry, validation, and Judge."""
    from ..workers import yolo_worker, yolo_worker_batched, sam_worker
    from .endpoints import EndpointPool, build_targets

    model = cfg.vlm.model or ""

    # --- Endpoint pools: round-robin with per-endpoint concurrency caps ---
    # Supports both classic (one URL per instance) and router (one URL, many
    # model names, e.g. gateway.frontierai.ru + odie2..odie8) deployments.
    vlm_targets = build_targets(
        targets=cfg.vlm.targets,
        endpoints=cfg.vlm.endpoints,
        router_url=cfg.vlm.router_url,
        models=cfg.vlm.models,
    )
    vlm_pool = EndpointPool(
        vlm_targets,
        max_concurrency_per_endpoint=cfg.vlm.max_concurrency_per_endpoint,
    )
    judge_pool = None
    if cfg.judge.enabled:
        if cfg.judge.router_url and cfg.judge.models:
            judge_targets = build_targets(
                targets=cfg.judge.targets,
                router_url=cfg.judge.router_url,
                models=cfg.judge.models,
            )
        elif cfg.judge.targets:
            judge_targets = build_targets(targets=cfg.judge.targets)
        elif cfg.judge.endpoints:
            judge_targets = build_targets(endpoints=cfg.judge.endpoints)
        else:
            judge_targets = vlm_targets
        judge_pool = EndpointPool(
            judge_targets,
            max_concurrency_per_endpoint=cfg.judge.max_concurrency_per_endpoint,
        )
    logger.info(
        "VLM pool: %d target(s), capacity=%d. Judge pool: %s",
        len(vlm_pool.targets), vlm_pool.total_capacity,
        f"{len(judge_pool.targets)} targets, capacity={judge_pool.total_capacity}"
        if judge_pool else "disabled",
    )

    # --- Resume: find already-completed tasks ---
    completed = state_logger.load_completed()
    logger.info("Resume: %d already-completed task entries found.", len(completed))

    # Determine which images will actually be submitted BEFORE starting any
    # worker threads: progress_monitor_worker's total_images must equal the
    # number of images that will emit progress events. Resume-skipped images
    # (all 3 tasks already done) never enter yolo_queue, so counting them
    # would make progress_queue.join() hang forever waiting for events that
    # are never sent.
    submit_list: list[Path] = []
    skipped = 0
    for frame_path in images:
        image_id = frame_path.stem
        tasks_done = sum(1 for t in ("bboxing", "captions", "pointing") if (image_id, t) in completed)
        if tasks_done == 3:
            skipped += 1
            continue
        submit_list.append(frame_path)

    yolo_queue: queue.Queue = queue.Queue(maxsize=32)
    sam_queue: queue.Queue = queue.Queue(maxsize=32)
    vlm_queue: queue.Queue = queue.Queue(maxsize=128)
    judge_queue: queue.Queue = queue.Queue(maxsize=128)
    progress_queue: queue.Queue = queue.Queue()
    errors: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    # YOLO worker (batched or single)
    yolo_batch_size = cfg.yolo.batch_size
    yolo_kwargs: dict[str, Any] = {
        "input_queue": yolo_queue,
        "output_queue": sam_queue,
        "vlm_queue": vlm_queue,
        "yolo_model": yolo_model,
        "prompts": prompts,
        "fail_fast": False,
        "stop_event": stop_event,
        "errors": errors,
        "progress_queue": progress_queue,
    }
    if yolo_batch_size > 1:
        yolo_target = yolo_worker_batched
        yolo_kwargs["batch_size"] = yolo_batch_size
    else:
        yolo_target = yolo_worker

    yolo_thread = threading.Thread(target=yolo_target, kwargs=yolo_kwargs,
                                   name="yolo", daemon=True)

    # Multiple SAM2 worker threads share one Sam2Segmenter (one GPU, one set
    # of loaded weights) but must not call set_image()/predict_from_boxes()
    # concurrently on it (that mutates shared instance state). A single
    # hardcoded SAM2 thread previously meant that a full vlm_queue (e.g. a
    # saturated VLM endpoint) blocked SAM2 entirely, starving `pointing`
    # output while bboxing/captions kept progressing from already-queued
    # work. cfg.sam2.workers threads + a shared lock around just the GPU
    # call gives concurrency headroom: one thread waiting on vlm_queue no
    # longer freezes the others from segmenting the next frame.
    sam2_workers = max(1, cfg.sam2.workers)
    segmenter_lock = threading.Lock()

    def _record_pointing_skipped(task: Any) -> None:
        # Makes "no detections, pointing legitimately skipped" a distinct,
        # queryable state.jsonl event instead of only a transient
        # progress_queue message — lets `espada-bpc status` tell it apart
        # from "stuck, nothing produced".
        state_logger.event(
            image_id=task.image_id,
            relative_path=str(task.frame_path),
            stage="sam2",
            task_name="pointing",
            event="skip",
            error="No detections; pointing skipped.",
        )

    sam_threads = [
        threading.Thread(
            target=sam_worker,
            kwargs={
                "input_queue": sam_queue,
                "vlm_queue": vlm_queue,
                "segmenter": segmenter,
                "prompts": prompts,
                "fail_fast": False,
                "stop_event": stop_event,
                "errors": errors,
                "progress_queue": progress_queue,
                "segmenter_lock": segmenter_lock,
                "on_pointing_skipped": _record_pointing_skipped,
            },
            name=f"sam-{i}", daemon=True,
        )
        for i in range(sam2_workers)
    ]

    vlm_threads = [
        threading.Thread(
            target=_bpc_vlm_worker,
            kwargs={
                "input_queue": vlm_queue,
                "judge_queue": judge_queue if cfg.judge.enabled else None,
                "api_token": api_token,
                "model": model,
                "cfg": cfg,
                "dirs": dirs,
                "state_logger": state_logger,
                "completed": completed,
                "progress_queue": progress_queue,
                "stop_event": stop_event,
                "errors": errors,
                "prompts": prompts,
                "worker_id": f"vlm_{i}",
                "vlm_pool": vlm_pool,
                "judge_pool": judge_pool,
            },
            name=f"vlm-{i}", daemon=True,
        )
        for i in range(cfg.vlm.workers)
    ]

    judge_threads = []
    if cfg.judge.enabled:
        judge_threads = [
            threading.Thread(
                target=_bpc_judge_worker,
                kwargs={
                    "input_queue": judge_queue,
                    "api_token": api_token,
                    "model": model,
                    "cfg": cfg,
                    "dirs": dirs,
                    "state_logger": state_logger,
                    "prompts": prompts,
                    "worker_id": f"judge_{i}",
                    "judge_pool": judge_pool,
                },
                name=f"judge-{i}", daemon=True,
            )
            for i in range(cfg.judge.workers)
        ]

    progress_thread = threading.Thread(
        target=progress_monitor_worker,
        kwargs={
            "total_images": len(submit_list),
            "progress_queue": progress_queue,
        },
        name="progress-monitor", daemon=True,
    )

    for t in [yolo_thread, *sam_threads, *vlm_threads, *judge_threads, progress_thread]:
        t.start()

    # Submit tasks (skip list was already computed above, before any thread
    # started, so progress_thread's total_images matches exactly).
    for frame_path in submit_list:
        yolo_queue.put(FrameTask(image_id=frame_path.stem, frame_path=frame_path))

    logger.info("Submitted %d images (%d skipped by resume).", len(submit_list), skipped)

    # Shutdown chain
    yolo_queue.put(SENTINEL)
    yolo_queue.join()
    yolo_thread.join()

    # One SENTINEL per SAM2 thread — each thread consumes exactly one item
    # from input_queue.get() per loop iteration, so N threads need N
    # sentinels or the extras would never see one and hang on the next get().
    for _ in sam_threads:
        sam_queue.put(SENTINEL)
    sam_queue.join()
    for t in sam_threads:
        t.join()

    for _ in vlm_threads:
        vlm_queue.put(SENTINEL)
    vlm_queue.join()
    for t in vlm_threads:
        t.join()

    if cfg.judge.enabled:
        for _ in judge_threads:
            judge_queue.put(SENTINEL)
        judge_queue.join()
        for t in judge_threads:
            t.join()

    # progress_monitor_worker exits on its own once every submitted image is
    # finalized (see src/workers.py); join() then returns immediately. If
    # submit_list was empty it never entered its wait loop, so join it too
    # for a clean shutdown either way.
    progress_queue.join()
    progress_thread.join()
    logger.info("BPC pipeline complete.")


def _bpc_vlm_worker(
    *,
    input_queue: queue.Queue,
    judge_queue: queue.Queue | None,
    api_token: str,
    model: str,
    cfg: BpcConfig,
    dirs: dict[str, Path],
    state_logger: StateLogger,
    completed: set[tuple[str, str]],
    progress_queue: queue.Queue,
    stop_event: threading.Event,
    errors: queue.Queue,
    prompts: dict[str, Any],
    worker_id: str,
    vlm_pool: Any = None,
    judge_pool: Any = None,
) -> None:
    """VLM worker: validates results, saves atomically, optionally enqueues Judge."""
    while True:
        item = input_queue.get()
        try:
            if item is SENTINEL:
                return
            if stop_event.is_set():
                continue

            task: VlmTask = item
            image_id = task.frame_path.stem
            task_name = task.task_name
            relative_path = str(task.frame_path)

            dest = dirs.get(task_name, dirs["bboxing"]).parent / task_name / f"{image_id}.jsonl"
            legacy_dest = dest.with_suffix(".json")
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Resume: skip if already done. Migrate a legacy per-image JSON on
            # sight so completed runs also acquire LLaMA Factory shards.
            if (image_id, task_name) in completed:
                if not dest.exists():
                    legacy = load_existing_output(legacy_dest)
                    if legacy is not None:
                        atomic_save_llama_factory_record(legacy, task.frame_path, dest)
                logger.debug("RESUME SKIP task %s/%s", task_name, image_id)
                progress_queue.put(PipelineProgressEvent(image_id=image_id, kind=f"{task_name}_done"))
                continue

            # Get image size for coordinate validation
            try:
                image_width, image_height = _get_image_size(task.frame_path)
            except Exception as exc:
                logger.warning("Cannot get image size for %s: %s", image_id, exc)
                image_width, image_height = 0, 0

            # Try to reuse existing valid output (partial resume within a task)
            existing = load_existing_output(dest)
            if existing is None:
                existing = load_existing_output(legacy_dest)
            if existing is not None:
                try:
                    _run_task_validation(
                        result=existing, task_name=task_name,
                        image_width=image_width, image_height=image_height, cfg=cfg,
                    )
                    logger.debug("RESUME existing valid %s/%s", task_name, image_id)
                    if not dest.exists():
                        atomic_save_llama_factory_record(existing, task.frame_path, dest)
                    progress_queue.put(PipelineProgressEvent(image_id=image_id, kind=f"{task_name}_done"))
                    if judge_queue is not None and task_name in cfg.judge.tasks:
                        _enqueue_judge(
                            judge_queue,
                            {
                                "task_name": task_name, "image_id": image_id,
                                "relative_path": relative_path,
                                "original_result": existing, "images": task.images,
                            },
                            state_logger=state_logger,
                        )
                    continue
                except ValidationError:
                    pass  # existing file is invalid, re-process

            result = _vlm_call_with_retry(
                task_name=task_name,
                image_id=image_id,
                relative_path=relative_path,
                system_text=task.system_text,
                user_text=task.user_text,
                images=task.images,
                endpoints=None if vlm_pool is not None else cfg.vlm.endpoints,
                endpoint_pool=vlm_pool,
                api_token=api_token,
                model=model,
                state_logger=state_logger,
                max_attempts=cfg.retry.max_attempts,
                backoff_seconds=cfg.retry.backoff_seconds,
                timeout_s=cfg.vlm.timeout_s,
                worker_id=worker_id,
            )
            if result is None:
                # All retries exhausted — already logged in state/failed
                progress_queue.put(PipelineProgressEvent(image_id=image_id, kind="image_failed"))
                continue

            ok = _validate_and_save(
                result=result,
                task_name=task_name,
                image_id=image_id,
                relative_path=relative_path,
                image_path=task.frame_path,
                dest=dest,
                image_width=image_width,
                image_height=image_height,
                cfg=cfg,
                state_logger=state_logger,
            )
            if not ok:
                # Validation failed after all retries — log to failed
                state_logger.failed(
                    image_id=image_id,
                    relative_path=relative_path,
                    stage="validation",
                    task_name=task_name,
                    reason="Validation failed after VLM retries exhausted",
                    attempt=cfg.retry.max_attempts,
                )
                progress_queue.put(PipelineProgressEvent(image_id=image_id, kind="image_failed"))
                continue

            state_logger.event(
                image_id=image_id,
                relative_path=relative_path,
                stage="vlm",
                task_name=task_name,
                event="done",
                worker_id=worker_id,
                output_path=str(dest),
            )
            progress_queue.put(PipelineProgressEvent(image_id=image_id, kind=f"{task_name}_done"))

            # Enqueue Judge if enabled
            if judge_queue is not None and task_name in cfg.judge.tasks:
                _enqueue_judge(
                    judge_queue,
                    {
                        "task_name": task_name, "image_id": image_id,
                        "relative_path": relative_path,
                        "original_result": result, "images": task.images,
                    },
                    state_logger=state_logger,
                )

        except Exception as exc:
            logger.exception("Unexpected error in VLM worker for %s: %s", item, exc)
            errors.put(exc)
        finally:
            input_queue.task_done()


def _bpc_judge_worker(
    *,
    input_queue: queue.Queue,
    api_token: str,
    model: str,
    cfg: BpcConfig,
    dirs: dict[str, Path],
    state_logger: StateLogger,
    prompts: dict[str, Any],
    worker_id: str,
    judge_pool: Any = None,
) -> None:
    """Judge worker: reads judge_queue items, runs VLM Judge, saves judged outputs."""
    while True:
        item = input_queue.get()
        try:
            if item is SENTINEL:
                return
            task_name = item["task_name"]
            image_id = item["image_id"]
            relative_path = item["relative_path"]
            original_result = item["original_result"]
            images = item["images"]

            judged_dir_key = f"{task_name}_judged"
            judged_dir = dirs.get(judged_dir_key)
            if judged_dir is None:
                judged_dir = dirs["bboxing"].parent / f"{task_name}-judged"
                judged_dir.mkdir(parents=True, exist_ok=True)
            judged_dest = judged_dir / f"{image_id}.jsonl"

            _judge_task(
                task_name=task_name,
                image_id=image_id,
                relative_path=relative_path,
                original_result=original_result,
                judged_dest=judged_dest,
                images=images,
                cfg=cfg,
                state_logger=state_logger,
                api_token=api_token,
                model=model,
                prompts=prompts,
                worker_id=worker_id,
                endpoint_pool=judge_pool,
            )
        except Exception as exc:
            logger.exception("Unexpected error in Judge worker: %s", exc)
        finally:
            input_queue.task_done()


def _run_task_validation(
    *,
    result: dict[str, Any],
    task_name: str,
    image_width: int,
    image_height: int,
    cfg: BpcConfig,
) -> None:
    """Runs validation for a task. Raises ValidationError on failure."""
    if task_name == "bboxing":
        validate_bbox_result(result, image_width=image_width, image_height=image_height,
                             cfg=cfg.validation.bbox)
    elif task_name == "pointing":
        validate_pointing_result(result, image_width=image_width, image_height=image_height,
                                 cfg=cfg.validation.pointing)
    elif task_name == "captions":
        validate_caption_result(result, cfg=cfg.validation.captions)
