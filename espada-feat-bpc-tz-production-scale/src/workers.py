from __future__ import annotations

import queue
import logging
from typing import Callable
from .yolo import YoloV8
from .sam2 import Sam2Segmenter
import numpy as np
from tqdm import tqdm
from pathlib import Path
from .utils import (
    build_extended_detections,
    filter_big_bboxes,
    draw_detections_cv2,
    _try_save_vlm_json,
    draw_sam2_masks_with_centroids,
)
from .structs import (
    _build_bbox_vlm_task,
    _build_caption_vlm_task,
    _build_pointing_vlm_task,
    YoloStageResult,
    PipelineProgressEvent,
)
from .vlm import chat_completion
import threading
from PIL import Image
import cv2


logger = logging.getLogger("pipeline")
SENTINEL = object()


def _put_with_backpressure_log(
    q: queue.Queue,
    item: object,
    *,
    queue_name: str,
    warn_every: float = 30.0,
) -> None:
    """Blocking queue.put() that logs a warning every ``warn_every`` seconds
    while it waits, instead of blocking silently forever.

    A silent unbounded put() on a saturated downstream queue (e.g. vlm_queue
    backed by a slow/overloaded HTTP endpoint) makes a producer thread (YOLO
    or SAM2) look identical to a deadlock from the logs — there is no signal
    that it is still alive and simply waiting for consumers to catch up.
    This still blocks (no items are dropped — every frame must eventually be
    processed), it only makes the wait observable.
    """
    while True:
        try:
            q.put(item, timeout=warn_every)
            return
        except queue.Full:
            logger.warning(
                "%s is full (maxsize=%d) — producer blocked for >%.0fs waiting "
                "for consumers to catch up. Still waiting (no items dropped).",
                queue_name, q.maxsize, warn_every,
            )


def _extract_image_id(item: object) -> str | None:
    """Extracts image_id from a worker item when possible."""

    if item is SENTINEL:
        return None

    image_id = getattr(item, "image_id", None)
    if image_id:
        return image_id

    frame_path = getattr(item, "frame_path", None)
    if frame_path is not None:
        return Path(frame_path).stem

    return None


def progress_monitor_worker(
    *,
    total_images: int,
    progress_queue: queue.Queue,
) -> None:
    """Tracks completed images and updates a user-facing progress bar."""

    states: dict[str, dict[str, object]] = {}
    finalized: set[str] = set()

    with tqdm(total=total_images, desc="Completed images") as pbar:
        while len(finalized) < total_images:
            event = progress_queue.get()
            try:
                if event is SENTINEL:
                    return

                if event.image_id not in states:
                    states[event.image_id] = {
                        "bbox_done": False,
                        "caption_done": False,
                        "point_state": "unknown",  # "unknown", "done", "skipped"
                        "failed": False,
                    }

                state = states[event.image_id]

                if event.kind == "bboxing_done":
                    state["bbox_done"] = True

                elif event.kind == "captions_done":
                    state["caption_done"] = True

                elif event.kind == "pointing_done":
                    state["point_state"] = "done"

                elif event.kind == "pointing_skipped":
                    state["point_state"] = "skipped"

                elif event.kind == "image_failed":
                    state["failed"] = True

                if event.image_id in finalized:
                    continue

                if state["failed"]:
                    finalized.add(event.image_id)
                    pbar.update(1)
                    continue

                if (
                    state["bbox_done"]
                    and state["caption_done"]
                    and state["point_state"] in {"done", "skipped"}
                ):
                    finalized.add(event.image_id)
                    pbar.update(1)
            finally:
                progress_queue.task_done()


def _handle_worker_error(
    *,
    exc: Exception,
    stage_name: str,
    item: object,
    fail_fast: bool,
    stop_event: threading.Event,
    errors: queue.Queue,
    progress_queue: queue.Queue,
) -> None:
    """Logs worker errors and optionally stops the pipeline."""

    # Log the extracted image_id, not the raw item repr: FrameTask/
    # YoloStageResult carry full numpy image arrays, so "%s" % item used to
    # dump megabytes of pixel data into the log per failure — burying the
    # actual exception text and making tracebacks unreadable. exc_info=True
    # gives the real traceback instead of only str(exc) (which drops the
    # call site entirely, the reason a per-frame SAM2/YOLO crash was
    # previously undiagnosable from logs alone).
    image_id = _extract_image_id(item)
    logger.warning("%s worker failed for image_id=%s: %s", stage_name, image_id, exc, exc_info=True)
    errors.put(exc)

    if image_id is not None:
        progress_queue.put(PipelineProgressEvent(image_id=image_id, kind="image_failed"))

    if fail_fast:
        stop_event.set()


def _read_frame_images(frame_path: Path) -> tuple[np.ndarray, "Image.Image"]:
    """Reads a frame as BGR (cv2) and RGB (PIL), raising on failure."""

    original_image = cv2.imread(str(frame_path))
    if original_image is None:
        raise ValueError(f"Failed to read image: {frame_path}")

    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    return original_image, pil_image


def _emit_yolo_outputs(
    *,
    task: object,
    original_image: np.ndarray,
    pil_image: "Image.Image",
    detections_df,
    prompts: dict[str, object],
    output_queue: queue.Queue,
    vlm_queue: queue.Queue,
) -> None:
    """Shared YOLO post-processing: filter boxes, build VLM tasks, enqueue.

    Used by both the per-image and batched YOLO workers so the downstream
    behavior stays identical regardless of how inference was batched.
    """

    detections = build_extended_detections(detections_df)

    image_h, image_w = original_image.shape[:2]
    detections = filter_big_bboxes(
        image_size=(image_h, image_w),
        bboxes_info=detections,
        threshold=40,
    )

    vis_image_boxes = draw_detections_cv2(original_image, detections)

    _put_with_backpressure_log(
        vlm_queue,
        _build_bbox_vlm_task(
            frame_path=task.frame_path,
            original_image=original_image,
            detections=detections,
            vis_image_boxes=vis_image_boxes,
            prompts=prompts,
        ),
        queue_name="vlm_queue",
    )
    _put_with_backpressure_log(
        vlm_queue,
        _build_caption_vlm_task(
            frame_path=task.frame_path,
            original_image=original_image,
            prompts=prompts,
        ),
        queue_name="vlm_queue",
    )

    _put_with_backpressure_log(
        output_queue,
        YoloStageResult(
            image_id=task.image_id,
            frame_path=task.frame_path,
            original_image=original_image,
            pil_image=pil_image,
            detections=detections,
            vis_image_boxes=vis_image_boxes,
        ),
        queue_name="sam_queue",
    )


def yolo_worker(
    *,
    input_queue: queue.Queue,
    output_queue: queue.Queue,
    vlm_queue: queue.Queue,
    yolo_model: YoloV8,
    prompts: dict[str, object],
    fail_fast: bool,
    stop_event: threading.Event,
    errors: queue.Queue,
    progress_queue: queue.Queue,
) -> None:
    """Runs YOLO on incoming images one at a time and enqueues downstream tasks."""

    while True:
        # Getting FrameTask dataclass
        item = input_queue.get()

        try:
            if item is SENTINEL:
                return

            if stop_event.is_set():
                continue

            task = item

            original_image, pil_image = _read_frame_images(task.frame_path)
            detections_df = yolo_model.predict(original_image)

            _emit_yolo_outputs(
                task=task,
                original_image=original_image,
                pil_image=pil_image,
                detections_df=detections_df,
                prompts=prompts,
                output_queue=output_queue,
                vlm_queue=vlm_queue,
            )
        except Exception as exc:
            _handle_worker_error(
                exc=exc,
                stage_name="YOLO",
                item=item,
                fail_fast=fail_fast,
                stop_event=stop_event,
                errors=errors,
                progress_queue=progress_queue,
            )
        finally:
            input_queue.task_done()


def yolo_worker_batched(
    *,
    input_queue: queue.Queue,
    output_queue: queue.Queue,
    vlm_queue: queue.Queue,
    yolo_model: YoloV8,
    prompts: dict[str, object],
    fail_fast: bool,
    stop_event: threading.Event,
    errors: queue.Queue,
    progress_queue: queue.Queue,
    batch_size: int = 16,
) -> None:
    """Like ``yolo_worker`` but accumulates up to ``batch_size`` frames and runs
    a single batched GPU inference per group, improving GPU utilization.

    Frames are drained from the queue until either ``batch_size`` frames are
    collected or the SENTINEL is seen; each collected group is then processed
    with ``predict_on_batch`` and post-processed exactly like the per-image
    worker (via ``_emit_yolo_outputs``), so outputs are identical.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    done = False
    while not done:
        batch_tasks: list[object] = []
        images: list[np.ndarray] = []
        pil_images: list["Image.Image"] = []

        # Collect a batch (block for the first item, then drain without blocking).
        while len(batch_tasks) < batch_size:
            try:
                item = input_queue.get() if not batch_tasks else input_queue.get_nowait()
            except queue.Empty:
                break

            if item is SENTINEL:
                input_queue.task_done()
                done = True
                break

            if stop_event.is_set():
                input_queue.task_done()
                continue

            try:
                original_image, pil_image = _read_frame_images(item.frame_path)
            except Exception as exc:
                _handle_worker_error(
                    exc=exc,
                    stage_name="YOLO",
                    item=item,
                    fail_fast=fail_fast,
                    stop_event=stop_event,
                    errors=errors,
                    progress_queue=progress_queue,
                )
                input_queue.task_done()
                continue

            batch_tasks.append(item)
            images.append(original_image)
            pil_images.append(pil_image)

        if not batch_tasks:
            continue

        # Single batched inference for the whole group.
        try:
            detection_dfs = yolo_model.predict_on_batch(images)
        except Exception as exc:
            # Whole-batch failure: report per task so progress isn't stuck.
            for task in batch_tasks:
                _handle_worker_error(
                    exc=exc,
                    stage_name="YOLO",
                    item=task,
                    fail_fast=fail_fast,
                    stop_event=stop_event,
                    errors=errors,
                    progress_queue=progress_queue,
                )
                input_queue.task_done()
            continue

        for task, original_image, pil_image, detections_df in zip(
            batch_tasks, images, pil_images, detection_dfs
        ):
            try:
                _emit_yolo_outputs(
                    task=task,
                    original_image=original_image,
                    pil_image=pil_image,
                    detections_df=detections_df,
                    prompts=prompts,
                    output_queue=output_queue,
                    vlm_queue=vlm_queue,
                )
            except Exception as exc:
                _handle_worker_error(
                    exc=exc,
                    stage_name="YOLO",
                    item=task,
                    fail_fast=fail_fast,
                    stop_event=stop_event,
                    errors=errors,
                    progress_queue=progress_queue,
                )
            finally:
                input_queue.task_done()


def sam_worker(
    *,
    input_queue: queue.Queue,
    vlm_queue: queue.Queue,
    segmenter: Sam2Segmenter,
    prompts: dict[str, object],
    fail_fast: bool,
    stop_event: threading.Event,
    errors: queue.Queue,
    progress_queue: queue.Queue,
    segmenter_lock: threading.Lock | None = None,
    on_pointing_skipped: "Callable[[object], None] | None" = None,
) -> None:
    """Runs SAM2 using YOLO boxes and enqueues pointing VLM tasks.

    ``segmenter_lock``, when provided, serializes ``set_image``+``predict_from_boxes``
    across multiple sam_worker threads sharing one Sam2Segmenter instance —
    that object's ``set_image()`` mutates shared state (``self._image``), so
    concurrent callers without the lock could interleave and produce masks
    for the wrong image. The lock is held only around the GPU call, not
    around the downstream queue put, so one thread waiting on a full
    vlm_queue does not block the others from segmenting the next frame.

    ``on_pointing_skipped``, when provided, is called with the task whenever
    an image has no detections and pointing is skipped — lets callers (e.g.
    the BPC pipeline) record this as a distinct state.jsonl event instead of
    it only existing transiently on progress_queue, so `espada-bpc status`
    can tell "correctly skipped, no boxes" apart from "stuck, nothing produced".
    """

    while True:
        item = input_queue.get()
        try:
            if item is SENTINEL:
                return

            if stop_event.is_set():
                continue

            task = item
            bboxes = task.detections.get("bbox", [])
            if not bboxes:
                logger.info("No detections for %s. Pointing is skipped.", task.frame_path)
                progress_queue.put(
                    PipelineProgressEvent(image_id=task.image_id, kind="pointing_skipped")
                )
                if on_pointing_skipped is not None:
                    on_pointing_skipped(task)
                continue

            input_boxes = np.array(bboxes, dtype=np.float32)

            if segmenter_lock is not None:
                with segmenter_lock:
                    segmenter.set_image(task.pil_image)
                    out = segmenter.predict_from_boxes(input_boxes, multimask_output=False)
            else:
                segmenter.set_image(task.pil_image)
                out = segmenter.predict_from_boxes(input_boxes, multimask_output=False)
            masks = out.masks

            vis_image_points, meta = draw_sam2_masks_with_centroids(
                task.original_image,
                masks,
                alpha=0.45,
                circle_radius=8,
                text_scale=1.0,
                text_thickness=3,
                start_index=1,
            )

            _put_with_backpressure_log(
                vlm_queue,
                _build_pointing_vlm_task(
                    frame_path=task.frame_path,
                    original_image=task.original_image,
                    meta=meta,
                    vis_image_points=vis_image_points,
                    prompts=prompts,
                ),
                queue_name="vlm_queue",
            )
        except Exception as exc:
            _handle_worker_error(
                exc=exc,
                stage_name="SAM2",
                item=item,
                fail_fast=fail_fast,
                stop_event=stop_event,
                errors=errors,
                progress_queue=progress_queue,
            )
        finally:
            input_queue.task_done()


def vlm_worker(
    *,
    input_queue: queue.Queue,
    base_url: str,
    api_token: str,
    model_vlm: str,
    fail_fast: bool,
    stop_event: threading.Event,
    errors: queue.Queue,
    progress_queue: queue.Queue,
    output_dir: Path = Path("outputs"),
) -> None:
    """Consumes prepared VLM tasks, executes requests and saves results.

    Results are written to ``output_dir/<task_name>/<frame>.jsonl``. ``output_dir``
    is configurable so multi-GPU / multi-node runs can target a shared location.
    """

    while True:
        item = input_queue.get()
        try:
            if item is SENTINEL:
                logger.info("%s got SENTINEL and exits", threading.current_thread().name)
                return

            if stop_event.is_set():
                continue

            task = item
            result = chat_completion(
                base_url=base_url,
                api_token=api_token,
                model=model_vlm,
                system_text=task.system_text,
                user_text=task.user_text,
                images=task.images,
            )

            _try_save_vlm_json(
                result.content,
                frame_path=str(task.frame_path),
                out_dir=output_dir / task.task_name,
            )

            progress_queue.put(
                PipelineProgressEvent(
                    image_id=task.frame_path.stem,
                    kind=f"{task.task_name}_done",
                )
            )
        except Exception as exc:
            _handle_worker_error(
                exc=exc,
                stage_name="VLM",
                item=item,
                fail_fast=fail_fast,
                stop_event=stop_event,
                errors=errors,
                progress_queue=progress_queue,
            )
        finally:
            input_queue.task_done()
