import logging
import os
import queue
import threading
from pathlib import Path

import click
from dotenv import load_dotenv
from tqdm import tqdm

from src.sam2 import Sam2Segmenter
from src.utils import _collect_input_images
from src.structs import FrameTask
from src.vlm import load_yaml
from src.yolo import YoloV8

from src.workers import (
    yolo_worker,
    yolo_worker_batched,
    sam_worker,
    vlm_worker,
    SENTINEL,
    progress_monitor_worker,
)


logger = logging.getLogger("pipeline")


def init_models(models_cfg: dict[str, object] | None = None) -> tuple[Sam2Segmenter, YoloV8]:
    """Initializes SAM2 and YOLOv8 models from the ``models`` config section."""

    cfg = models_cfg or {}
    device = str(cfg.get("device", "cuda"))
    sam2_dtype = cfg.get("sam2_dtype")
    yolo_model_path = cfg.get("yolo_model")
    yolo_batch_size = int(cfg.get("yolo_batch_size", 1))
    empty_cache = bool(cfg.get("empty_cache", False))

    segmenter = Sam2Segmenter(device=device, dtype=sam2_dtype, empty_cache=empty_cache)
    yolo_model = YoloV8(
        model_path=yolo_model_path,
        device=device,
        batch_size=yolo_batch_size,
        empty_cache=empty_cache,
    )

    return segmenter, yolo_model


def _require_env(name: str, required: bool = True) -> str:
    """Reads an environment variable and optionally requires it."""

    value = os.getenv(name)
    if required and not value:
        raise RuntimeError(f"{name} must be set.")

    return value or ""


def run_pipeline_with_queues(
    *,
    saved: list[Path],
    segmenter: Sam2Segmenter,
    yolo_model: YoloV8,
    prompts: dict[str, object],
    base_url: str,
    api_token: str,
    model_vlm: str,
    fail_fast: bool,
    vlm_workers_number: int,
    yolo_batch_size: int = 1,
    output_dir: Path = Path("outputs"),
    finalize_outputs: bool = True,
) -> None:
    """Runs the pipeline through in-process queues and worker threads.

    When ``yolo_batch_size > 1`` the batched YOLO worker is used to group frames
    into a single GPU inference call; otherwise the per-image worker is used.
    ``output_dir`` sets where stage JSONL files are written. When
    ``finalize_outputs`` is true, per-image shards are combined into one
    LLaMA Factory dataset per task after all workers stop.
    """

    yolo_queue: queue.Queue = queue.Queue(maxsize=16)
    sam_queue: queue.Queue = queue.Queue(maxsize=16)
    vlm_queue: queue.Queue = queue.Queue(maxsize=64)
    progress_queue: queue.Queue = queue.Queue()
    errors: queue.Queue = queue.Queue()

    stop_event = threading.Event()

    yolo_kwargs = {
        "input_queue": yolo_queue,
        "output_queue": sam_queue,
        "vlm_queue": vlm_queue,
        "yolo_model": yolo_model,
        "prompts": prompts,
        "fail_fast": fail_fast,
        "stop_event": stop_event,
        "errors": errors,
        "progress_queue": progress_queue,
    }
    if yolo_batch_size > 1:
        yolo_target = yolo_worker_batched
        yolo_kwargs["batch_size"] = yolo_batch_size
    else:
        yolo_target = yolo_worker

    yolo_thread = threading.Thread(
        target=yolo_target,
        kwargs=yolo_kwargs,
        name="yolo-worker",
        daemon=True,
    )

    sam_thread = threading.Thread(
        target=sam_worker,
        kwargs={
            "input_queue": sam_queue,
            "vlm_queue": vlm_queue,
            "segmenter": segmenter,
            "prompts": prompts,
            "fail_fast": fail_fast,
            "stop_event": stop_event,
            "errors": errors,
            "progress_queue": progress_queue,
        },
        name="sam-worker",
        daemon=True,
    )

    vlm_threads = [
        threading.Thread(
            target=vlm_worker,
            kwargs={
                "input_queue": vlm_queue,
                "base_url": base_url,
                "api_token": api_token,
                "model_vlm": model_vlm,
                "fail_fast": fail_fast,
                "stop_event": stop_event,
                "errors": errors,
                "progress_queue": progress_queue,
                "output_dir": output_dir,
            },
            name=f"vlm-worker-{i}",
            daemon=True,
        )
        for i in range(vlm_workers_number)
    ]

    progress_thread = threading.Thread(
        target=progress_monitor_worker,
        kwargs={
            "total_images": len(saved),
            "progress_queue": progress_queue,
        },
        name="progress-monitor",
        daemon=True,
    )

    yolo_thread.start()
    sam_thread.start()
    for thread in vlm_threads:
        thread.start()
    progress_thread.start()

    for frame_path in tqdm(saved, total=len(saved), desc="Submitting frames"):
        yolo_queue.put(FrameTask(image_id=frame_path.stem, frame_path=frame_path))

    logger.info("All frames submitted to yolo_queue.")

    yolo_queue.put(SENTINEL)
    logger.info("Waiting for yolo_queue to drain...")
    yolo_queue.join()
    yolo_thread.join()
    logger.info("yolo_queue drained.")

    sam_queue.put(SENTINEL)
    logger.info("Waiting for sam_queue to drain...")
    sam_queue.join()
    sam_thread.join()
    logger.info("sam_queue drained.")

    for _ in range(vlm_workers_number):
        vlm_queue.put(SENTINEL)
    logger.info("Waiting for vlm_queue to drain...")
    vlm_queue.join()
    for thread in vlm_threads:
        thread.join()
    logger.info("vlm_queue drained.")

    logger.info("Waiting for progress queue to drain...")
    progress_queue.join()
    progress_queue.put(SENTINEL)
    progress_thread.join()
    logger.info("Progress monitoring completed.")

    if not errors.empty() and fail_fast:
        raise RuntimeError("Pipeline stopped because one of the workers failed.") from errors.get()

    if finalize_outputs:
        from src.bpc.storage import compile_llama_factory_jsonl

        for task_name in ("bboxing", "captions", "pointing"):
            compile_llama_factory_jsonl(
                output_dir / task_name,
                output_dir / f"{task_name}.jsonl",
            )


@click.command()
@click.argument("data_path", type=click.Path(exists=True), required=False)
@click.option("--img-path", type=click.Path(exists=True))
@click.option("--fail-fast/--no-fail-fast", default=False, help="Stop on first error.")
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
)
@click.option(
    "--vlm-workers",
    type=int,
    default=None,
    help="Number of VLM worker threads (overrides config models.vlm_workers).",
)
@click.option(
    "--gpus",
    type=str,
    default=None,
    help="GPUs for multi-GPU mode: '0,1,2,3' or 'all'. Overrides config models.gpus.",
)
@click.option(
    "--num-nodes",
    type=int,
    default=1,
    show_default=True,
    help="Total number of machines (nodes) sharing this dataset.",
)
@click.option(
    "--node-rank",
    type=int,
    default=0,
    show_default=True,
    help="0-based index of THIS machine among --num-nodes.",
)
@click.option(
    "--output-dir",
    type=str,
    default=None,
    help="Directory for LLaMA Factory JSONL outputs (overrides config models.output_dir).",
)
def main(
    data_path: str | None,
    img_path: str | None,
    fail_fast: bool,
    log_level: str,
    vlm_workers: int | None,
    gpus: str | None,
    num_nodes: int,
    node_rank: int,
    output_dir: str | None,
) -> None:
    """Runs the full image pipeline with in-process queues between stages."""

    if data_path and img_path:
        raise ValueError("Cannot process both img_path and data_path. Choose one.")

    if num_nodes < 1:
        raise ValueError("--num-nodes must be >= 1.")
    if not (0 <= node_rank < num_nodes):
        raise ValueError("--node-rank must be in [0, --num-nodes).")

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    load_dotenv()

    base_url = _require_env("BASE_URL")
    api_token = _require_env("API_TOKEN", required=False)
    model_vlm = _require_env("MODEL_VLM")

    config = load_yaml("configs/config.yaml")
    prompts = load_yaml("configs/prompts.yaml")

    resize_cfg = config["resize"]
    models_cfg = config.get("models", {})
    processed_images_dir = Path(config["pipeline"]["processed_images_dir"])

    # CLI flag wins over config; config wins over the built-in default.
    if vlm_workers is None:
        vlm_workers = int(models_cfg.get("vlm_workers", 16))

    yolo_batch_size = int(models_cfg.get("yolo_batch_size", 1))

    out_dir = Path(output_dir or models_cfg.get("output_dir", "outputs"))

    gpus_spec = gpus if gpus is not None else models_cfg.get("gpus")

    input_data_path = data_path or config["pipeline"]["raw_images_dir"]
    saved = _collect_input_images(
        data_path=input_data_path,
        img_path=img_path,
        processed_images_dir=processed_images_dir,
        resize_cfg=resize_cfg,
    )

    logger.info("Processed %d image(s).", len(saved))
    if not saved:
        logger.warning("No frames found for processing.")
        return

    # Decide single-process vs multi-GPU. Multi-GPU is used when more than one
    # GPU is resolved; otherwise we keep the simple in-process pipeline.
    from src.multigpu import resolve_gpu_ids, run_multigpu

    gpu_ids = resolve_gpu_ids(gpus_spec)

    if len(gpu_ids) > 1 or num_nodes > 1:
        if not gpu_ids:
            raise RuntimeError(
                "Multi-node requested but no CUDA GPUs resolved. Set --gpus or run on a GPU host."
            )
        logger.info(
            "Multi-GPU mode: gpus=%s, node %d/%d", gpu_ids, node_rank, num_nodes
        )
        run_multigpu(
            frames=saved,
            gpu_ids=gpu_ids,
            prompts=prompts,
            base_url=base_url,
            api_token=api_token,
            model_vlm=model_vlm,
            fail_fast=fail_fast,
            vlm_workers_number=vlm_workers,
            yolo_batch_size=yolo_batch_size,
            output_dir=out_dir,
            models_cfg=models_cfg,
            num_nodes=num_nodes,
            node_rank=node_rank,
        )
        return

    logger.info("Single-process mode. Initializing models...")
    segmenter, yolo_model = init_models(models_cfg)
    logger.info("SAM2 and YOLOv8 initialized successfully.")

    run_pipeline_with_queues(
        saved=saved,
        segmenter=segmenter,
        yolo_model=yolo_model,
        prompts=prompts,
        base_url=base_url,
        api_token=api_token,
        model_vlm=model_vlm,
        fail_fast=fail_fast,
        vlm_workers_number=vlm_workers,
        yolo_batch_size=yolo_batch_size,
        output_dir=out_dir,
    )


if __name__ == "__main__":
    main()
