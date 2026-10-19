#!/usr/bin/env python3
"""espada-bpc CLI — scalable BPC pipeline entry point.

Commands:
  run            Run the full BPC annotation pipeline.
  prepare-images Resize/convert raw images to the required format.
  status         Show progress of a running or finished pipeline run.
  retry-failed   Re-queue tasks from logs/failed.jsonl for a given stage/task.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from pathlib import Path

import click
from dotenv import load_dotenv

from src.vlm.prompts_processing import load_yaml


logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_images(
    input_dir: Path,
    allowed_extensions: list[str],
    *,
    recursive: bool = True,
) -> list[Path]:
    exts = {e.lower() for e in allowed_extensions}
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in exts)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """ESPADA BPC — scalable image annotation pipeline."""


# ---------------------------------------------------------------------------
# prepare-images
# ---------------------------------------------------------------------------

@cli.command("prepare-images")
@click.option("--input", "input_dir", required=True, type=click.Path(exists=True),
              help="Directory with raw images.")
@click.option("--output", "output_dir", required=True, type=click.Path(),
              help="Destination directory for prepared JPEG images.")
@click.option("--size", default=768, show_default=True,
              help="Maximum side length in pixels.")
@click.option("--format", "fmt", default="jpg", show_default=True,
              type=click.Choice(["jpg", "jpeg"]), help="Output format.")
@click.option("--quality", default=85, show_default=True,
              help="JPEG quality (1-95).")
def prepare_images(input_dir: str, output_dir: str, size: int, fmt: str, quality: int) -> None:
    """Resize and convert raw images to JPEG, applying EXIF orientation."""
    from PIL import Image, ExifTags

    src = Path(input_dir)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    supported = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
    files = sorted(p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in supported)

    if not files:
        click.echo(f"No supported images found in {src}", err=True)
        raise SystemExit(1)

    ok = 0
    skipped = 0
    errors = 0
    for src_path in files:
        rel = src_path.relative_to(src)
        dst_path = dst / rel.with_suffix(".jpg")
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if dst_path.exists():
            skipped += 1
            continue
        try:
            with Image.open(src_path) as img:
                # Apply EXIF orientation
                try:
                    exif = img._getexif()
                    if exif:
                        orientation_key = next(
                            k for k, v in ExifTags.TAGS.items() if v == "Orientation"
                        )
                        orientation = exif.get(orientation_key)
                        _ROTATIONS = {3: 180, 6: 270, 8: 90}
                        if orientation in _ROTATIONS:
                            img = img.rotate(_ROTATIONS[orientation], expand=True)
                except Exception:
                    pass

                img = img.convert("RGB")

                # Resize if needed
                w, h = img.size
                max_side = max(w, h)
                if max_side > size:
                    scale = size / max_side
                    new_w, new_h = int(w * scale), int(h * scale)
                    img = img.resize((new_w, new_h), Image.LANCZOS)

                img.save(dst_path, format="JPEG", quality=quality, optimize=True,
                         progressive=True)
                ok += 1
        except Exception as exc:
            click.echo(f"ERROR {src_path}: {exc}", err=True)
            errors += 1

    click.echo(f"prepare-images: {ok} converted, {skipped} skipped, {errors} errors.")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@cli.command("run")
@click.option("--input", "input_dir", required=True, type=click.Path(exists=True),
              help="Directory with already-prepared JPEG images.")
@click.option("--output", "output_root", required=True, type=click.Path(),
              help="Root output directory (outputs/, logs/, visualizations/ go here).")
@click.option("--config", "config_path", default="configs/bpc.yaml", show_default=True,
              type=click.Path(), help="Path to bpc.yaml config.")
@click.option("--log-level", default="INFO", show_default=True,
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
@click.option("--gpus", default=None, type=str,
              help="GPU spec: '0,1' or 'all'. Overrides config.")
@click.option("--num-nodes", default=1, show_default=True, type=int,
              help="Total number of machines.")
@click.option("--node-rank", default=0, show_default=True, type=int,
              help="0-based index of this machine.")
@click.option("--batch-size", default=1000, show_default=True, type=int,
              help="Images per index batch for checkpoint/resume (progress.json).")
@click.option("--input-recursive/--no-input-recursive", default=True, show_default=True,
              help="Recursively scan --input. Disable for flat huge NFS directories.")
@click.option("--skip-preflight", is_flag=True,
              help="Skip full cv2 preflight over every image before processing.")
def run(
    input_dir: str,
    output_root: str,
    config_path: str,
    log_level: str,
    gpus: str | None,
    num_nodes: int,
    node_rank: int,
    batch_size: int,
    input_recursive: bool,
    skip_preflight: bool,
) -> None:
    """Run the full BPC pipeline on a prepared image directory."""
    from src.bpc.config import load_bpc_config
    from src.bpc.logging_setup import setup_bpc_logging
    from src.bpc.storage import setup_output_dirs
    from src.bpc.state import StateLogger
    from src.bpc.batching import make_index_batches, ProgressCheckpoint

    if num_nodes < 1:
        raise click.BadParameter("--num-nodes must be >= 1")
    if not (0 <= node_rank < num_nodes):
        raise click.BadParameter("--node-rank must be in [0, num-nodes)")

    load_dotenv()
    cfg = load_bpc_config(config_path)

    out_root = Path(output_root)
    logs_dir = out_root / "logs"
    setup_bpc_logging(logs_dir, console_level=log_level)

    # Save config snapshot
    out_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, out_root / "config.yaml")

    run_id = str(uuid.uuid4())
    state_logger = StateLogger(logs_dir, run_id)

    api_token = os.getenv("API_TOKEN", "")
    from src.bpc.endpoints import build_targets
    vlm_targets = build_targets(
        targets=cfg.vlm.targets,
        endpoints=cfg.vlm.endpoints,
        router_url=cfg.vlm.router_url,
        models=cfg.vlm.models,
    )
    targets_embed_model = any(target.model for target in vlm_targets)

    if targets_embed_model:
        # Router-style targets carry their own model name via EndpointTarget,
        # so a single fixed vlm.model is not required.
        model = cfg.vlm.model or ""
    else:
        model = cfg.vlm.model or os.getenv("MODEL_VLM", "")
        if not model:
            raise click.ClickException(
                "MODEL_VLM env var or vlm.model in config must be set "
                "(unless using router_url + models)."
            )
        cfg.vlm.model = model

        # Override endpoints from env if needed
        base_url = os.getenv("BASE_URL", "")
        if base_url and cfg.vlm.endpoints == ["http://localhost:8001"]:
            cfg.vlm.endpoints = [base_url]

    # Auto-scale VLM worker threads to saturate all endpoints. When explicit
    # targets carry per-target caps, total capacity is the SUM of those caps;
    # otherwise it falls back to N * default_cap.
    capacity = sum(
        target.max_concurrency or cfg.vlm.max_concurrency_per_endpoint
        for target in vlm_targets
    )
    if cfg.vlm.workers < capacity:
        logger.info(
            "Auto-scaling vlm.workers %d -> %d to match endpoint capacity "
            "(%d targets, total capacity=%d).",
            cfg.vlm.workers, capacity, len(vlm_targets), capacity,
        )
        cfg.vlm.workers = capacity

    # Collect images
    src = Path(input_dir)
    logger.info(
        "Collecting images from %s (recursive=%s, extensions=%s)",
        src, input_recursive, cfg.input.allowed_extensions,
    )
    images = _collect_images(src, cfg.input.allowed_extensions, recursive=input_recursive)
    logger.info("Found %d images in %s", len(images), src)

    if not images:
        logger.warning("No images found. Exiting.")
        return

    # Multi-node sharding
    if num_nodes > 1:
        from src.sharding import node_shard
        images = node_shard(images, num_nodes, node_rank)
        logger.info("Node %d/%d: processing %d images", node_rank, num_nodes, len(images))

    prompts = load_yaml("configs/prompts.yaml")
    if skip_preflight:
        logger.warning(
            "Skipping full image preflight for %d images; corrupt/unreadable files "
            "will be handled by worker-stage failures.",
            len(images),
        )
        valid_images = images
    else:
        # Preflight
        from src.bpc.preflight import run_preflight
        valid_images, _ = run_preflight(images, cfg.input,
                                        preflight_log=logs_dir / "preflight.log")

    if not valid_images:
        logger.warning("No valid images after preflight. Exiting.")
        return

    # Setup output dirs (also needed by the multi-GPU path before spawning,
    # so every child process finds the directories already created).
    dirs = setup_output_dirs(out_root, save_visualizations=cfg.storage.save_visualizations)

    # --- Multi-GPU: one process per GPU, each running the full BPC pipeline
    # on its shard (see src/bpc/multigpu.py). Falls back to the single-process
    # path below only when exactly one GPU is resolved and num_nodes == 1.
    # --gpus explicitly passed wins; otherwise default to "all" (auto-discover
    # every visible CUDA device) rather than cfg.yolo.devices, which is a
    # single-process device-pinning setting, not a multi-GPU fan-out spec.
    from src.multigpu import resolve_gpu_ids
    gpu_ids = resolve_gpu_ids(gpus if gpus is not None else "all")

    if len(gpu_ids) > 1 or num_nodes > 1:
        if not gpu_ids:
            raise click.ClickException(
                "Multi-node requested but no CUDA GPUs resolved. Set --gpus or run on a GPU host."
            )
        logger.info("Multi-GPU mode: gpus=%s, node %d/%d", gpu_ids, node_rank, num_nodes)
        from src.bpc.multigpu import run_bpc_multigpu
        run_bpc_multigpu(
            frames=valid_images,
            gpu_ids=gpu_ids,
            output_root=out_root,
            cfg_path=config_path,
            api_token=api_token,
            run_id=run_id,
            batch_size=batch_size,
            num_nodes=num_nodes,
            node_rank=node_rank,
        )
        logger.info("All pending batches complete (multi-GPU).")
        return

    # --- Single-process path (1 GPU, 1 node) ---
    from runner import init_models
    from src.bpc.pipeline import run_bpc_pipeline
    models_cfg: dict = {
        "device": gpu_ids[0] if gpu_ids else (cfg.yolo.devices[0] if cfg.yolo.devices else "cuda"),
        "sam2_dtype": cfg.sam2.dtype,
        "yolo_model": cfg.yolo.model,
        "yolo_batch_size": cfg.yolo.batch_size,
        "empty_cache": cfg.sam2.empty_cache,
    }
    if models_cfg["device"].isdigit():
        models_cfg["device"] = f"cuda:{models_cfg['device']}"
    logger.info("Initializing models on %s...", models_cfg["device"])
    segmenter, yolo_model = init_models(models_cfg)
    logger.info("Models ready.")

    # --- Index-based batching with checkpoint/resume ---
    # Build numbered batches over the deterministically-sorted image list so a
    # restart can skip completed batches without scanning millions of files.
    shard_key = f"node{node_rank}"
    progress_path = logs_dir / "progress.json"
    checkpoint = ProgressCheckpoint(progress_path)
    checkpoint.init_run(total_images=len(valid_images), batch_size=batch_size)

    batches = make_index_batches(valid_images, batch_size)
    pending = checkpoint.pending_batches(batches)
    logger.info(
        "Batches: %d total, %d already done, %d pending (batch_size=%d).",
        len(batches), len(batches) - len(pending), len(pending), batch_size,
    )

    for batch in pending:
        logger.info(
            "Processing batch %d [%d:%d] (%d images)...",
            batch.batch_id, batch.start, batch.end, batch.size,
        )
        run_bpc_pipeline(
            images=batch.paths,
            output_root=out_root,
            cfg=cfg,
            prompts=prompts,
            api_token=api_token,
            segmenter=segmenter,
            yolo_model=yolo_model,
            dirs=dirs,
            state_logger=state_logger,
            run_id=run_id,
        )
        checkpoint.mark_batch_done(batch.batch_id, shard_key=shard_key)
        logger.info("Batch %d done. Checkpoint advanced to %d.",
                    batch.batch_id, checkpoint.last_completed_batch)

    from src.bpc.storage import finalize_llama_factory_outputs

    counts = finalize_llama_factory_outputs(dirs)
    logger.info("LLaMA Factory datasets compiled: %s", counts)
    logger.info("All pending batches complete.")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@cli.command("status")
@click.option("--run", "run_dir", required=True, type=click.Path(exists=True),
              help="Output root directory of a pipeline run.")
def status(run_dir: str) -> None:
    """Show progress summary for a pipeline run."""
    from src.bpc.status import print_status
    print_status(Path(run_dir))


# ---------------------------------------------------------------------------
# retry-failed
# ---------------------------------------------------------------------------

@cli.command("retry-failed")
@click.option("--run", "run_dir", required=True, type=click.Path(exists=True),
              help="Output root directory of the pipeline run.")
@click.option("--stage", default=None, type=str, help="Filter by stage (e.g. 'vlm').")
@click.option("--task", default=None, type=str,
              help="Filter by task_name (e.g. 'bboxing').")
def retry_failed(run_dir: str, stage: str | None, task: str | None) -> None:
    """Re-queue failed tasks by removing their state entries so the next run retries them."""
    failed_path = Path(run_dir) / "logs" / "failed.jsonl"
    state_path = Path(run_dir) / "logs" / "state.jsonl"

    if not failed_path.exists():
        click.echo("No failed.jsonl found. Nothing to retry.")
        return

    with failed_path.open(encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    to_retry = [
        e for e in entries
        if (stage is None or e.get("stage") == stage)
        and (task is None or e.get("task_name") == task)
    ]

    if not to_retry:
        click.echo("No matching failed tasks found.")
        return

    # Remove matching 'done' entries from state.jsonl so resume re-runs them
    retry_keys = {(e["image_id"], e["task_name"]) for e in to_retry}
    new_state_lines = []
    if state_path.exists():
        with state_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = (rec.get("image_id"), rec.get("task_name"))
                    if key in retry_keys and rec.get("event") in ("done", "failed"):
                        continue  # drop: will be re-processed
                except json.JSONDecodeError:
                    pass
                new_state_lines.append(line)

    state_path.write_text("\n".join(new_state_lines) + "\n", encoding="utf-8")

    # Remove output files for retry tasks so validation doesn't skip them
    outputs_root = Path(run_dir) / "outputs"
    removed_files = 0
    for e in to_retry:
        task_name = e.get("task_name", "")
        image_id = e.get("image_id", "")
        for suffix in (".jsonl", ".json"):
            out_file = outputs_root / task_name / f"{image_id}{suffix}"
            if out_file.exists():
                out_file.unlink()
                removed_files += 1

    # Truncate failed.jsonl to remove the retried entries
    remaining = [e for e in entries if (e["image_id"], e["task_name"]) not in retry_keys]
    with failed_path.open("w", encoding="utf-8") as f:
        for e in remaining:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # Keep aggregate datasets consistent with the removed per-image shards.
    from src.bpc.storage import compile_llama_factory_jsonl

    for task_dir in outputs_root.iterdir() if outputs_root.exists() else ():
        if task_dir.is_dir():
            compile_llama_factory_jsonl(task_dir, outputs_root / f"{task_dir.name}.jsonl")

    click.echo(
        f"Queued {len(to_retry)} task(s) for retry "
        f"(removed {removed_files} output file(s), cleaned state.jsonl)."
    )
    click.echo("Run 'espada-bpc run ...' again to re-process them.")


if __name__ == "__main__":
    cli()
