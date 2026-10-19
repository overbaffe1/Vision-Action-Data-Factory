"""Multi-GPU / multi-node orchestration for the BPC pipeline.

Same process-per-GPU design as ``src/multigpu.py`` (see that module's
docstring for the rationale), adapted to call ``run_bpc_pipeline`` instead
of the legacy ``runner.run_pipeline_with_queues``. Each spawned process:

  1. pins itself to its GPU via CUDA_VISIBLE_DEVICES,
  2. loads its own YOLO + SAM2 models,
  3. builds its own StateLogger/EndpointPool/dirs (these hold locks/sockets
     and are not picklable across the spawn boundary, so they must be
     reconstructed in-process rather than passed in),
  4. runs the full BPC pipeline (validation + Judge + retry + resume) on its
     shard of frames, writing to the shared ``output_root``.

logs/state.jsonl and logs/failed.jsonl are opened in append ("a") mode by
each process's own StateLogger; concurrent appends of short JSON lines from
multiple processes are safe on POSIX (O_APPEND), so no cross-process lock
is required — each process's writes are self-consistent and interleave at
the line level.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from ..sharding import GpuShard, normalize_device, plan_shards

logger = logging.getLogger("bpc.multigpu")


def _bpc_shard_worker(
    rank: int,
    shards: list[GpuShard],
    *,
    output_root: str,
    cfg_path: str,
    api_token: str,
    run_id: str,
    batch_size: int,
) -> None:
    """Entry point for one spawned process; runs the BPC pipeline on ``shards[rank]``.

    Imports torch/models lazily so this module stays importable without GPU deps.
    Rebuilds cfg/state_logger/dirs from scratch (see module docstring) rather
    than receiving them pre-built, since those objects are not picklable.
    """
    shard = shards[rank]
    device = normalize_device(shard.gpu_id)

    # Pin this process to a single visible GPU so model.to("cuda") is correct
    # and processes never contend for the same card.
    if device.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
        device = "cuda:0"  # after masking, the only visible device is index 0

    if not shard.items:
        logger.info("BPC GPU worker rank=%s (%s) has no frames; exiting.", rank, shard.gpu_id)
        return

    from .config import load_bpc_config
    from .state import StateLogger
    from .storage import setup_output_dirs
    from .batching import make_index_batches, ProgressCheckpoint
    from .pipeline import run_bpc_pipeline
    from .logging_setup import setup_bpc_logging
    from ..vlm.prompts_processing import load_yaml
    from runner import init_models

    out_root = Path(output_root)
    logs_dir = out_root / "logs"

    # torch.multiprocessing.spawn starts a fresh interpreter per process — it
    # does NOT inherit the parent's configured logging handlers, so every
    # spawned worker must set up its own. Suffix by gpu_id so 4 processes
    # don't share one FileHandler (logging.FileHandler writes are not
    # guaranteed atomic across processes the way StateLogger's raw
    # append("a") is).
    setup_bpc_logging(logs_dir, file_suffix=f".gpu{shard.gpu_id}")

    state_logger = StateLogger(logs_dir, run_id)

    cfg = load_bpc_config(cfg_path)
    cfg.yolo.devices = [device]
    cfg.sam2.devices = [device]

    dirs = setup_output_dirs(out_root, save_visualizations=cfg.storage.save_visualizations)
    prompts = load_yaml("configs/prompts.yaml")

    models_cfg: dict[str, Any] = {
        "device": device,
        "sam2_dtype": cfg.sam2.dtype,
        "yolo_model": cfg.yolo.model,
        "yolo_batch_size": cfg.yolo.batch_size,
        "empty_cache": cfg.sam2.empty_cache,
    }

    logger.info(
        "BPC GPU worker rank=%s gpu=%s device=%s frames=%d",
        rank, shard.gpu_id, device, len(shard.items),
    )

    segmenter, yolo_model = init_models(models_cfg)

    # Index-batch this shard's frames so a crash mid-shard resumes from the
    # next pending batch instead of rescanning/reprocessing everything, same
    # as the single-GPU path in espada_bpc.py.
    shard_key = f"{shard.gpu_id}"
    progress_path = logs_dir / f"progress.{shard.gpu_id}.json"
    checkpoint = ProgressCheckpoint(progress_path)
    frames = [Path(p) for p in shard.items]
    checkpoint.init_run(total_images=len(frames), batch_size=batch_size)

    batches = make_index_batches(frames, batch_size)
    pending = checkpoint.pending_batches(batches)
    logger.info(
        "GPU %s: %d batches total, %d already done, %d pending.",
        shard.gpu_id, len(batches), len(batches) - len(pending), len(pending),
    )

    for batch in pending:
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

    logger.info("BPC GPU worker rank=%s (%s) finished.", rank, shard.gpu_id)


def run_bpc_multigpu(
    *,
    frames: list[Path],
    gpu_ids: list[str],
    output_root: Path,
    cfg_path: str,
    api_token: str,
    run_id: str,
    batch_size: int = 1000,
    num_nodes: int = 1,
    node_rank: int = 0,
) -> list[GpuShard]:
    """Spawn one process per GPU and run the BPC pipeline over this node's shard.

    Returns the list of ``GpuShard`` this node planned (useful for tests/logging).
    Raises if ``gpu_ids`` is empty (caller should fall back to single-process).
    """
    if not gpu_ids:
        raise ValueError("run_bpc_multigpu requires at least one GPU id")

    shards = plan_shards(frames, gpu_ids=gpu_ids, num_nodes=num_nodes, node_rank=node_rank)
    n_local = sum(len(s.items) for s in shards)
    logger.info(
        "BPC node %d/%d: %d frames across GPUs %s", node_rank, num_nodes, n_local, gpu_ids,
    )

    _spawn_with_kwargs(
        shards=shards,
        output_root=str(output_root),
        cfg_path=cfg_path,
        api_token=api_token,
        run_id=run_id,
        batch_size=batch_size,
    )

    if num_nodes == 1:
        from .storage import finalize_llama_factory_outputs, setup_output_dirs

        dirs = setup_output_dirs(output_root, save_visualizations=False)
        counts = finalize_llama_factory_outputs(dirs)
        logger.info("LLaMA Factory datasets compiled: %s", counts)
    else:
        logger.info(
            "Skipping final JSONL compilation in multi-node mode; compile after all nodes finish."
        )

    return shards


def _spawn_with_kwargs(*, shards: list[GpuShard], **kwargs: Any) -> None:
    """Spawn processes passing keyword args via a positional tuple.

    ``torch.multiprocessing.spawn`` only forwards positional args (after
    ``rank``) and does not accept keyword args, so we wrap the worker to
    unpack a single picklable kwargs dict (strings/ints/paths only — no
    locks, sockets, or dataclass instances holding those).
    """
    import torch.multiprocessing as mp

    mp.spawn(
        _shard_worker_entry,
        args=(shards, kwargs),
        nprocs=len(shards),
        join=True,
    )


def _shard_worker_entry(rank: int, shards: list[GpuShard], kwargs: dict[str, Any]) -> None:
    """Adapter so spawn can forward keyword args as a single picklable dict."""
    _bpc_shard_worker(rank, shards, **kwargs)
