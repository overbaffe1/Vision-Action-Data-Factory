"""Multi-GPU / multi-node orchestration.

Design: **one process per GPU runs the full pipeline on its shard.**

Rather than sharing one model across GIL-bound threads (the original design,
which serialized GPU work and used a single card), we spawn one OS process per
GPU. Each process:

  1. pins itself to its GPU,
  2. loads its own YOLO + SAM2 models,
  3. runs the existing queue-based pipeline on its shard of frames,
  4. writes LLaMA Factory JSONL shards to the shared ``output_dir``.

This sidesteps the GIL entirely, scales linearly with GPU count, and is
naturally multi-node: every machine runs the same command with its own
``node_rank``; the node-level shard guarantees disjoint work (see
``src.sharding``).

The spawned worker function lives at module level so it is picklable by
``torch.multiprocessing.spawn`` / ``multiprocessing`` with the spawn context.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .sharding import GpuShard, normalize_device, parse_gpu_ids, plan_shards

logger = logging.getLogger("pipeline")


def resolve_gpu_ids(spec: str | None) -> list[str]:
    """Resolve a ``--gpus`` spec to concrete GPU ids.

    ``"all"`` / empty queries ``torch.cuda.device_count()`` at runtime. Anything
    else is parsed by the pure ``parse_gpu_ids``. Returns ``[]`` when no CUDA
    GPUs are available (caller falls back to single-process execution).
    """
    explicit = parse_gpu_ids(spec)
    if explicit:
        return explicit

    # "all" or empty -> discover.
    try:
        import torch

        count = torch.cuda.device_count()
    except Exception:
        count = 0
    return [str(i) for i in range(count)]


def _shard_worker(
    rank: int,
    shards: list[GpuShard],
    *,
    prompts: dict,
    base_url: str,
    api_token: str,
    model_vlm: str,
    fail_fast: bool,
    vlm_workers_number: int,
    yolo_batch_size: int,
    output_dir: str,
    models_cfg: dict,
) -> None:
    """Entry point for one spawned process; runs the pipeline on ``shards[rank]``.

    Imports torch/runner lazily so this module stays importable without GPU deps.
    """
    shard = shards[rank]
    device = normalize_device(shard.gpu_id)

    # Pin this process to a single visible GPU so model.to("cuda") is correct and
    # processes never contend for the same card.
    if device.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
        # After masking, the only visible device is index 0.
        device = "cuda:0"

    import runner  # lazy: pulls torch/models

    if not shard.items:
        logger.info("GPU worker rank=%s (%s) has no frames; exiting.", rank, shard.gpu_id)
        return

    cfg = dict(models_cfg)
    cfg["device"] = device
    cfg["yolo_batch_size"] = yolo_batch_size

    logger.info(
        "GPU worker rank=%s gpu=%s device=%s frames=%d",
        rank,
        shard.gpu_id,
        device,
        len(shard.items),
    )

    segmenter, yolo_model = runner.init_models(cfg)
    runner.run_pipeline_with_queues(
        saved=[Path(p) for p in shard.items],
        segmenter=segmenter,
        yolo_model=yolo_model,
        prompts=prompts,
        base_url=base_url,
        api_token=api_token,
        model_vlm=model_vlm,
        fail_fast=fail_fast,
        vlm_workers_number=vlm_workers_number,
        yolo_batch_size=yolo_batch_size,
        output_dir=Path(output_dir),
        finalize_outputs=False,
    )


def run_multigpu(
    *,
    frames: list[Path],
    gpu_ids: list[str],
    prompts: dict,
    base_url: str,
    api_token: str,
    model_vlm: str,
    fail_fast: bool,
    vlm_workers_number: int,
    yolo_batch_size: int,
    output_dir: Path,
    models_cfg: dict,
    num_nodes: int = 1,
    node_rank: int = 0,
) -> list[GpuShard]:
    """Spawn one process per GPU and run the pipeline over this node's shard.

    Returns the list of ``GpuShard`` this node planned (useful for tests/logging).
    Raises if ``gpu_ids`` is empty (caller should fall back to single-process).
    """
    if not gpu_ids:
        raise ValueError("run_multigpu requires at least one GPU id")

    shards = plan_shards(
        frames, gpu_ids=gpu_ids, num_nodes=num_nodes, node_rank=node_rank
    )
    n_local = sum(len(s.items) for s in shards)
    logger.info(
        "Node %d/%d: %d frames across GPUs %s",
        node_rank,
        num_nodes,
        n_local,
        gpu_ids,
    )

    _spawn_with_kwargs(
        shards=shards,
        prompts=prompts,
        base_url=base_url,
        api_token=api_token,
        model_vlm=model_vlm,
        fail_fast=fail_fast,
        vlm_workers_number=vlm_workers_number,
        yolo_batch_size=yolo_batch_size,
        output_dir=str(output_dir),
        models_cfg=models_cfg,
    )

    # Child processes write disjoint atomic shards. Only the parent compiles
    # shared task-level files, avoiding concurrent replacement races.
    if num_nodes == 1:
        from .bpc.storage import compile_llama_factory_jsonl

        for task_name in ("bboxing", "captions", "pointing"):
            compile_llama_factory_jsonl(
                output_dir / task_name,
                output_dir / f"{task_name}.jsonl",
            )
    else:
        logger.info(
            "Skipping final JSONL compilation in multi-node mode; compile after all nodes finish."
        )

    return shards


def _spawn_with_kwargs(*, shards: list[GpuShard], **kwargs) -> None:
    """Spawn processes passing keyword args via a positional tuple.

    ``torch.multiprocessing.spawn`` only forwards positional args (after ``rank``)
    and does not accept keyword args, so we wrap ``_shard_worker`` to unpack a
    single kwargs dict.
    """
    import torch.multiprocessing as mp

    mp.spawn(
        _shard_worker_entry,
        args=(shards, kwargs),
        nprocs=len(shards),
        join=True,
    )


def _shard_worker_entry(rank: int, shards: list[GpuShard], kwargs: dict) -> None:
    """Adapter so spawn can forward keyword args as a single picklable dict."""
    _shard_worker(rank, shards, **kwargs)
