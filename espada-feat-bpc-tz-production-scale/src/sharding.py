"""Pure-logic sharding for multi-GPU and multi-node execution.

The pipeline distributes a flat list of input frames across:

  * multiple **nodes** (separate machines), and
  * multiple **GPUs** within each node.

This module contains only deterministic, dependency-free helpers so the
distribution math can be unit-tested exhaustively without any GPU or torch.

Two-level sharding model
------------------------
Given ``N`` frames, ``num_nodes`` machines and a per-node GPU list, every frame
is owned by exactly one ``(node_rank, gpu_index)`` pair:

1. Node level: frames are split into ``num_nodes`` contiguous blocks; node
   ``r`` takes block ``r``.
2. GPU level: that node's block is split into ``len(gpu_ids)`` contiguous
   sub-blocks; GPU ``g`` takes sub-block ``g``.

Contiguous (not round-robin) splitting keeps each shard's frames close together
on disk, which helps OS read-ahead, and makes the math trivial to verify:
the union of all shards over all nodes equals the original list, with no
overlaps and no drops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

T = TypeVar("T")


def contiguous_block(n: int, parts: int, index: int) -> tuple[int, int]:
    """Return the ``[start, end)`` slice bounds of block ``index`` of ``parts``.

    Splits ``range(n)`` into ``parts`` contiguous blocks as evenly as possible
    (earlier blocks get the extra item when ``n`` is not divisible). This is the
    same balanced split used by the reference data pipeline:
    ``start = index*n // parts``, ``end = (index+1)*n // parts``.
    """
    if parts < 1:
        raise ValueError("parts must be >= 1")
    if not (0 <= index < parts):
        raise ValueError(f"index {index} out of range for parts={parts}")
    if n < 0:
        raise ValueError("n must be >= 0")
    start = index * n // parts
    end = (index + 1) * n // parts
    return start, end


def split_contiguous(items: Sequence[T], parts: int) -> list[list[T]]:
    """Split a sequence into ``parts`` contiguous, balanced sub-lists."""
    n = len(items)
    out: list[list[T]] = []
    for i in range(parts):
        start, end = contiguous_block(n, parts, i)
        out.append(list(items[start:end]))
    return out


def node_shard(items: Sequence[T], num_nodes: int, node_rank: int) -> list[T]:
    """Return the slice of ``items`` owned by node ``node_rank`` of ``num_nodes``."""
    if num_nodes < 1:
        raise ValueError("num_nodes must be >= 1")
    if not (0 <= node_rank < num_nodes):
        raise ValueError(f"node_rank {node_rank} out of range for num_nodes={num_nodes}")
    start, end = contiguous_block(len(items), num_nodes, node_rank)
    return list(items[start:end])


@dataclass(frozen=True, slots=True)
class GpuShard:
    """One GPU's work assignment within a node."""

    gpu_id: str  # e.g. "0", "1", "cuda:0"
    rank: int  # 0-based position within this node's GPU list
    items: list  # frames assigned to this GPU


def gpu_shards(items: Sequence[T], gpu_ids: Sequence[str]) -> list[GpuShard]:
    """Split a node's items across its GPUs, one ``GpuShard`` per GPU."""
    if not gpu_ids:
        raise ValueError("gpu_ids must be non-empty")
    blocks = split_contiguous(items, len(gpu_ids))
    return [
        GpuShard(gpu_id=gpu_id, rank=rank, items=block)
        for rank, (gpu_id, block) in enumerate(zip(gpu_ids, blocks))
    ]


def plan_shards(
    items: Sequence[T],
    *,
    gpu_ids: Sequence[str],
    num_nodes: int = 1,
    node_rank: int = 0,
) -> list[GpuShard]:
    """Full two-level plan for THIS node: node slice, then per-GPU shards.

    Returns the list of ``GpuShard`` this node's processes should run. Combining
    the results of ``plan_shards`` over every ``node_rank`` reproduces the whole
    input exactly once.
    """
    mine = node_shard(items, num_nodes, node_rank)
    return gpu_shards(mine, gpu_ids)


def normalize_device(gpu_id: str) -> str:
    """Map a GPU id to a torch device string: ``"0" -> "cuda:0"``, passthrough else."""
    g = gpu_id.strip()
    if g.isdigit():
        return f"cuda:{g}"
    return g


def parse_gpu_ids(spec: str | Sequence[str] | None) -> list[str]:
    """Parse a ``--gpus`` spec into a list of GPU id strings.

    Accepts ``"0,1,2,3"``, ``"cuda:0,cuda:1"``, a list, or ``None``/``""``
    (single implicit device -> empty list, caller decides default). Does NOT
    query hardware (kept pure); ``"all"`` is resolved elsewhere where torch is
    available.
    """
    if spec is None:
        return []
    if isinstance(spec, (list, tuple)):
        return [str(x).strip() for x in spec if str(x).strip()]
    raw = spec.strip()
    if not raw or raw.lower() == "all":
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]
