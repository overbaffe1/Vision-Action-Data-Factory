"""Exhaustive unit tests for multi-GPU / multi-node sharding logic.

The central correctness property: across every node and every GPU, the union of
all shards reproduces the input EXACTLY ONCE — no overlaps, no drops, order
preserved within the flattened result.
"""

from __future__ import annotations

import pytest

from src.sharding import (
    GpuShard,
    contiguous_block,
    gpu_shards,
    node_shard,
    normalize_device,
    parse_gpu_ids,
    plan_shards,
    split_contiguous,
)

pytestmark = pytest.mark.unit


# --- contiguous_block ------------------------------------------------------


@pytest.mark.parametrize(
    "n,parts,index,expected",
    [
        (10, 2, 0, (0, 5)),
        (10, 2, 1, (5, 10)),
        (10, 3, 0, (0, 3)),
        (10, 3, 1, (3, 6)),
        (10, 3, 2, (6, 10)),  # last block absorbs remainder
        (0, 4, 0, (0, 0)),  # empty input
        (1, 4, 0, (0, 0)),  # fewer items than parts: empty early blocks
        (1, 4, 3, (0, 1)),  # the single item lands in the last block
        (5, 1, 0, (0, 5)),
    ],
)
def test_contiguous_block(n, parts, index, expected):
    assert contiguous_block(n, parts, index) == expected


def test_contiguous_block_covers_range_exactly():
    n, parts = 17, 5
    bounds = [contiguous_block(n, parts, i) for i in range(parts)]
    # blocks are contiguous and cover [0, n)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == n
    for (s1, e1), (s2, e2) in zip(bounds, bounds[1:]):
        assert e1 == s2  # no gap, no overlap


def test_contiguous_block_invalid():
    with pytest.raises(ValueError):
        contiguous_block(10, 0, 0)
    with pytest.raises(ValueError):
        contiguous_block(10, 2, 2)
    with pytest.raises(ValueError):
        contiguous_block(-1, 2, 0)


# --- split_contiguous ------------------------------------------------------


def test_split_contiguous_reassembles():
    items = list(range(23))
    parts = 4
    chunks = split_contiguous(items, parts)
    assert len(chunks) == parts
    # flatten == original, order preserved
    flat = [x for c in chunks for x in c]
    assert flat == items
    # balanced: sizes differ by at most 1
    sizes = [len(c) for c in chunks]
    assert max(sizes) - min(sizes) <= 1


def test_split_contiguous_more_parts_than_items():
    chunks = split_contiguous([1, 2], 5)
    # exactly 2 non-empty blocks of size 1; the rest empty; order preserved.
    assert sum(len(c) for c in chunks) == 2
    assert sorted(len(c) for c in chunks) == [0, 0, 0, 1, 1]
    assert [x for c in chunks for x in c] == [1, 2]


def test_split_contiguous_empty():
    assert split_contiguous([], 3) == [[], [], []]


# --- node_shard ------------------------------------------------------------


def test_node_shard_partitions():
    items = list(range(20))
    collected: list[int] = []
    for r in range(4):
        collected.extend(node_shard(items, 4, r))
    assert collected == items  # every item exactly once, in order


def test_node_shard_single_node_is_identity():
    items = list(range(7))
    assert node_shard(items, 1, 0) == items


def test_node_shard_invalid():
    with pytest.raises(ValueError):
        node_shard([1], 0, 0)
    with pytest.raises(ValueError):
        node_shard([1], 2, 2)


# --- gpu_shards ------------------------------------------------------------


def test_gpu_shards_basic():
    shards = gpu_shards(list(range(10)), ["0", "1", "2"])
    assert [s.gpu_id for s in shards] == ["0", "1", "2"]
    assert [s.rank for s in shards] == [0, 1, 2]
    flat = [x for s in shards for x in s.items]
    assert flat == list(range(10))


def test_gpu_shards_empty_gpu_list():
    with pytest.raises(ValueError):
        gpu_shards([1, 2], [])


def test_gpu_shards_more_gpus_than_items():
    shards = gpu_shards([42], ["0", "1", "2"])
    # one GPU gets the item, others get empty shards (still scheduled)
    nonempty = [s for s in shards if s.items]
    assert len(nonempty) == 1
    assert nonempty[0].items == [42]


# --- plan_shards: the full property ----------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2, 7, 16, 100, 101])
@pytest.mark.parametrize("num_nodes", [1, 2, 4])
@pytest.mark.parametrize("gpus", [["0"], ["0", "1"], ["0", "1", "2", "3"]])
def test_plan_shards_exact_partition(n, num_nodes, gpus):
    """Union over all nodes & GPUs == input exactly once, no overlap/drop."""
    items = list(range(n))
    seen: list[int] = []
    for node_rank in range(num_nodes):
        shards = plan_shards(items, gpu_ids=gpus, num_nodes=num_nodes, node_rank=node_rank)
        # every node returns exactly one shard per GPU
        assert len(shards) == len(gpus)
        for s in shards:
            seen.extend(s.items)
    assert sorted(seen) == items  # coverage
    assert len(seen) == len(set(seen))  # disjoint


def test_plan_shards_two_nodes_two_gpus_layout():
    items = list(range(8))
    node0 = plan_shards(items, gpu_ids=["0", "1"], num_nodes=2, node_rank=0)
    node1 = plan_shards(items, gpu_ids=["0", "1"], num_nodes=2, node_rank=1)
    assert node0[0].items == [0, 1]
    assert node0[1].items == [2, 3]
    assert node1[0].items == [4, 5]
    assert node1[1].items == [6, 7]


def test_plan_shards_returns_gpushard_type():
    shards = plan_shards([1, 2, 3], gpu_ids=["0"])
    assert all(isinstance(s, GpuShard) for s in shards)


# --- normalize_device ------------------------------------------------------


@pytest.mark.parametrize(
    "gpu_id,expected",
    [
        ("0", "cuda:0"),
        ("3", "cuda:3"),
        ("cuda:1", "cuda:1"),
        ("cpu", "cpu"),
        (" 2 ", "cuda:2"),
    ],
)
def test_normalize_device(gpu_id, expected):
    assert normalize_device(gpu_id) == expected


# --- parse_gpu_ids ---------------------------------------------------------


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("0,1,2,3", ["0", "1", "2", "3"]),
        ("cuda:0,cuda:1", ["cuda:0", "cuda:1"]),
        ("0, 1 ,2", ["0", "1", "2"]),
        ("", []),
        ("all", []),
        ("ALL", []),
        (None, []),
        (["0", "1"], ["0", "1"]),
        ([0, 1], ["0", "1"]),
    ],
)
def test_parse_gpu_ids(spec, expected):
    assert parse_gpu_ids(spec) == expected
