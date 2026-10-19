"""Unit tests for multi-GPU orchestration (Stage 2b).

We never actually spawn processes here (that needs real torch + GPUs). Instead
we patch the spawn boundary and the heavy ``runner`` module, and verify:
  * GPU id resolution ("all" via torch, explicit lists, no-GPU fallback),
  * that ``run_multigpu`` plans the right shards and forwards them to spawn,
  * that the per-shard worker pins CUDA_VISIBLE_DEVICES and runs the pipeline
    on its own shard with the right device.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src import multigpu
from src.sharding import GpuShard

pytestmark = pytest.mark.unit


# --- resolve_gpu_ids -------------------------------------------------------


def test_resolve_explicit_list():
    assert multigpu.resolve_gpu_ids("0,1,2") == ["0", "1", "2"]


def test_resolve_all_uses_torch_device_count():
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(device_count=lambda: 4)
    )
    with patch.dict(sys.modules, {"torch": fake_torch}):
        assert multigpu.resolve_gpu_ids("all") == ["0", "1", "2", "3"]


def test_resolve_empty_no_gpu():
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(device_count=lambda: 0)
    )
    with patch.dict(sys.modules, {"torch": fake_torch}):
        assert multigpu.resolve_gpu_ids(None) == []


# --- run_multigpu plans shards & forwards to spawn -------------------------


def test_run_multigpu_plans_and_spawns():
    frames = [Path(f"/x/{i}.jpg") for i in range(8)]
    captured = {}

    def fake_spawn(**kwargs):
        captured.update(kwargs)

    with patch.object(multigpu, "_spawn_with_kwargs", side_effect=fake_spawn):
        shards = multigpu.run_multigpu(
            frames=frames,
            gpu_ids=["0", "1"],
            prompts={},
            base_url="u",
            api_token="",
            model_vlm="m",
            fail_fast=False,
            vlm_workers_number=2,
            yolo_batch_size=4,
            output_dir=Path("/out"),
            models_cfg={"device": "cuda"},
            num_nodes=1,
            node_rank=0,
        )

    assert [s.gpu_id for s in shards] == ["0", "1"]
    assert shards[0].items == frames[:4]
    assert shards[1].items == frames[4:]
    # forwarded shards match
    assert captured["shards"] == shards
    assert captured["output_dir"] == "/out"
    assert captured["yolo_batch_size"] == 4


def test_run_multigpu_multinode_only_processes_local_slice():
    frames = [Path(f"/x/{i}.jpg") for i in range(8)]
    seen = {}

    with patch.object(
        multigpu, "_spawn_with_kwargs", side_effect=lambda **k: seen.update(k)
    ):
        shards = multigpu.run_multigpu(
            frames=frames,
            gpu_ids=["0", "1"],
            prompts={},
            base_url="u",
            api_token="",
            model_vlm="m",
            fail_fast=False,
            vlm_workers_number=1,
            yolo_batch_size=1,
            output_dir=Path("/out"),
            models_cfg={},
            num_nodes=2,
            node_rank=1,
        )

    # node 1 of 2 owns frames[4:8], split across 2 GPUs -> [4,5] and [6,7]
    flat = [p for s in shards for p in s.items]
    assert flat == frames[4:8]


def test_run_multigpu_requires_gpus():
    with pytest.raises(ValueError, match="at least one GPU"):
        multigpu.run_multigpu(
            frames=[Path("/x/0.jpg")],
            gpu_ids=[],
            prompts={},
            base_url="u",
            api_token="",
            model_vlm="m",
            fail_fast=False,
            vlm_workers_number=1,
            yolo_batch_size=1,
            output_dir=Path("/out"),
            models_cfg={},
        )


# --- _shard_worker: device pinning + pipeline invocation -------------------


def test_shard_worker_pins_device_and_runs(monkeypatch):
    shard0 = GpuShard(gpu_id="0", rank=0, items=[Path("/x/0.jpg"), Path("/x/1.jpg")])
    shard1 = GpuShard(gpu_id="3", rank=1, items=[Path("/x/2.jpg")])
    shards = [shard0, shard1]

    fake_runner = MagicMock()
    fake_runner.init_models.return_value = ("seg", "yolo")
    monkeypatch.setitem(sys.modules, "runner", fake_runner)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    # Run worker for rank 1 (gpu "3").
    multigpu._shard_worker(
        1,
        shards,
        prompts={"p": 1},
        base_url="u",
        api_token="tok",
        model_vlm="m",
        fail_fast=True,
        vlm_workers_number=2,
        yolo_batch_size=8,
        output_dir="/out",
        models_cfg={"device": "cuda", "sam2_dtype": "bfloat16"},
    )

    # GPU 3 pinned -> only visible device, remapped to cuda:0
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "3"
    cfg_used = fake_runner.init_models.call_args.args[0]
    assert cfg_used["device"] == "cuda:0"
    assert cfg_used["yolo_batch_size"] == 8
    assert cfg_used["sam2_dtype"] == "bfloat16"  # other cfg preserved

    # pipeline ran on shard 1's frames only
    kwargs = fake_runner.run_pipeline_with_queues.call_args.kwargs
    assert [str(p) for p in kwargs["saved"]] == ["/x/2.jpg"]
    assert kwargs["output_dir"] == Path("/out")
    assert kwargs["vlm_workers_number"] == 2


def test_shard_worker_empty_shard_is_noop(monkeypatch):
    shards = [GpuShard(gpu_id="0", rank=0, items=[])]
    fake_runner = MagicMock()
    monkeypatch.setitem(sys.modules, "runner", fake_runner)

    multigpu._shard_worker(
        0,
        shards,
        prompts={},
        base_url="u",
        api_token="",
        model_vlm="m",
        fail_fast=False,
        vlm_workers_number=1,
        yolo_batch_size=1,
        output_dir="/out",
        models_cfg={},
    )
    fake_runner.init_models.assert_not_called()
    fake_runner.run_pipeline_with_queues.assert_not_called()


def test_shard_worker_entry_delegates(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        multigpu,
        "_shard_worker",
        lambda rank, shards, **kw: calls.update(rank=rank, shards=shards, kw=kw),
    )
    shards = [GpuShard(gpu_id="0", rank=0, items=[])]
    multigpu._shard_worker_entry(0, shards, {"a": 1})
    assert calls["rank"] == 0
    assert calls["kw"] == {"a": 1}
