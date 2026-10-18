"""End-to-end tests for multi-GPU / multi-node execution on generated data.

Real process spawning needs GPUs, so we emulate it: ``_spawn_with_kwargs`` is
patched to run each shard's pipeline IN-PROCESS (sequentially) using the real
``run_pipeline_with_queues`` with fake YOLO/SAM2 models and a mocked VLM. This
exercises the complete path — sharding -> per-shard pipeline -> JSONL output —
and asserts the core correctness property:

    every input frame produces exactly one bboxing + caption + pointing JSON,
    with NO duplicates and NO drops, whether split across 1 GPU, many GPUs, or
    many nodes.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

cv2 = pytest.importorskip("cv2", reason="opencv-python not installed")
pd = pytest.importorskip("pandas", reason="pandas not installed")

from src import multigpu  # noqa: E402
from src.sharding import normalize_device  # noqa: E402
from src.vlm.prompts_processing import load_yaml  # noqa: E402
from tests.conftest import make_synthetic_dataset  # noqa: E402

pytestmark = [pytest.mark.smoke, pytest.mark.needs_cv2, pytest.mark.needs_pandas]

_PROMPTS = load_yaml(Path(__file__).resolve().parents[2] / "configs" / "prompts.yaml")


# --- fakes (shared with single-node smoke) ---------------------------------


def _one_detection_df(image: np.ndarray) -> pd.DataFrame:
    h, w = image.shape[:2]
    return pd.DataFrame(
        [
            {
                "class": "person",
                "class_id": 0,
                "conf": 0.9,
                "x1": 2.0,
                "y1": 2.0,
                "x2": float(w // 4),
                "y2": float(h // 4),
            }
        ]
    )


class FakeYolo:
    def predict(self, image):
        return _one_detection_df(image)

    def predict_on_batch(self, images):
        return [_one_detection_df(img) for img in images]


class FakeSegmenter:
    def __init__(self):
        self._size = (0, 0)

    def set_image(self, pil_image):
        self._size = (pil_image.height, pil_image.width)

    def predict_from_boxes(self, boxes, multimask_output=False):
        h, w = self._size
        masks = np.zeros((len(boxes), h, w), dtype=bool)
        for i in range(len(boxes)):
            masks[i, 2 : max(3, h // 4), 2 : max(3, w // 4)] = True
        return SimpleNamespace(masks=masks)


def _fake_chat_completion(*, base_url, api_token, model, system_text, user_text, images):
    if len(images) == 1:
        content = (
            '{"grounding_question": "Describe.",'
            ' "grounding_answer": "A person stands in a plain room near a wall."}'
        )
    elif "point" in (system_text + user_text).lower():
        content = (
            '{"grounding_question": "Point.",'
            ' "grounding_answer": [{"label": "person", "point_2d": [5, 5]}]}'
        )
    else:
        content = (
            '{"grounding_question": "Locate.",'
            ' "grounding_answer": [{"label": "person", "bbox_2d": [2, 2, 20, 20]}]}'
        )
    return SimpleNamespace(content=content)


def _make_inprocess_spawn(output_dir: Path, devices_seen: list[str]):
    """Return a stand-in for ``_spawn_with_kwargs`` that runs shards in-process.

    Each shard runs the REAL ``run_pipeline_with_queues`` with fake models, so
    the full per-shard pipeline (incl. JSON writing to ``output_dir``) executes.
    """
    from runner import run_pipeline_with_queues

    def fake_spawn(*, shards, **kwargs):
        for shard in shards:
            if not shard.items:
                continue
            devices_seen.append(normalize_device(shard.gpu_id))
            with patch("src.workers.chat_completion", side_effect=_fake_chat_completion):
                run_pipeline_with_queues(
                    saved=[Path(p) for p in shard.items],
                    segmenter=FakeSegmenter(),
                    yolo_model=FakeYolo(),
                    prompts=kwargs["prompts"],
                    base_url=kwargs["base_url"],
                    api_token=kwargs["api_token"],
                    model_vlm=kwargs["model_vlm"],
                    fail_fast=kwargs["fail_fast"],
                    vlm_workers_number=kwargs["vlm_workers_number"],
                    yolo_batch_size=kwargs["yolo_batch_size"],
                    output_dir=Path(kwargs["output_dir"]),
                )

    return fake_spawn


def _assert_complete_outputs(output_dir: Path, frame_paths):
    """Every frame -> exactly one JSON per stage, no extras, no missing."""
    expected_stems = sorted(Path(p).stem for p in frame_paths)
    for stage in ("bboxing", "captions", "pointing"):
        stage_dir = output_dir / stage
        assert stage_dir.is_dir(), f"missing stage dir {stage}"
        stems = sorted(p.stem for p in stage_dir.glob("*.jsonl"))
        assert stems == expected_stems, (
            f"stage {stage}: expected {len(expected_stems)} unique outputs, got {len(stems)}"
        )


# --- the tests -------------------------------------------------------------


def _runner_module():
    import runner

    return runner


@pytest.mark.parametrize("gpu_ids", [["0"], ["0", "1"], ["0", "1", "2", "3"]])
def test_multigpu_single_node_full_coverage(gpu_ids, tmp_path):
    frames = make_synthetic_dataset(tmp_path / "raw", count=10, seed=7)
    out_dir = tmp_path / "outputs"
    devices_seen: list[str] = []

    with patch.object(multigpu, "_spawn_with_kwargs", _make_inprocess_spawn(out_dir, devices_seen)):
        shards = multigpu.run_multigpu(
            frames=frames,
            gpu_ids=gpu_ids,
            prompts=_PROMPTS,
            base_url="http://fake",
            api_token="",
            model_vlm="m",
            fail_fast=True,
            vlm_workers_number=2,
            yolo_batch_size=1,
            output_dir=out_dir,
            models_cfg={},
        )

    # one shard per GPU; union == all frames exactly once
    assert len(shards) == len(gpu_ids)
    flat = [p for s in shards for p in s.items]
    assert sorted(str(p) for p in flat) == sorted(str(p) for p in frames)
    _assert_complete_outputs(out_dir, frames)


def test_multigpu_batched_path(tmp_path):
    frames = make_synthetic_dataset(tmp_path / "raw", count=9, seed=3)
    out_dir = tmp_path / "outputs"
    devices_seen: list[str] = []

    with patch.object(multigpu, "_spawn_with_kwargs", _make_inprocess_spawn(out_dir, devices_seen)):
        multigpu.run_multigpu(
            frames=frames,
            gpu_ids=["0", "1"],
            prompts=_PROMPTS,
            base_url="http://fake",
            api_token="",
            model_vlm="m",
            fail_fast=True,
            vlm_workers_number=2,
            yolo_batch_size=4,  # batched YOLO worker
            output_dir=out_dir,
            models_cfg={},
        )

    _assert_complete_outputs(out_dir, frames)


def test_multinode_union_covers_dataset_once(tmp_path):
    """Simulate 2 nodes x 2 GPUs: each node runs its slice into a SHARED output
    dir; together they must cover the dataset exactly once (no dup, no drop)."""
    frames = make_synthetic_dataset(tmp_path / "raw", count=12, seed=11)
    out_dir = tmp_path / "outputs"  # shared across nodes (e.g. NFS)
    num_nodes = 2

    all_flat: list[str] = []
    for node_rank in range(num_nodes):
        devices_seen: list[str] = []
        with patch.object(
            multigpu, "_spawn_with_kwargs", _make_inprocess_spawn(out_dir, devices_seen)
        ):
            shards = multigpu.run_multigpu(
                frames=frames,
                gpu_ids=["0", "1"],
                prompts=_PROMPTS,
                base_url="http://fake",
                api_token="",
                model_vlm="m",
                fail_fast=True,
                vlm_workers_number=2,
                yolo_batch_size=1,
                output_dir=out_dir,
                models_cfg={},
                num_nodes=num_nodes,
                node_rank=node_rank,
            )
        all_flat.extend(str(p) for s in shards for p in s.items)

    # disjoint across nodes + full coverage
    assert sorted(all_flat) == sorted(str(p) for p in frames)
    assert len(all_flat) == len(set(all_flat))
    _assert_complete_outputs(out_dir, frames)


def test_multinode_four_nodes_two_gpus(tmp_path):
    """4 machines x 2 A100 — the user's larger target topology."""
    frames = make_synthetic_dataset(tmp_path / "raw", count=20, seed=5)
    out_dir = tmp_path / "outputs"
    num_nodes = 4

    all_flat: list[str] = []
    for node_rank in range(num_nodes):
        with patch.object(
            multigpu, "_spawn_with_kwargs", _make_inprocess_spawn(out_dir, [])
        ):
            shards = multigpu.run_multigpu(
                frames=frames,
                gpu_ids=["0", "1"],
                prompts=_PROMPTS,
                base_url="http://fake",
                api_token="",
                model_vlm="m",
                fail_fast=True,
                vlm_workers_number=1,
                yolo_batch_size=2,
                output_dir=out_dir,
                models_cfg={},
                num_nodes=num_nodes,
                node_rank=node_rank,
            )
        all_flat.extend(str(p) for s in shards for p in s.items)

    assert sorted(all_flat) == sorted(str(p) for p in frames)
    assert len(all_flat) == len(set(all_flat))
    _assert_complete_outputs(out_dir, frames)


def test_more_gpus_than_frames_is_safe(tmp_path):
    """Edge: 4 GPUs but only 2 frames — empty shards must be skipped cleanly."""
    frames = make_synthetic_dataset(tmp_path / "raw", count=2, seed=1)
    out_dir = tmp_path / "outputs"

    with patch.object(multigpu, "_spawn_with_kwargs", _make_inprocess_spawn(out_dir, [])):
        multigpu.run_multigpu(
            frames=frames,
            gpu_ids=["0", "1", "2", "3"],
            prompts=_PROMPTS,
            base_url="http://fake",
            api_token="",
            model_vlm="m",
            fail_fast=True,
            vlm_workers_number=1,
            yolo_batch_size=1,
            output_dir=out_dir,
            models_cfg={},
        )

    _assert_complete_outputs(out_dir, frames)
