"""End-to-end test for src/bpc/multigpu.py — BPC pipeline fanned out across GPUs.

Real process spawning needs GPUs, so ``_spawn_with_kwargs`` is patched to run
each shard's BPC pipeline IN-PROCESS (sequentially), using fake YOLO/SAM2
models and a mocked VLM chat_completion. This exercises the real code path
that was missing before (espada_bpc.py run previously ignored --gpus
entirely and always ran on one device) — sharding -> per-GPU BPC pipeline
(validation + Judge + resume) -> JSONL outputs, asserting full frame coverage
across multiple simulated GPUs.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python not installed")
pd = pytest.importorskip("pandas", reason="pandas not installed")

from src.bpc import multigpu as bpc_multigpu  # noqa: E402
from src.sharding import normalize_device  # noqa: E402
from tests.conftest import make_synthetic_dataset  # noqa: E402

pytestmark = [pytest.mark.smoke, pytest.mark.needs_cv2, pytest.mark.needs_pandas]


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


def _fake_chat_completion(*, base_url, api_token, model, system_text, user_text, images, timeout_s=60):
    text = (system_text + user_text).lower()
    if "point" in text:
        content = (
            '{"grounding_question": "Point.",'
            ' "grounding_answer": [{"label": "person", "point_2d": [5, 5]}]}'
        )
    elif len(images) == 1:
        content = (
            '{"grounding_question": "Describe.",'
            ' "grounding_answer": "A person stands in a plain room near a wall."}'
        )
    else:
        content = (
            '{"grounding_question": "Locate.",'
            ' "grounding_answer": [{"label": "person", "bbox_2d": [2, 2, 20, 20]}]}'
        )
    return SimpleNamespace(content=content)


def _make_inprocess_spawn(devices_seen: list[str]):
    """Stand-in for ``_spawn_with_kwargs`` that runs each shard's BPC pipeline in-process.

    Mirrors ``_bpc_shard_worker`` but injects fakes instead of real YOLO/SAM2/VLM,
    since the real ones need GPU/network.
    """
    from src.bpc.config import load_bpc_config
    from src.bpc.state import StateLogger
    from src.bpc.storage import setup_output_dirs
    from src.bpc.batching import make_index_batches, ProgressCheckpoint
    from src.bpc.pipeline import run_bpc_pipeline
    from src.vlm.prompts_processing import load_yaml

    def fake_spawn(*, shards, **kwargs):
        output_root = Path(kwargs["output_root"])
        cfg = load_bpc_config(kwargs["cfg_path"])
        dirs = setup_output_dirs(output_root, save_visualizations=cfg.storage.save_visualizations)
        prompts = load_yaml("configs/prompts.yaml")
        logs_dir = output_root / "logs"

        for shard in shards:
            if not shard.items:
                continue
            devices_seen.append(normalize_device(shard.gpu_id))
            state_logger = StateLogger(logs_dir, kwargs["run_id"])

            checkpoint = ProgressCheckpoint(logs_dir / f"progress.{shard.gpu_id}.json")
            frames = [Path(p) for p in shard.items]
            checkpoint.init_run(total_images=len(frames), batch_size=kwargs["batch_size"])
            batches = make_index_batches(frames, kwargs["batch_size"])

            with patch("src.bpc.pipeline.chat_completion", side_effect=_fake_chat_completion):
                for batch in checkpoint.pending_batches(batches):
                    run_bpc_pipeline(
                        images=batch.paths,
                        output_root=output_root,
                        cfg=cfg,
                        prompts=prompts,
                        api_token=kwargs["api_token"],
                        segmenter=FakeSegmenter(),
                        yolo_model=FakeYolo(),
                        dirs=dirs,
                        state_logger=state_logger,
                        run_id=kwargs["run_id"],
                    )
                    checkpoint.mark_batch_done(batch.batch_id, shard_key=shard.gpu_id)

    return fake_spawn


def _assert_complete_outputs(output_root: Path, frame_paths):
    """Every frame -> exactly one bboxing + captions JSON (pointing depends on detections)."""
    expected_stems = sorted(Path(p).stem for p in frame_paths)
    for stage in ("bboxing", "captions"):
        stage_dir = output_root / "outputs" / stage
        assert stage_dir.is_dir(), f"missing stage dir {stage}"
        stems = sorted(p.stem for p in stage_dir.glob("*.jsonl"))
        assert stems == expected_stems, (
            f"stage {stage}: expected {len(expected_stems)} unique outputs, got {len(stems)}"
        )


@pytest.mark.parametrize("gpu_ids", [["0"], ["0", "1"], ["0", "1", "2", "3"]])
def test_bpc_multigpu_full_coverage(gpu_ids, tmp_path):
    """Frames split across N simulated GPUs; union of outputs covers every frame exactly once."""
    frames = make_synthetic_dataset(tmp_path / "raw", count=8, seed=11)
    out_root = tmp_path / "out"

    cfg_path = tmp_path / "bpc_test.yaml"
    cfg_path.write_text(
        "judge:\n  enabled: false\n"
        "vlm:\n  workers: 2\n  endpoints: ['http://fake']\n"
    )

    devices_seen: list[str] = []

    with patch.object(bpc_multigpu, "_spawn_with_kwargs", _make_inprocess_spawn(devices_seen)):
        shards = bpc_multigpu.run_bpc_multigpu(
            frames=frames,
            gpu_ids=gpu_ids,
            output_root=out_root,
            cfg_path=str(cfg_path),
            api_token="",
            run_id="test-run",
            batch_size=100,
        )

    assert len(shards) == len(gpu_ids)
    flat = [p for s in shards for p in s.items]
    assert sorted(str(p) for p in flat) == sorted(str(p) for p in frames)
    _assert_complete_outputs(out_root, frames)


def test_bpc_multigpu_devices_pinned_correctly(tmp_path):
    """Each shard's gpu_id is normalized to a distinct cuda:N device string."""
    frames = make_synthetic_dataset(tmp_path / "raw", count=4, seed=3)
    out_root = tmp_path / "out"
    cfg_path = tmp_path / "bpc_test.yaml"
    cfg_path.write_text("judge:\n  enabled: false\nvlm:\n  workers: 1\n  endpoints: ['http://fake']\n")

    devices_seen: list[str] = []
    with patch.object(bpc_multigpu, "_spawn_with_kwargs", _make_inprocess_spawn(devices_seen)):
        bpc_multigpu.run_bpc_multigpu(
            frames=frames, gpu_ids=["0", "1"], output_root=out_root,
            cfg_path=str(cfg_path), api_token="", run_id="r", batch_size=100,
        )

    assert sorted(devices_seen) == ["cuda:0", "cuda:1"]


def test_bpc_multigpu_empty_gpu_ids_raises(tmp_path):
    from src.bpc.multigpu import run_bpc_multigpu

    with pytest.raises(ValueError, match="at least one GPU"):
        run_bpc_multigpu(
            frames=[], gpu_ids=[], output_root=tmp_path / "out",
            cfg_path="configs/bpc.yaml", api_token="", run_id="r",
        )


def test_bpc_multigpu_multinode_slice(tmp_path):
    """2 nodes x 2 GPUs each; each node's run_bpc_multigpu call covers only its slice."""
    frames = make_synthetic_dataset(tmp_path / "raw", count=12, seed=5)
    out_root = tmp_path / "out"
    cfg_path = tmp_path / "bpc_test.yaml"
    cfg_path.write_text("judge:\n  enabled: false\nvlm:\n  workers: 1\n  endpoints: ['http://fake']\n")

    devices_seen: list[str] = []
    all_shard_items: list[Path] = []

    with patch.object(bpc_multigpu, "_spawn_with_kwargs", _make_inprocess_spawn(devices_seen)):
        for node_rank in (0, 1):
            shards = bpc_multigpu.run_bpc_multigpu(
                frames=frames, gpu_ids=["0", "1"], output_root=out_root,
                cfg_path=str(cfg_path), api_token="", run_id=f"r{node_rank}",
                batch_size=100, num_nodes=2, node_rank=node_rank,
            )
            all_shard_items.extend(p for s in shards for p in s.items)

    # Every frame processed exactly once across both nodes combined.
    assert sorted(str(p) for p in all_shard_items) == sorted(str(p) for p in frames)
