"""CLI-level routing tests for runner.main.

Verifies flag wiring and the single-process vs multi-GPU/multi-node decision
without running real models. ``init_models``, ``run_pipeline_with_queues`` and
``run_multigpu`` are patched; ``_collect_input_images`` and ``resolve_gpu_ids``
are stubbed so no GPU or disk image work happens.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("click")
from click.testing import CliRunner  # noqa: E402

import runner  # noqa: E402  (torch/etc stubbed in root conftest)

pytestmark = pytest.mark.smoke


def _common_patches(tmp_path, gpu_ids):
    frames = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    return [
        patch.object(runner, "load_dotenv", lambda: None),
        patch.object(runner, "_require_env", lambda name, required=True: "x"),
        patch.object(
            runner,
            "load_yaml",
            side_effect=lambda p: (
                {
                    "resize": {"target_size": [8, 8], "jpeg_quality": 85, "progressive": True,
                               "supported_extensions": [".jpg"]},
                    "pipeline": {"raw_images_dir": str(tmp_path), "processed_images_dir": str(tmp_path)},
                    "models": {"vlm_workers": 4, "yolo_batch_size": 2},
                }
                if "config" in str(p)
                else {"vlm": {}}
            ),
        ),
        patch.object(runner, "_collect_input_images", lambda **kw: frames),
        patch("src.multigpu.resolve_gpu_ids", lambda spec: list(gpu_ids)),
    ]


def _run(args, patches):
    runner_cm = CliRunner()
    for p in patches:
        p.start()
    try:
        return runner_cm.invoke(runner.main, args, catch_exceptions=False)
    finally:
        for p in reversed(patches):
            p.stop()


def test_single_gpu_uses_inprocess_pipeline(tmp_path):
    patches = _common_patches(tmp_path, gpu_ids=["0"])
    with patch.object(runner, "init_models", return_value=("seg", "yolo")) as im, patch.object(
        runner, "run_pipeline_with_queues"
    ) as rp, patch("src.multigpu.run_multigpu") as rmg:
        patches2 = patches + [
            # keep these active during invoke
        ]
        res = _run([], patches2)
        # call the heavy fns are referenced after patches started; assert via mocks
    assert res.exit_code == 0, res.output
    im.assert_called_once()
    rp.assert_called_once()
    rmg.assert_not_called()


def test_multi_gpu_uses_multigpu(tmp_path):
    patches = _common_patches(tmp_path, gpu_ids=["0", "1"])
    with patch.object(runner, "init_models", return_value=("seg", "yolo")) as im, patch.object(
        runner, "run_pipeline_with_queues"
    ) as rp, patch("src.multigpu.run_multigpu") as rmg:
        res = _run([], patches)
    assert res.exit_code == 0, res.output
    rmg.assert_called_once()
    rp.assert_not_called()
    im.assert_not_called()  # models are built inside each spawned process


def test_multinode_single_gpu_still_uses_multigpu(tmp_path):
    """--num-nodes 2 forces the multi-GPU path even with one GPU id."""
    patches = _common_patches(tmp_path, gpu_ids=["0"])
    with patch.object(runner, "run_pipeline_with_queues") as rp, patch(
        "src.multigpu.run_multigpu"
    ) as rmg:
        res = _run(["--num-nodes", "2", "--node-rank", "1"], patches)
    assert res.exit_code == 0, res.output
    rmg.assert_called_once()
    assert rmg.call_args.kwargs["num_nodes"] == 2
    assert rmg.call_args.kwargs["node_rank"] == 1
    rp.assert_not_called()


def test_vlm_workers_cli_overrides_config(tmp_path):
    patches = _common_patches(tmp_path, gpu_ids=["0"])
    with patch.object(runner, "init_models", return_value=("seg", "yolo")), patch.object(
        runner, "run_pipeline_with_queues"
    ) as rp, patch("src.multigpu.run_multigpu"):
        res = _run(["--vlm-workers", "99"], patches)
    assert res.exit_code == 0, res.output
    assert rp.call_args.kwargs["vlm_workers_number"] == 99


def test_output_dir_flag_threaded(tmp_path):
    patches = _common_patches(tmp_path, gpu_ids=["0"])
    with patch.object(runner, "init_models", return_value=("seg", "yolo")), patch.object(
        runner, "run_pipeline_with_queues"
    ) as rp, patch("src.multigpu.run_multigpu"):
        res = _run(["--output-dir", "/tmp/custom_out"], patches)
    assert res.exit_code == 0, res.output
    assert rp.call_args.kwargs["output_dir"] == Path("/tmp/custom_out")


def test_invalid_node_rank_rejected(tmp_path):
    patches = _common_patches(tmp_path, gpu_ids=["0"])
    # _run uses catch_exceptions=False, so the validation error propagates.
    with pytest.raises(ValueError, match="node-rank"):
        _run(["--num-nodes", "2", "--node-rank", "5"], patches)
