#!/usr/bin/env python3
"""Create a tiny LeRobot v3.0 LIBERO-shaped dataset for local smoke tests.

Features match HuggingFaceVLA/libero (v3.0):
  observation.images.image  (256, 256, 3)  # agentview
  observation.images.image2 (256, 256, 3)  # wrist
  observation.state          (8,)
  action                    (7,)
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

FEATURES = {
    "observation.images.image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.image2": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": ["state"],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["actions"],  # match HuggingFaceVLA/libero
    },
}

TASKS = [
    "pick up the alphabet soup and place it in the basket",
    "put the cream cheese in the bowl",
    "open the middle drawer of the cabinet",
    "turn on the stove",
    "place the wine bottle on the rack",
]


def _frame(ep: int, t: int, n_frames: int, rng: np.random.Generator) -> dict:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:, :, 0] = 30 + (ep * 35) % 200
    image[:, :, 1] = 70 + (t * 3) % 160
    image[:, :, 2] = 110
    image[t % 256, :, :] = 255

    image2 = np.zeros((256, 256, 3), dtype=np.uint8)
    image2[:, :, 1] = 40 + (ep * 25) % 180
    image2[:, :, 2] = 100 + t % 120
    image2[:, t % 256, :] = 255

    state = rng.normal(0.0, 0.1, size=(8,)).astype(np.float32)
    state[0] = float(t) / float(n_frames)
    action = rng.normal(0.0, 0.05, size=(7,)).astype(np.float32)
    # Match common LIBERO gripper convention in demos: roughly {-1, +1}.
    action[-1] = 1.0 if t > n_frames // 2 else -1.0

    return {
        "observation.images.image": image,
        "observation.images.image2": image2,
        "observation.state": state,
        "action": action,
        "task": TASKS[ep % len(TASKS)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/smoke/libero_mini"),
        help="Output dataset root (LeRobot root).",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--frames", type=int, default=16, help="Frames per episode.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repo-id", default="local/libero-mini-smoke")
    args = parser.parse_args()

    out = args.out.expanduser().resolve()
    if out.exists():
        shutil.rmtree(out)

    rng = np.random.default_rng(args.seed)
    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=FEATURES,
        root=out,
        robot_type="panda",
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=2,
    )

    for ep in range(args.episodes):
        for t in range(args.frames):
            ds.add_frame(_frame(ep, t, args.frames, rng))
        ds.save_episode()
        print(f"[ok] episode {ep}: {args.frames} frames / task={TASKS[ep % len(TASKS)]!r}")

    ds.finalize()

    info = out / "meta" / "info.json"
    print(f"[done] dataset at {out}")
    print(f"       meta: {info} exists={info.is_file()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
