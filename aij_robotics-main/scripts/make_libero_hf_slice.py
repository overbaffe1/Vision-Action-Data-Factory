#!/usr/bin/env python3
"""Build a tiny LeRobot v3.0 root from a partial HuggingFaceVLA/libero download.

Uses real PNG-in-parquet frames (not synthetic colors) so smoke covers the
same on-disk layout participants get from the Hub.

Example:
  # After downloading meta/ + data/chunk-000/file-000.parquet into cache/...
  python3 scripts/make_libero_hf_slice.py \\
    --src cache/libero_probe/hf_partial \\
    --out examples/smoke/libero_hf_slice \\
    --episodes 3
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Partial HF download root")
    parser.add_argument("--out", type=Path, default=Path("examples/smoke/libero_hf_slice"))
    parser.add_argument("--episodes", type=int, default=3, help="Keep first N episode indices")
    parser.add_argument("--repo-id", default="HuggingFaceVLA/libero")
    args = parser.parse_args()

    src = args.src.expanduser().resolve()
    out = args.out.expanduser().resolve()
    info_src = src / "meta" / "info.json"
    data_src = src / "data" / "chunk-000" / "file-000.parquet"
    ep_src = src / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    for path, label in ((info_src, "info.json"), (data_src, "data parquet"), (ep_src, "episodes parquet")):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")

    if out.exists():
        shutil.rmtree(out)
    (out / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (out / "data" / "chunk-000").mkdir(parents=True)

    info = json.loads(info_src.read_text(encoding="utf-8"))
    keep = set(range(args.episodes))

    data = pd.read_parquet(data_src)
    data = data[data["episode_index"].isin(keep)].copy()
    if data.empty:
        raise RuntimeError(f"No frames for episodes {sorted(keep)} in {data_src}")
    data = data.reset_index(drop=True)
    # Reindex global frame ids for a self-contained slice.
    data["index"] = range(len(data))
    data["frame_index"] = data.groupby("episode_index").cumcount()
    data.to_parquet(out / "data" / "chunk-000" / "file-000.parquet", index=False)

    episodes = pd.read_parquet(ep_src)
    episodes = episodes[episodes["episode_index"].isin(keep)].copy()
    if len(episodes) != args.episodes:
        raise RuntimeError(
            f"Expected {args.episodes} episode rows after filter, got {len(episodes)}. "
            "Download a data shard that contains those episodes."
        )
    # Point every kept episode at the single local data file and contiguous indices.
    cursor = 0
    lengths = data.groupby("episode_index").size().to_dict()
    rows = []
    for _, row in episodes.sort_values("episode_index").iterrows():
        ep = int(row["episode_index"])
        length = int(lengths[ep])
        new = row.copy()
        new["data/chunk_index"] = 0
        new["data/file_index"] = 0
        new["dataset_from_index"] = cursor
        new["dataset_to_index"] = cursor + length
        new["length"] = length
        new["meta/episodes/chunk_index"] = 0
        new["meta/episodes/file_index"] = 0
        rows.append(new)
        cursor += length
    pd.DataFrame(rows).to_parquet(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)

    # Keep tasks / stats if present (harmless for smoke).
    for name in ("tasks.parquet", "stats.json"):
        src_file = src / "meta" / name
        if src_file.is_file():
            shutil.copy2(src_file, out / "meta" / name)

    info["total_episodes"] = int(args.episodes)
    info["total_frames"] = int(len(data))
    info["splits"] = {"train": f"0:{args.episodes}"}
    # Real Hub info.json advertises video_path even though visual features are dtype=image
    # and videos/ is absent. Null it so loaders do not look for missing MP4s.
    info["video_path"] = None
    # Preserve feature schema exactly (including action names=["actions"]).
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")

    # Marker for humans / docs (LeRobot uses root + this identity from train config).
    (out / "README_SLICE.md").write_text(
        f"HF slice of `{args.repo_id}` with {args.episodes} real episodes "
        f"({len(data)} frames) for local smoke.\n",
        encoding="utf-8",
    )

    print(f"[done] {out}")
    print(f"       episodes={args.episodes} frames={len(data)}")
    print(f"       features={list(info['features'])}")
    print(f"       video_path={info.get('video_path')!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
