#!/usr/bin/env python3
"""Download FlagEval/ERQA from Hugging Face and convert to lmms-eval layout.

lmms-eval ERQA task expects:
  <ERQA_DATA_DIR>/erqa.jsonl
  <ERQA_DATA_DIR>/images/erqa_XXXXXX_YY.png

Source (images already in parquet, no TFRecord/TensorFlow needed):
  https://huggingface.co/datasets/FlagEval/ERQA

Example:
  python3 scripts/prepare_erqa.py
  python3 scripts/prepare_erqa.py --output-dir data/erqa_converted --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "erqa_converted"
HF_REPO = "FlagEval/ERQA"


def _answer_letter(raw: Any) -> str:
    s = str(raw).strip().upper()
    if not s:
        return ""
    for ch in s:
        if ch in "ABCD":
            return ch
    return s[:1]


def _save_rgb(image: Any, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image.convert("RGB").save(path, format="PNG")


def convert_flageval(output_dir: Path, *, force: bool = False) -> int:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "datasets required: uv pip install --break-system-packages datasets pillow"
        ) from exc

    output_dir = output_dir.expanduser().resolve()
    images_dir = output_dir / "images"
    jsonl_path = output_dir / "erqa.jsonl"

    if jsonl_path.is_file() and images_dir.is_dir() and not force:
        n_lines = sum(1 for _ in jsonl_path.open(encoding="utf-8"))
        n_imgs = sum(1 for _ in images_dir.rglob("*.png"))
        print(f"[erqa] already present: {jsonl_path} ({n_lines} rows, {n_imgs} png)")
        print(f"[erqa] export ERQA_DATA_DIR={output_dir}")
        print("[erqa] re-run with --force to overwrite")
        return n_lines

    print(f"[erqa] loading {HF_REPO} (split=test) …")
    ds = load_dataset(HF_REPO, split="test")
    output_dir.mkdir(parents=True, exist_ok=True)
    if force and images_dir.exists():
        for p in images_dir.rglob("*"):
            if p.is_file():
                p.unlink()
    images_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(ds):
        example_id = f"erqa_{index:06d}"
        images = sample.get("images") or []
        if not isinstance(images, (list, tuple)):
            images = [images]

        image_paths: list[str] = []
        for image_index, image in enumerate(images):
            rel = Path("images") / f"{example_id}_{image_index:02d}.png"
            _save_rgb(image, output_dir / rel)
            image_paths.append(rel.as_posix())

        visual_indices = sample.get("visual_indices")
        if visual_indices is None:
            visual_indices = list(range(len(image_paths)))
        else:
            visual_indices = [int(v) for v in list(visual_indices)]

        qtype = sample.get("question_type") or ""
        if isinstance(qtype, (list, tuple)):
            qtype = qtype[0] if qtype else ""

        rows.append(
            {
                "id": example_id,
                "question": str(sample.get("question") or ""),
                "answer": _answer_letter(sample.get("answer")),
                "question_type": str(qtype),
                "visual_indices": visual_indices,
                "image_paths": image_paths,
                # keep HF id for debugging; lmms-eval ignores unknown keys
                "source_question_id": str(sample.get("question_id") or ""),
            }
        )
        if (index + 1) % 50 == 0 or index + 1 == len(ds):
            print(f"[erqa] converted {index + 1}/{len(ds)}")

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[erqa] wrote {len(rows)} examples → {output_dir}")
    print(f"[erqa]   JSONL: {jsonl_path}")
    print(f"[erqa]   Images: {images_dir}")
    print(f"[erqa] export ERQA_DATA_DIR={output_dir}")
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Converted root (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing erqa.jsonl / images/",
    )
    args = parser.parse_args(argv)
    count = convert_flageval(args.output_dir, force=args.force)
    if count <= 0:
        print("[erqa] ERROR: no examples written", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
